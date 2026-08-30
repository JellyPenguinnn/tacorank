"""Deterministic LinUCB ranker over choices already approved by safety policy.

This ranker is intentionally stateless. It reconstructs its small ridge model
from verified current-run and compatibility-filtered prior-run history on every
call, so no mutable policy state can drift from the event record. SearchPolicy
generates the legal action set; LinUCB may only reorder that set and cannot
invent a parent, method, or family.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .graph_view import as_list, enum_value, get_value


def _normalized(value: Any) -> str:
    return str(enum_value(value) or "").strip().lower()


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        list(row) + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("LinUCB ridge matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [row[size:] for row in augmented]


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(value * item for value, item in zip(row, vector)) for row in matrix]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class LinUCBLegalChoiceRanker:
    """Rank legal choices using stable, method-specific outcomes.

    The model is rebuilt from the planner context on every call.  It therefore
    learns from the append-only record without introducing mutable model state,
    and it can use both current-run feedback and compatibility-filtered prior
    runs.  Negative and inconclusive clean outcomes are observations too; a
    single lucky seed is deliberately down-weighted and seed uncertainty is
    charged directly in the reward.
    """

    def __init__(self, alpha: float = 0.25, ridge: float = 1.0):
        if alpha < 0 or ridge <= 0:
            raise ValueError("LinUCB alpha must be non-negative and ridge positive")
        self.alpha = float(alpha)
        self.ridge = float(ridge)

    def __call__(self, choices: Sequence[Any], context: Any) -> Any:
        if not choices:
            raise ValueError("LinUCB cannot rank an empty legal choice set")
        history = as_list(get_value(context, "family_history", None))
        historical = as_list(get_value(context, "historical_feedback", None))
        families = sorted(
            {
                str(get_value(item, "family", ""))
                for item in [*history, *historical]
                if str(get_value(item, "family", ""))
            }
            | {
                str(get_value(choice, "family", ""))
                for choice in choices
                if str(get_value(choice, "family", ""))
            }
        )
        observed_methods = {
            method_id
            for item in [*history, *historical]
            for method_id in self._method_ids(item)
        }
        # Preserve the legacy family-only tie-break until the ledger has a
        # method-bearing observation.  Once it does, include the legal arms so
        # an untried method receives normal UCB exploration credit.
        methods = sorted(
            observed_methods
            | (
                {
                    method_id
                    for choice in choices
                    for method_id in self._method_ids(choice)
                }
                if observed_methods
                else set()
            )
        )
        dimension = 1 + len(families) + len(methods) + 3 + 1
        matrix = [
            [self.ridge if row == column else 0.0 for column in range(dimension)]
            for row in range(dimension)
        ]
        reward_vector = [0.0] * dimension
        summaries = [
            *as_list(get_value(context, "baseline", None)),
            *history,
            *historical,
        ]
        by_id = {
            str(get_value(item, "experiment_id", "")): item
            for item in summaries
            if str(get_value(item, "experiment_id", ""))
        }
        baseline_score = _number(
            get_value(get_value(context, "baseline", None), "primary_score", None)
        )

        for summary in [*history, *historical]:
            reward = self._reward(summary, by_id)
            if (
                reward is None
                or _normalized(get_value(summary, "integrity", None)) != "clean"
                or get_value(summary, "output_accepted", True) is not True
            ):
                continue
            family = str(get_value(summary, "family", ""))
            parent = by_id.get(str(get_value(summary, "parent_experiment_id", "")))
            parent_score = _number(
                get_value(summary, "parent_stable_primary_score", None)
            )
            if parent_score is None:
                parent_score = _number(get_value(parent, "stable_primary_score", None))
            if parent_score is None:
                parent_score = _number(get_value(parent, "primary_score", None))
            features = self._features(
                family,
                self._method_key(summary),
                _normalized(get_value(summary, "actual_cost", "medium")),
                parent_score,
                baseline_score,
                families,
                methods,
            )
            clipped_reward = max(-0.1, min(0.1, reward))
            weight = self._observation_weight(summary)
            for row in range(dimension):
                reward_vector[row] += weight * clipped_reward * features[row]
                for column in range(dimension):
                    matrix[row][column] += weight * features[row] * features[column]

        inverse = _inverse(matrix)
        theta = _matvec(inverse, reward_vector)
        scored: list[tuple[float, int, Any]] = []
        for index, choice in enumerate(choices):
            parent = get_value(choice, "parent", None)
            parent_score = _number(
                get_value(parent, "stable_primary_score", None)
            )
            if parent_score is None:
                parent_score = _number(get_value(parent, "primary_score", None))
            features = self._features(
                str(get_value(choice, "family", "")),
                self._method_key(choice),
                _normalized(get_value(choice, "cost_tier", "medium")),
                parent_score,
                baseline_score,
                families,
                methods,
            )
            projected = _matvec(inverse, features)
            uncertainty = math.sqrt(max(0.0, _dot(features, projected)))
            cost_penalty = {"low": 0.0, "medium": 0.001, "high": 0.003}.get(
                _normalized(get_value(choice, "cost_tier", "medium")), 0.001
            )
            score = _dot(theta, features) + self.alpha * uncertainty - cost_penalty
            scored.append((score, -index, choice))
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def _method_ids(value: Any) -> tuple[str, ...]:
        ids = as_list(get_value(value, "method_card_ids", None))
        normalized = tuple(str(item) for item in ids if str(item))
        if normalized:
            return normalized
        method_id = str(get_value(value, "method_card_id", ""))
        return (method_id,) if method_id else ()

    @classmethod
    def _method_key(cls, value: Any) -> str:
        return "|".join(cls._method_ids(value)) or "<unspecified>"

    @staticmethod
    def _observation_weight(summary: Any) -> float:
        stability = _normalized(get_value(summary, "stability", ""))
        return {
            "confirmed": 1.0,
            "single_seed": 0.35,
            "unstable": 0.10,
            "not_applicable": 0.5,
        }.get(stability, 0.5)

    @classmethod
    def _reward(
        cls,
        summary: Any,
        by_id: dict[str, Any],
    ) -> float | None:
        explicit = _number(get_value(summary, "risk_adjusted_reward", None))
        if explicit is not None:
            return explicit
        child_score = _number(get_value(summary, "stable_primary_score", None))
        parent = by_id.get(str(get_value(summary, "parent_experiment_id", "")))
        parent_score = _number(
            get_value(summary, "parent_stable_primary_score", None)
        )
        if parent_score is None:
            parent_score = _number(get_value(parent, "stable_primary_score", None))
        if parent_score is None:
            parent_score = _number(get_value(parent, "primary_score", None))
        reward = None
        if child_score is not None and parent_score is not None:
            reward = child_score - parent_score
        if reward is None:
            reward = _number(get_value(summary, "parent_delta", None))
        stderr = _number(get_value(summary, "seed_stderr", None))
        if reward is not None and stderr is not None:
            reward -= 2.0 * stderr
        return reward

    @staticmethod
    def _features(
        family: str,
        method: str,
        cost_tier: str,
        parent_score: float | None,
        baseline_score: float | None,
        families: Sequence[str],
        methods: Sequence[str],
    ) -> list[float]:
        values = [1.0]
        values.extend(1.0 if family == item else 0.0 for item in families)
        values.extend(1.0 if method == item else 0.0 for item in methods)
        values.extend(1.0 if cost_tier == item else 0.0 for item in ("low", "medium", "high"))
        parent_advantage = 0.0
        if parent_score is not None and baseline_score is not None:
            parent_advantage = max(-0.1, min(0.1, parent_score - baseline_score))
        values.append(parent_advantage)
        return values
