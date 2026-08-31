"""Deterministic LinUCB ranker over choices already approved by safety policy.

This ranker is intentionally stateless.  It reconstructs its small ridge model
from the verified planner history on every call, so the append-only ledger
remains the sole source of truth.  SearchPolicy generates the legal action set;
LinUCB may only reorder that set and cannot invent a parent, method, or family.
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
    """Rank a legal choice set using verified outcomes and uncertainty."""

    def __init__(self, alpha: float = 0.25, ridge: float = 1.0):
        if alpha < 0 or ridge <= 0:
            raise ValueError("LinUCB alpha must be non-negative and ridge positive")
        self.alpha = float(alpha)
        self.ridge = float(ridge)

    def __call__(self, choices: Sequence[Any], context: Any) -> Any:
        if not choices:
            raise ValueError("LinUCB cannot rank an empty legal choice set")
        history = as_list(get_value(context, "family_history", None))
        families = sorted(
            {
                str(get_value(item, "family", ""))
                for item in history
                if str(get_value(item, "family", ""))
            }
            | {
                str(get_value(choice, "family", ""))
                for choice in choices
                if str(get_value(choice, "family", ""))
            }
        )
        methods = sorted(
            {
                str(method_id)
                for item in history
                for method_id in as_list(
                    get_value(item, "method_card_ids", None)
                )
                if str(method_id)
            }
            | {
                str(get_value(choice, "method_card_id", ""))
                for choice in choices
                if str(get_value(choice, "method_card_id", ""))
            }
        )
        # Intercept + family + method + cost + parent advantage.  Method-level
        # evidence prevents a family average from hiding a consistently weak
        # card, while the legal-choice boundary still owns what may be tried.
        dimension = 1 + len(families) + len(methods) + 3 + 1
        matrix = [
            [self.ridge if row == column else 0.0 for column in range(dimension)]
            for row in range(dimension)
        ]
        reward_vector = [0.0] * dimension
        summaries = []
        for field in ("baseline", "family_history"):
            summaries.extend(as_list(get_value(context, field, None)))
        by_id = {
            str(get_value(item, "experiment_id", "")): item
            for item in summaries
            if str(get_value(item, "experiment_id", ""))
        }
        baseline_score = _number(
            get_value(get_value(context, "baseline", None), "primary_score", None)
        )

        for summary in history:
            reward = _number(get_value(summary, "parent_delta", None))
            if (
                reward is None
                or _normalized(get_value(summary, "integrity", None)) != "clean"
                or get_value(summary, "output_accepted", None) is not True
            ):
                continue
            family = str(get_value(summary, "family", ""))
            summary_methods = as_list(get_value(summary, "method_card_ids", None))
            method_id = str(summary_methods[0]) if summary_methods else ""
            parent = by_id.get(str(get_value(summary, "parent_experiment_id", "")))
            features = self._features(
                family,
                method_id,
                _normalized(get_value(summary, "actual_cost", "medium")),
                _number(get_value(parent, "primary_score", None)),
                baseline_score,
                families,
                methods,
            )
            # Full confirmed aggregates are the strongest observations.  Proxy
            # and single-seed evidence remains useful for direction finding but
            # has less leverage, so one noisy observation cannot dominate the
            # learned ordering.
            fidelity = _normalized(
                get_value(summary, "highest_completed_fidelity", "")
            )
            stability = _normalized(get_value(summary, "stability", ""))
            quality = 1.0 if fidelity == "full" else 0.35
            if stability == "single_seed":
                quality *= 0.5
            elif stability == "unstable":
                quality *= 0.25
            clipped_reward = max(-0.1, min(0.1, reward))
            weighted_reward = quality * clipped_reward
            for row in range(dimension):
                reward_vector[row] += weighted_reward * features[row]
                for column in range(dimension):
                    matrix[row][column] += quality * features[row] * features[column]

        inverse = _inverse(matrix)
        theta = _matvec(inverse, reward_vector)
        scored: list[tuple[float, int, Any]] = []
        for index, choice in enumerate(choices):
            parent = get_value(choice, "parent", None)
            features = self._features(
                str(get_value(choice, "family", "")),
                str(get_value(choice, "method_card_id", "")),
                _normalized(get_value(choice, "cost_tier", "medium")),
                _number(get_value(parent, "primary_score", None)),
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
    def _features(
        family: str,
        method_id: str,
        cost_tier: str,
        parent_score: float | None,
        baseline_score: float | None,
        families: Sequence[str],
        methods: Sequence[str],
    ) -> list[float]:
        values = [1.0]
        values.extend(1.0 if family == item else 0.0 for item in families)
        values.extend(1.0 if method_id == item else 0.0 for item in methods)
        values.extend(1.0 if cost_tier == item else 0.0 for item in ("low", "medium", "high"))
        parent_advantage = 0.0
        if parent_score is not None and baseline_score is not None:
            parent_advantage = max(-0.1, min(0.1, parent_score - baseline_score))
        values.append(parent_advantage)
        return values
