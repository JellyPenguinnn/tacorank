"""Read-only AIDE graph projection for the research planner.

The durable experiment lineage lives in Git and the event ledger owned by the
orchestrator.  Person 1 only receives a verified ``PlannerContext`` and builds
this in-memory view from it; this module deliberately has no persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


def get_value(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a shared model or a mapping.

    Shared Pydantic models are owned by Person 2 and are not redefined here.
    Supporting mappings keeps the projection easy to unit test before the
    shared schema lands.
    """

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def has_value(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        return name in value
    return hasattr(value, name)


def enum_value(value: Any) -> Any:
    """Return the serialized value for shared-schema enums."""

    return value.value if isinstance(value, Enum) else value


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        # PlannerContext uses scalar summaries for baseline/current_best and
        # collections for frontier/history.  Treat a non-iterable model as a
        # one-item collection so both forms project consistently.
        return [value]


@dataclass(frozen=True)
class ExperimentNodeView:
    """A verified, read-only summary of one experiment."""

    experiment_id: str
    parent_experiment_id: str | None
    parent_commit_sha: str | None
    family: str | None
    hypothesis: str
    trust_verdict: str | None
    stability: str | None
    integrity: str | None
    decision: str | None
    highest_completed_fidelity: str | None
    primary_score: float | None
    child_count: int
    actual_cost: Any
    metric_set: Any
    metric_deltas: Mapping[str, float]
    prediction_change: float | None
    method_card_ids: tuple[str, ...]
    summary: Any

    @property
    def is_root(self) -> bool:
        return self.parent_experiment_id is None

    @property
    def is_trusted(self) -> bool:
        verdict = str(enum_value(self.trust_verdict) or "").lower()
        integrity = str(enum_value(self.integrity) or "").lower()
        return verdict in {"accepted", "verified"} and integrity in {
            None,
            "clean",
            "",
        }

    @property
    def is_parent_eligible(self) -> bool:
        explicit = get_value(self.summary, "parent_eligible", None)
        if explicit is False:
            return False
        status = str(enum_value(get_value(self.summary, "status", ""))).lower()
        if status in {"retracted", "pruned", "invalid", "suspicious", "unstable"}:
            return False
        if str(enum_value(self.decision) or "").lower() in {"reject", "prune", "invalid"}:
            return False
        if not self.is_trusted:
            return False
        if self.is_root:
            # A root is parent-eligible only after the orchestrator has
            # recorded a verified baseline result; a merely named root is not
            # enough to authorize a branch.
            return explicit is True or str(enum_value(self.decision) or "").lower() in {
                "accept",
                "accepted",
                "promote",
                "verified",
            }
        if explicit is True:
            return True
        return str(enum_value(self.highest_completed_fidelity) or "").lower() == "full"

    @classmethod
    def from_summary(cls, summary: Any) -> "ExperimentNodeView | None":
        """Project one verified summary, including non-frontier portfolio nodes."""

        experiment_id = str(get_value(summary, "experiment_id", ""))
        if not experiment_id:
            return None
        score = get_value(summary, "primary_score", None)
        if score is None:
            score = get_value(summary, "highest_primary_score", None)
        if score is None:
            score = get_value(summary, "score", None)
        if score is None:
            score = get_value(get_value(summary, "metric_set", None), "primary_score", None)
        trust = get_value(summary, "trust_verdict", None)
        trust = trust or get_value(summary, "trust", None)
        trust_verdict = enum_value(get_value(trust, "verdict", trust))
        integrity = get_value(summary, "integrity", None)
        integrity = integrity or get_value(trust, "integrity", None)
        stability = get_value(summary, "stability", None)
        stability = stability or get_value(trust, "stability", None)
        metric_deltas = get_value(summary, "metric_deltas", None) or {}
        if not isinstance(metric_deltas, Mapping):
            metric_deltas = {}
        prediction_change = get_value(summary, "prediction_change", None)
        if prediction_change is not None and not isinstance(prediction_change, (int, float)):
            prediction_change = get_value(prediction_change, "changed_row_fraction", None)
        try:
            score = None if score is None else float(score)
        except (TypeError, ValueError):
            score = None
        try:
            child_count = int(get_value(summary, "child_count", 0) or 0)
        except (TypeError, ValueError):
            child_count = 0
        try:
            prediction_change = (
                None if prediction_change is None else float(prediction_change)
            )
        except (TypeError, ValueError):
            prediction_change = None
        return cls(
            experiment_id=experiment_id,
            parent_experiment_id=get_value(summary, "parent_experiment_id", None),
            parent_commit_sha=get_value(summary, "latest_patch_commit_sha", None)
            or get_value(summary, "commit_sha", None)
            or get_value(summary, "latest_commit_sha", None)
            or get_value(summary, "base_commit_sha", None),
            family=get_value(summary, "family", None),
            hypothesis=str(
                get_value(summary, "hypothesis_summary", None)
                or get_value(summary, "hypothesis", "")
            ),
            trust_verdict=trust_verdict,
            stability=enum_value(stability),
            integrity=enum_value(integrity),
            decision=enum_value(get_value(summary, "decision", None)),
            highest_completed_fidelity=enum_value(
                get_value(summary, "highest_completed_fidelity", None)
                or get_value(summary, "highest_fidelity_completed", None)
            ),
            primary_score=score,
            child_count=child_count,
            actual_cost=enum_value(get_value(summary, "actual_cost", None)),
            metric_set=get_value(summary, "metric_set", None),
            metric_deltas={str(key): float(value) for key, value in metric_deltas.items()},
            prediction_change=prediction_change,
            method_card_ids=tuple(
                map(str, as_list(get_value(summary, "method_card_ids", None)))
            ),
            summary=summary,
        )


class GraphView:
    """A transient AIDE view reconstructed from ``PlannerContext``."""

    def __init__(self, nodes: Iterable[ExperimentNodeView]):
        unique: dict[str, ExperimentNodeView] = {}
        for node in nodes:
            if node.experiment_id:
                unique[node.experiment_id] = node
        self._nodes = unique

    @classmethod
    def from_context(cls, context: Any) -> "GraphView":
        # Person 2's typed PlannerContext declares eligible_frontier as the
        # authoritative set. In particular, an explicit empty list must stay
        # empty instead of resurrecting the baseline or stale history nodes.
        if has_value(context, "eligible_frontier"):
            summaries = as_list(get_value(context, "eligible_frontier", None))
        else:
            summaries = []
            for field in ("baseline", "current_best", "family_history"):
                summaries.extend(as_list(get_value(context, field, None)))

        nodes: list[ExperimentNodeView] = []
        for summary in summaries:
            node = ExperimentNodeView.from_summary(summary)
            if node is not None:
                nodes.append(node)
        return cls(nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, experiment_id: str) -> ExperimentNodeView | None:
        return self._nodes.get(experiment_id)

    def nodes(self) -> tuple[ExperimentNodeView, ...]:
        return tuple(self._nodes.values())

    def children_of(self, experiment_id: str) -> tuple[ExperimentNodeView, ...]:
        return tuple(
            node
            for node in self._nodes.values()
            if node.parent_experiment_id == experiment_id
        )

    def ancestors_of(self, experiment_id: str) -> tuple[ExperimentNodeView, ...]:
        result: list[ExperimentNodeView] = []
        current = self.get(experiment_id)
        seen: set[str] = set()
        while current and current.parent_experiment_id and current.parent_experiment_id not in seen:
            seen.add(current.parent_experiment_id)
            parent = self.get(current.parent_experiment_id)
            if parent is None:
                break
            result.append(parent)
            current = parent
        return tuple(result)

    def eligible_parents(self) -> tuple[ExperimentNodeView, ...]:
        return tuple(node for node in self._nodes.values() if node.is_parent_eligible)
