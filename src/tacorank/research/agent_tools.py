"""Typed, read-only tools exposed to the bounded research planner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from ..schemas import ResearchObservation, ResearchTurn, ResearchTurnAction, ResourceDelta
from .graph_view import as_list, get_value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _hash_request(value: Any) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _summary(value: Any) -> dict[str, Any]:
    """Keep only aggregate, label-free research evidence."""

    allowed = (
        "experiment_id", "parent_experiment_id", "family", "hypothesis_summary",
        "trust_verdict", "stability", "integrity", "seed_mean", "seed_stderr",
        "seed_count", "failure_hypotheses", "diagnostic_limitations",
        "diagnostic_best_slice", "diagnostic_worst_slice", "decision",
        "highest_completed_fidelity", "population", "output_accepted",
        "primary_score", "metric_deltas", "parent_delta", "previous_best_delta",
        "prediction_change", "prediction_spearman_vs_parent", "diagnostic_metrics",
        "actual_cost", "method_card_ids", "component_experiment_ids",
        "parent_eligible", "best_eligible", "status", "supporting_event_ids",
    )
    result = {}
    for name in allowed:
        value = get_value(value, name, None)
        if value is not None:
            result[name] = _jsonable(value)
    return result


def _method_card(value: Any) -> dict[str, Any]:
    allowed = (
        "method_id", "family", "status", "cost_tier", "summary", "tags",
        "mechanism", "prerequisites", "allowed_data", "expected_effect",
        "falsifier", "prohibition_conditions",
    )
    return {
        name: _jsonable(get_value(value, name, None))
        for name in allowed
        if get_value(value, name, None) is not None
    }


@dataclass(frozen=True)
class ToolResult:
    observation: ResearchObservation


class ResearchToolRegistry:
    """A controller-owned registry with no filesystem or code access."""

    def __init__(self, context: Any, literature_skill: Any = None):
        self.context = context
        self.literature_skill = literature_skill

    def _observation(
        self,
        turn: ResearchTurn,
        *,
        status: str,
        result: dict[str, Any],
        source_event_ids: Sequence[str] = (),
        literature_evidence: Sequence[Any] = (),
        resource_delta: ResourceDelta | None = None,
    ) -> ToolResult:
        return ToolResult(
            ResearchObservation(
                context_id=str(get_value(self.context, "context_id", "")),
                tool_name=turn.action,
                request_sha256=_hash_request(turn.model_dump(mode="json")),
                status=status,
                result=result,
                source_event_ids=list(dict.fromkeys(str(item) for item in source_event_ids)),
                literature_evidence=list(literature_evidence),
                resource_delta=resource_delta or ResourceDelta(),
            )
        )

    def _history(self) -> list[Any]:
        return as_list(get_value(self.context, "family_history", ()))

    def execute_sync(self, turn: ResearchTurn) -> ToolResult:
        if turn.action == ResearchTurnAction.INSPECT_FRONTIER:
            frontier = as_list(get_value(self.context, "research_frontier", ()))
            if not frontier:
                frontier = as_list(get_value(self.context, "eligible_frontier", ()))
            return self._observation(
                turn, status="available" if frontier else "empty",
                result={"frontier": [_jsonable(item) for item in frontier[:12]]},
                source_event_ids=[
                    event_id for item in frontier[:12]
                    for event_id in as_list(get_value(item, "source_event_ids", ()))
                ],
            )
        if turn.action == ResearchTurnAction.COMPARE_EXPERIMENTS:
            wanted = set(turn.experiment_ids[:4])
            values = [item for item in self._history() if get_value(item, "experiment_id", "") in wanted]
            return self._observation(
                turn, status="available" if values else "empty",
                result={"experiments": [_summary(item) for item in values]},
                source_event_ids=[event_id for item in values for event_id in as_list(get_value(item, "supporting_event_ids", ()))],
            )
        if turn.action == ResearchTurnAction.INSPECT_DIAGNOSTICS:
            wanted = set(turn.experiment_ids[:4])
            values = [item for item in self._history() if not wanted or get_value(item, "experiment_id", "") in wanted]
            result = {
                "diagnostics": [
                    {
                        "experiment_id": get_value(item, "experiment_id", ""),
                        "diagnostic_metrics": _jsonable(get_value(item, "diagnostic_metrics", {})),
                        "prediction_change": get_value(item, "prediction_change", None),
                        "prediction_spearman_vs_parent": get_value(item, "prediction_spearman_vs_parent", None),
                        "limitations": _jsonable(get_value(item, "diagnostic_limitations", [])),
                    }
                    for item in values[:8]
                ]
            }
            return self._observation(
                turn,
                status="available" if values else "empty",
                result=result,
                source_event_ids=[
                    event_id for item in values
                    for event_id in as_list(get_value(item, "supporting_event_ids", ()))
                ],
            )
        if turn.action == ResearchTurnAction.INSPECT_FAILURES:
            values = [
                item for item in self._history()
                if get_value(item, "failure_hypotheses", ())
                or str(get_value(item, "status", "")).lower() in {"invalid", "failed", "rejected"}
            ]
            return self._observation(
                turn, status="available" if values else "empty",
                result={"failures": [
                    {"experiment_id": get_value(item, "experiment_id", ""),
                     "family": get_value(item, "family", None),
                     "status": get_value(item, "status", None),
                     "failure_hypotheses": _jsonable(get_value(item, "failure_hypotheses", [])),
                     "limitations": _jsonable(get_value(item, "diagnostic_limitations", []))}
                    for item in values[:8]
                ]},
                source_event_ids=[event_id for item in values for event_id in as_list(get_value(item, "supporting_event_ids", ()))],
            )
        if turn.action == ResearchTurnAction.INSPECT_METHOD_CARDS:
            wanted = set(turn.method_card_ids[:6])
            cards = [
                item for item in as_list(get_value(self.context, "method_cards", ()))
                if not wanted or get_value(item, "method_id", "") in wanted
            ]
            result = {"method_cards": [_method_card(item) for item in cards[:8]]}
            return self._observation(turn, status="available" if cards else "empty", result=result)
        raise ValueError("execute_sync only accepts read-only non-literature tools")

    async def execute(self, turn: ResearchTurn) -> ToolResult:
        if turn.action != ResearchTurnAction.SEARCH_LITERATURE:
            return self.execute_sync(turn)
        search = getattr(self.literature_skill, "search", None)
        if search is None:
            return self._observation(turn, status="unavailable", result={"reason": "literature_disabled"})
        try:
            result = await search(self.context, str(turn.query))
        except Exception as error:
            return self._observation(
                turn,
                status="unavailable",
                result={"query": str(turn.query), "error": str(error)[:240]},
                resource_delta=None,
            )
        return self._observation(
            turn,
            status=result.status,
            result={"query": str(turn.query), "error": result.error},
            literature_evidence=result.evidence,
            resource_delta=getattr(result, "resource_delta", None)
            or getattr(self.literature_skill, "resource_delta", ResourceDelta()),
        )


__all__ = ["ResearchToolRegistry", "ToolResult"]
