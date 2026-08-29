"""Deterministic two-phase AIDE search policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph_view import ExperimentNodeView, GraphView, as_list, get_value
from .portfolio import ALL_FAMILIES, HIGH_VALUE_FAMILIES


@dataclass(frozen=True)
class PolicyChoice:
    action: str
    parent: ExperimentNodeView | None
    family: str | None
    cost_tier: str
    phase: str
    reason_code: str
    reason: str


def _score(node: ExperimentNodeView) -> float:
    return float("-inf") if node.primary_score is None else node.primary_score


def _allowed_families(context: Any) -> tuple[str, ...]:
    contract = get_value(context, "contract_summary", None)
    allowed = get_value(contract, "allowed_families", None) or get_value(
        contract, "experiment_families", None
    )
    if allowed is None:
        return ALL_FAMILIES
    return tuple(family for family in ALL_FAMILIES if family in set(map(str, as_list(allowed))))


def _family_history(context: Any) -> list[str]:
    history = as_list(get_value(context, "family_history", None))
    values = [str(get_value(item, "family", "")) for item in history]
    return [value for value in values if value]


def _cost_tier(node: ExperimentNodeView) -> str:
    value = node.actual_cost
    tier = get_value(value, "cost_tier", None)
    if tier is not None:
        return str(tier).lower()
    return str(value).lower() if value is not None else "medium"


class SearchPolicy:
    """Select an eligible parent and family without mutable state."""

    def __init__(self, frontier_limit: int = 3):
        if frontier_limit < 1:
            raise ValueError("frontier_limit must be positive")
        self.frontier_limit = frontier_limit

    def choose(self, context: Any) -> PolicyChoice:
        graph = GraphView.from_context(context)
        eligible = list(graph.eligible_parents())
        allowed = _allowed_families(context)
        history = _family_history(context)
        if not eligible:
            return PolicyChoice(
                action="blocked",
                parent=None,
                family=None,
                cost_tier="low",
                phase="none",
                reason_code="NO_ELIGIBLE_PARENT",
                reason="No verified full-fidelity parent is available in the planner context.",
            )
        if not allowed:
            return PolicyChoice(
                action="blocked",
                parent=None,
                family=None,
                cost_tier="low",
                phase="none",
                reason_code="NO_LEGAL_FAMILY",
                reason="The frozen contract exposes no legal experiment family.",
            )

        tried = set(history)
        breadth_family = next(
            (family for family in HIGH_VALUE_FAMILIES if family in allowed and family not in tried),
            None,
        )
        if breadth_family:
            baseline = next((node for node in eligible if node.is_root), None)
            parent = baseline or min(eligible, key=lambda node: node.experiment_id)
            return PolicyChoice(
                action="propose",
                parent=parent,
                family=breadth_family,
                cost_tier="medium",
                phase="breadth",
                reason_code="BREADTH_FAMILY_PROBE",
                reason=(
                    f"Probe untried high-value family {breadth_family} from "
                    f"{parent.experiment_id}."
                ),
            )

        frontier = sorted(
            eligible,
            key=lambda node: (
                -_score(node),
                node.child_count,
                node.experiment_id,
            ),
        )[: self.frontier_limit]
        recent = history[-2:]
        parent = next(
            (node for node in frontier if str(node.family or "") not in recent),
            frontier[0],
        )
        family = next(
            (candidate for candidate in allowed if candidate not in recent),
            allowed[0],
        )
        return PolicyChoice(
            action="propose",
            parent=parent,
            family=family,
            cost_tier="low" if _cost_tier(parent) == "low" else "medium",
            phase="depth",
            reason_code="EVIDENCE_GUIDED_DEPTH",
            reason=(
                f"Select {parent.experiment_id} using trusted score, child count, "
                "family diversity and deterministic tie-breaking."
            ),
        )
