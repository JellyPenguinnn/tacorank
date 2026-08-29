"""Deterministic two-phase AIDE search policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph_view import ExperimentNodeView, GraphView, as_list, enum_value, get_value
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
    method_card_id: str | None = None


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


def _normalized(value: Any) -> str:
    return str(enum_value(value) or "").strip().lower()


def _metric_delta(summary: Any, *names: str) -> float | None:
    deltas = get_value(summary, "metric_deltas", None) or {}
    if not isinstance(deltas, dict):
        try:
            deltas = dict(deltas)
        except (TypeError, ValueError):
            deltas = {}
    lowered = {str(key).lower(): value for key, value in deltas.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _playbook_choice(
    context: Any,
    eligible: list[ExperimentNodeView],
    allowed: tuple[str, ...],
) -> PolicyChoice | None:
    """Apply mandatory evidence routing before generic breadth/depth search."""

    history = as_list(get_value(context, "family_history", None))
    if not history:
        return None
    latest = history[-1]
    trust = get_value(latest, "trust", None)
    verdict = _normalized(
        get_value(latest, "trust_verdict", None) or get_value(trust, "verdict", None)
    )
    integrity = _normalized(
        get_value(latest, "integrity", None) or get_value(trust, "integrity", None)
    )
    stability = _normalized(
        get_value(latest, "stability", None) or get_value(trust, "stability", None)
    )
    fidelity = _normalized(
        get_value(latest, "highest_completed_fidelity", None)
        or get_value(latest, "highest_fidelity_completed", None)
    )

    if integrity == "compromised" or verdict == "suspicious":
        return PolicyChoice(
            action="blocked",
            parent=None,
            family=None,
            cost_tier="low",
            phase="playbook_gate",
            reason_code="SUSPICIOUS_RESULT_REQUIRES_QUARANTINE",
            reason="The latest evaluation is suspicious or integrity-compromised.",
        )
    if verdict == "no_op":
        return PolicyChoice(
            action="blocked",
            parent=None,
            family=None,
            cost_tier="low",
            phase="playbook_gate",
            reason_code="NO_OP_REQUIRES_RECOVERY",
            reason="The latest candidate did not change predictions meaningfully.",
        )
    if stability == "unstable":
        return PolicyChoice(
            action="blocked",
            parent=None,
            family=None,
            cost_tier="low",
            phase="playbook_gate",
            reason_code="UNSTABLE_RESULT_REQUIRES_CONFIRMATION",
            reason="The latest result requires seed confirmation before branching.",
        )
    if fidelity in {"smoke", "proxy"}:
        return PolicyChoice(
            action="blocked",
            parent=None,
            family=None,
            cost_tier="low",
            phase="playbook_gate",
            reason_code="FIDELITY_PROMOTION_REQUIRED",
            reason="A smoke or proxy result cannot create a new research branch.",
        )

    # Metric-shape routing only treats a clean, trusted full result as reward.
    if verdict not in {"accepted", "verified"} or integrity not in {"", "clean"}:
        return None
    if fidelity and fidelity != "full":
        return None

    latest_id = str(get_value(latest, "experiment_id", ""))
    best = max(eligible, key=lambda node: (_score(node), -node.child_count))
    parent = next((node for node in eligible if node.experiment_id == latest_id), best)
    family = str(get_value(latest, "family", ""))
    method_ids = {
        str(item) for item in as_list(get_value(latest, "method_card_ids", None))
    }
    hypothesis = " ".join(
        str(get_value(latest, field, ""))
        for field in ("hypothesis_summary", "hypothesis", "change_summary")
    ).lower()
    is_pairwise = (
        "objective_pairwise_bpr" in method_ids
        or (family == "objective" and ("pairwise" in hypothesis or "bpr" in hypothesis))
    )
    contract = get_value(context, "contract_summary", None)
    epsilon = _number(get_value(contract, "epsilon", 0.002)) or 0.002
    gauc_delta = _metric_delta(latest, "gauc", "GAUC")
    ndcg_delta = _metric_delta(latest, "ndcg@5", "nDCG@5", "ndcg")
    parent_delta = _number(get_value(latest, "parent_delta", None))
    prediction_change = _number(get_value(latest, "prediction_change", None))

    if is_pairwise and "objective" in allowed:
        if (
            gauc_delta is not None
            and ndcg_delta is not None
            and gauc_delta > epsilon
            and ndcg_delta < -epsilon
        ):
            return PolicyChoice(
                action="propose",
                parent=parent,
                family="objective",
                cost_tier="medium",
                phase="playbook",
                reason_code="PAIRWISE_GAUC_UP_NDCG_DOWN",
                reason=(
                    "Pairwise ranking improved GAUC but hurt nDCG@5; follow with "
                    "one listwise or top-weighted objective experiment."
                ),
                method_card_id="objective_listwise_user_softmax",
            )
        if (
            gauc_delta is not None
            and ndcg_delta is not None
            and gauc_delta > epsilon
            and ndcg_delta > epsilon
        ):
            return PolicyChoice(
                action="propose",
                parent=parent,
                family="objective",
                cost_tier="medium",
                phase="playbook",
                reason_code="PAIRWISE_BOTH_METRICS_UP",
                reason="Confirm or atomically refine the successful objective mechanism.",
                method_card_id="objective_pairwise_bpr",
            )
        if (
            parent_delta is not None
            and abs(parent_delta) <= epsilon
            and prediction_change is not None
            and prediction_change > 0.0
            and "temporal_history" in allowed
        ):
            return PolicyChoice(
                action="propose",
                parent=best,
                family="temporal_history",
                cost_tier="medium",
                phase="playbook",
                reason_code="PAIRWISE_MEANINGFUL_NO_GAIN",
                reason="Pairwise predictions changed without a trusted gain; move to compact history.",
                method_card_id="temporal_history_compact",
            )
    return None


def _cost_tier(node: ExperimentNodeView) -> str:
    value = node.actual_cost
    tier = get_value(value, "cost_tier", None)
    if tier is not None:
        return _normalized(tier)
    return _normalized(value) if value is not None else "medium"


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

        routed = _playbook_choice(context, eligible, allowed)
        if routed is not None:
            return routed

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
        # Diversity decides what to try, not whether to discard the strongest
        # verified parent. The latter caused a family-less baseline root to beat
        # a higher-scoring frontier candidate.
        parent = frontier[0]
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
