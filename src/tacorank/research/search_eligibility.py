"""Derived search eligibility for trusted, soft-pruned, and retired results.

The event ledger remains authoritative for experiment decisions.  These
transient flags only control which *research action* Person 1 may consider:
branching, checkpoint selection, one bounded refinement, or a diversity-tested
ensemble.  They never turn a negative result into an accepted checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .graph_view import enum_value, get_value


class PruneDisposition(str, Enum):
    FRONTIER = "frontier"
    SOFT = "soft_prune"
    HARD = "hard_prune"
    NULL = "null_result"
    PENDING = "pending"


@dataclass(frozen=True)
class SearchEligibility:
    branch_eligible: bool
    best_checkpoint_eligible: bool
    refinement_eligible: bool
    ensemble_eligible: bool
    disposition: PruneDisposition
    reasons: tuple[str, ...]


def _normalized(value: Any) -> str:
    return str(enum_value(value) or "").strip().lower()


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _metric_delta(summary: Any, *names: str) -> float | None:
    values = get_value(summary, "metric_deltas", None) or {}
    try:
        lowered = {str(key).lower(): float(value) for key, value in dict(values).items()}
    except (TypeError, ValueError):
        return None
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def classify_search_eligibility(summary: Any, context: Any) -> SearchEligibility:
    """Classify a verified summary without altering its canonical decision.

    A soft-pruned result must be clean, meaningfully different, and either
    close to its parent or exhibit a component-metric trade-off.  The score
    floor is deliberately bounded by the larger of five contract epsilons and
    one absolute primary-score point.  This keeps severe regressions out of
    both refinement and ensemble search.
    """

    contract = get_value(context, "contract_summary", None)
    epsilon = _number(get_value(contract, "epsilon", 0.002))
    epsilon = 0.002 if epsilon is None else epsilon
    no_op_threshold = _number(
        get_value(contract, "prediction_change_no_op_threshold", 0.001)
    )
    no_op_threshold = 0.001 if no_op_threshold is None else no_op_threshold
    severe_regression = max(5.0 * epsilon, 0.01)

    output_accepted = get_value(summary, "output_accepted", None)
    verdict = _normalized(get_value(summary, "trust_verdict", None))
    integrity = _normalized(get_value(summary, "integrity", None))
    stability = _normalized(get_value(summary, "stability", None))
    status = _normalized(get_value(summary, "status", None))
    decision = _normalized(get_value(summary, "decision", None))
    fidelity = _normalized(get_value(summary, "highest_completed_fidelity", None))
    population = _normalized(get_value(summary, "population", None))
    parent_delta = _number(get_value(summary, "parent_delta", None))
    prediction_change = _number(get_value(summary, "prediction_change", None))
    spearman = _number(get_value(summary, "prediction_spearman_vs_parent", None))
    child_count = int(get_value(summary, "child_count", 0) or 0)
    branch = bool(get_value(summary, "parent_eligible", False))
    best = bool(get_value(summary, "best_eligible", False))

    safety_reasons: list[str] = []
    if output_accepted is False:
        safety_reasons.append("OUTPUT_REJECTED")
    if integrity == "compromised" or verdict == "suspicious":
        safety_reasons.append("INTEGRITY_UNTRUSTED")
    if safety_reasons:
        return SearchEligibility(
            False,
            False,
            False,
            False,
            PruneDisposition.HARD,
            tuple(safety_reasons),
        )

    no_op = verdict == "no_op" or (
        prediction_change is not None and prediction_change <= no_op_threshold
    )
    if no_op:
        # A no-op is evidence, not a controller prune. It cannot itself become
        # a checkpoint or parent, but the planner separately receives the
        # bounded same-mechanism and independent-mechanism choices.
        return SearchEligibility(
            False,
            False,
            False,
            False,
            PruneDisposition.NULL,
            ("NO_MEANINGFUL_PREDICTION_CHANGE",),
        )

    hard_reasons: list[str] = []
    if stability == "unstable":
        hard_reasons.append("UNSTABLE")
    if status in {"invalid", "retracted", "suspicious"} or decision == "invalid":
        hard_reasons.append("INVALID_OR_RETRACTED")
    if parent_delta is not None and parent_delta < -severe_regression:
        hard_reasons.append("SEVERE_PRIMARY_REGRESSION")
    if hard_reasons:
        return SearchEligibility(
            False,
            False,
            False,
            False,
            PruneDisposition.HARD,
            tuple(dict.fromkeys(hard_reasons)),
        )

    if branch:
        return SearchEligibility(
            branch_eligible=True,
            best_checkpoint_eligible=best,
            refinement_eligible=False,
            ensemble_eligible=False,
            disposition=PruneDisposition.FRONTIER,
            reasons=("CANONICAL_PARENT_ELIGIBLE",),
        )

    completed_clean_evaluation = (
        output_accepted is True
        and integrity == "clean"
        and fidelity in {"proxy", "full"}
        and population in {"internal_proxy", "public_validation"}
        and verdict in {"negative", "inconclusive", "accepted", "verified"}
        and prediction_change is not None
        and prediction_change > no_op_threshold
        and parent_delta is not None
    )
    if not completed_clean_evaluation:
        return SearchEligibility(
            False,
            best,
            False,
            False,
            PruneDisposition.PENDING,
            ("RESULT_NOT_READY_FOR_PORTFOLIO",),
        )

    gauc = _metric_delta(summary, "gauc")
    ndcg = _metric_delta(summary, "ndcg@5", "ndcg")
    metric_tradeoff = (
        gauc is not None
        and ndcg is not None
        and ((gauc > epsilon and ndcg < -epsilon) or (ndcg > epsilon and gauc < -epsilon))
    )
    close_to_parent = parent_delta >= -severe_regression
    if not (close_to_parent or metric_tradeoff):
        return SearchEligibility(
            False,
            best,
            False,
            False,
            PruneDisposition.HARD,
            ("NEGATIVE_WITHOUT_PORTFOLIO_HEADROOM",),
        )

    # A soft node may receive at most one evidence-backed child.  SearchPolicy
    # additionally requires a documented follow-up method before using this flag.
    refinement = child_count == 0 and metric_tradeoff
    diverse = spearman is not None and abs(spearman) < 0.98
    ensemble = diverse and (close_to_parent or metric_tradeoff)
    reasons = ["CLEAN_MEANINGFUL_SOFT_RESULT"]
    if metric_tradeoff:
        reasons.append("COMPONENT_METRIC_TRADEOFF")
    if diverse:
        reasons.append("DIVERSE_FROM_PARENT")
    return SearchEligibility(
        False,
        best,
        refinement,
        ensemble,
        PruneDisposition.SOFT,
        tuple(reasons),
    )
