"""Deterministic method-card eligibility shared by policy and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph_view import as_list, enum_value, get_value, has_value
from .search_eligibility import classify_search_eligibility


@dataclass(frozen=True)
class MethodEligibility:
    eligible: bool
    reasons: tuple[str, ...]


def _normalized(value: Any) -> str:
    return str(enum_value(value) or "").strip().lower()


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


def _is_clean_full_result(summary: Any) -> bool:
    verdict = _normalized(get_value(summary, "trust_verdict", None))
    integrity = _normalized(get_value(summary, "integrity", None))
    fidelity = _normalized(get_value(summary, "highest_completed_fidelity", None))
    population = _normalized(get_value(summary, "population", None))
    return (
        verdict in {"accepted", "verified"}
        and integrity == "clean"
        and fidelity == "full"
        and population == "public_validation"
    )


def _is_clean_evaluated_result(summary: Any) -> bool:
    integrity = _normalized(get_value(summary, "integrity", None))
    fidelity = _normalized(get_value(summary, "highest_completed_fidelity", None))
    population = _normalized(get_value(summary, "population", None))
    return (
        integrity == "clean"
        and get_value(summary, "output_accepted", None) is True
        and fidelity in {"proxy", "full"}
        and population in {"internal_proxy", "public_validation"}
    )


def available_capabilities(context: Any) -> frozenset[str]:
    """Derive auditable facts from frozen data permissions and verified history."""

    contract = get_value(context, "contract_summary", None)
    allowed_data = {
        str(item) for item in as_list(get_value(contract, "allowed_data", None))
    }
    capabilities = {
        str(item)
        for item in as_list(get_value(contract, "research_capabilities", None))
    }
    history = as_list(get_value(context, "family_history", None))

    if {"train_interactions", "user_id", "long_view"}.issubset(allowed_data):
        capabilities.update(
            {"within_user_positive_negative_pairs", "user_impression_groups"}
        )
    if {"train_interactions", "date"}.issubset(allowed_data):
        capabilities.add("strict_temporal_cutoff")
    if "duration_ms" in allowed_data:
        capabilities.add("duration_features_legal")
    if "auxiliary_engagement_labels" in allowed_data:
        capabilities.add("legal_auxiliary_label")
    if "random_exposure_log" in allowed_data:
        capabilities.add("random_exposure_log")

    pairwise_results = [
        summary
        for summary in history
        if "objective_pairwise_bpr"
        in {str(item) for item in as_list(get_value(summary, "method_card_ids", None))}
        and _is_clean_evaluated_result(summary)
    ]
    if pairwise_results:
        capabilities.add("pairwise_tested")
        latest = pairwise_results[-1]
        raw_epsilon = get_value(contract, "epsilon", 0.002)
        epsilon = 0.002 if raw_epsilon is None else float(raw_epsilon)
        gauc = _metric_delta(latest, "gauc")
        ndcg = _metric_delta(latest, "ndcg@5", "ndcg")
        if gauc is not None and ndcg is not None and gauc > epsilon and ndcg < -epsilon:
            capabilities.add("ndcg_weakness")

    public_results = [
        summary
        for summary in history
        if _is_clean_full_result(summary)
        and _normalized(get_value(summary, "population", None))
        in {"", "public_validation"}
    ]
    if public_results:
        capabilities.add("standard_public_evaluation_complete")
    confirmed = [
        summary
        for summary in public_results
        if _normalized(get_value(summary, "stability", None)) == "confirmed"
        and bool(get_value(summary, "parent_eligible", False))
    ]
    if len(confirmed) >= 2:
        capabilities.add("two_confirmed_clean_members")
    authorized_ensemble_ids = (
        {
            str(item)
            for item in as_list(get_value(context, "ensemble_candidate_ids", None))
        }
        if has_value(context, "ensemble_candidate_ids")
        else None
    )
    if any(
        (
            authorized_ensemble_ids is None
            or str(get_value(summary, "experiment_id", ""))
            in authorized_ensemble_ids
        )
        and classify_search_eligibility(summary, context).ensemble_eligible
        for summary in history
    ):
        capabilities.add("diverse_clean_proxy_member")
    return frozenset(capabilities)


def method_card_map(context: Any) -> dict[str, Any]:
    return {
        str(get_value(card, "method_id", "")): card
        for card in as_list(get_value(context, "method_cards", None))
        if str(get_value(card, "method_id", ""))
    }


def evaluate_method_card(card: Any, context: Any, *, family: str | None = None) -> MethodEligibility:
    reasons: list[str] = []
    status = _normalized(get_value(card, "status", None))
    if status != "candidate":
        reasons.append("METHOD_STATUS_NOT_CANDIDATE")
    if family is not None and str(get_value(card, "family", "")) != family:
        reasons.append("METHOD_FAMILY_MISMATCH")

    contract = get_value(context, "contract_summary", None)
    permitted_data = {
        str(item) for item in as_list(get_value(contract, "allowed_data", None))
    }
    requested_data = {
        str(item) for item in as_list(get_value(card, "allowed_data", None))
    }
    if not requested_data:
        reasons.append("METHOD_ALLOWED_DATA_UNDECLARED")
    elif not requested_data.issubset(permitted_data):
        reasons.append("METHOD_DATA_NOT_ALLOWED")

    prerequisites = {
        str(item) for item in as_list(get_value(card, "prerequisites", None))
    }
    if not prerequisites.issubset(available_capabilities(context)):
        reasons.append("METHOD_PREREQUISITES_UNSATISFIED")

    active_prohibitions = {
        str(item)
        for item in as_list(get_value(contract, "active_prohibitions", None))
    }
    card_prohibitions = {
        str(item)
        for item in as_list(get_value(card, "prohibition_conditions", None))
    }
    if active_prohibitions.intersection(card_prohibitions):
        reasons.append("METHOD_PROHIBITED")
    return MethodEligibility(not reasons, tuple(reasons))


def eligible_method_cards(context: Any, family: str | None = None) -> tuple[Any, ...]:
    cards = as_list(get_value(context, "method_cards", None))
    return tuple(
        card
        for card in cards
        if evaluate_method_card(card, context, family=family).eligible
    )
