"""Deterministic, playbook-driven, score-guided depth-first search policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .graph_view import (
    ExperimentNodeView,
    GraphView,
    as_list,
    enum_value,
    get_value,
    has_value,
)
from .linucb import LinUCBLegalChoiceRanker
from .method_eligibility import eligible_method_cards, method_card_map
from .playbook import REQUIRED_RULE_ORDER
from .portfolio import HIGH_VALUE_FAMILIES
from .search_eligibility import classify_search_eligibility


DEFAULT_RULE_ORDER = REQUIRED_RULE_ORDER
DEFAULT_FAMILY_ORDER = HIGH_VALUE_FAMILIES + (
    "sampling",
    "ensemble",
    "evaluation",
    "other",
)
DEFAULT_METHOD_ORDER = {
    "objective": ("objective_pairwise_bpr", "objective_listwise_user_softmax"),
    "temporal_history": ("temporal_history_compact",),
    "multitask": ("multitask_single_auxiliary",),
    "duration_bias": ("duration_bias_censored_watch_time",),
    "features": ("temporal_drift_past_only",),
    "model": ("model_compact_ranker",),
    "ensemble": (
        "ensemble_diverse_residual_candidate",
        "ensemble_confirmed_members",
    ),
    "evaluation": ("evaluation_random_exposure_robustness",),
}


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
    component_experiment_ids: tuple[str, ...] = ()


LegalChoiceRanker = Callable[[Sequence[PolicyChoice], Any], PolicyChoice]


def _score(node: ExperimentNodeView) -> float:
    return float("-inf") if node.primary_score is None else node.primary_score


def _normalized(value: Any) -> str:
    return str(enum_value(value) or "").strip().lower()


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _metric_delta(summary: Any, *names: str) -> float | None:
    deltas = get_value(summary, "metric_deltas", None) or {}
    try:
        lowered = {str(key).lower(): float(value) for key, value in dict(deltas).items()}
    except (TypeError, ValueError):
        return None
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _diagnostic_suffix(summary: Any) -> str:
    values = get_value(summary, "diagnostic_metrics", None) or {}
    if not isinstance(values, dict):
        try:
            values = dict(values)
        except (TypeError, ValueError):
            return ""
    preferred = (
        "spearman_vs_fm_baseline",
        "user_rankable_fraction",
        "item_personalized_fraction",
        "parent_residual_std",
    )
    rendered = []
    for name in preferred:
        value = _number(values.get(name))
        if value is not None:
            rendered.append("%s=%.6g" % (name, value))
    return " Diagnostics: " + ", ".join(rendered) + "." if rendered else ""


def _allowed_families(context: Any) -> tuple[str, ...]:
    contract = get_value(context, "contract_summary", None)
    allowed = get_value(contract, "allowed_families", None)
    if allowed is None:
        allowed = get_value(contract, "experiment_families", None)
    if allowed is None:
        return ()
    allowed_set = {str(item) for item in as_list(allowed)}
    return tuple(family for family in _family_order(context) if family in allowed_set)


def _family_history(context: Any) -> list[str]:
    values = [
        str(get_value(item, "family", ""))
        for item in as_list(get_value(context, "family_history", None))
    ]
    return [value for value in values if value]


def _playbook(context: Any) -> Any:
    return get_value(context, "playbook", None)


def _playbook_is_valid(context: Any) -> bool:
    playbook = _playbook(context)
    if playbook is None or str(get_value(playbook, "schema_version", "")) != "1.0":
        return False
    rules = tuple(
        str(item) for item in as_list(get_value(playbook, "rule_order", None))
    )
    families = tuple(
        str(item) for item in as_list(get_value(playbook, "family_order", None))
    )
    methods = get_value(playbook, "method_order", None)
    try:
        objective_methods = tuple(str(item) for item in as_list(methods.get("objective")))
    except AttributeError:
        return False
    return (
        rules == REQUIRED_RULE_ORDER
        and bool(families)
        and len(families) == len(set(families))
        and bool(objective_methods)
        and objective_methods[0] == "objective_pairwise_bpr"
    )


def _rule_order(context: Any) -> tuple[str, ...]:
    configured = tuple(
        str(item) for item in as_list(get_value(_playbook(context), "rule_order", None))
    )
    return configured or DEFAULT_RULE_ORDER


def _family_order(context: Any) -> tuple[str, ...]:
    configured = tuple(
        str(item) for item in as_list(get_value(_playbook(context), "family_order", None))
    )
    return configured or DEFAULT_FAMILY_ORDER


def _method_order(context: Any, family: str) -> tuple[str, ...]:
    configured = get_value(_playbook(context), "method_order", None) or {}
    try:
        values = configured.get(family, ())
    except AttributeError:
        values = ()
    result = tuple(str(item) for item in as_list(values))
    return result or DEFAULT_METHOD_ORDER.get(family, ())


def _method_for_family(
    context: Any,
    family: str,
    *,
    preferred: str | None = None,
    parent_experiment_id: str | None = None,
) -> Any | None:
    eligible = {
        str(get_value(card, "method_id", "")): card
        for card in eligible_method_cards(context, family)
    }
    if preferred is None:
        # This card requires an explicit secondary component chosen by the
        # soft-portfolio route. Generic depth-first proposals have no such
        # component contract and must not select it.
        eligible.pop("ensemble_diverse_residual_candidate", None)
    attempted = _attempted_methods_for_parent(context, parent_experiment_id)
    if preferred is not None:
        return None if preferred in attempted else eligible.get(preferred)
    for method_id in _method_order(context, family):
        if method_id in eligible and method_id not in attempted:
            return eligible[method_id]
    remaining = sorted(set(eligible) - attempted)
    return eligible[remaining[0]] if remaining else None


def _attempted_methods_for_parent(
    context: Any, parent_experiment_id: str | None
) -> set[str]:
    return {
        method_id
        for summary in as_list(get_value(context, "family_history", None))
        if parent_experiment_id is not None
        and str(get_value(summary, "parent_experiment_id", ""))
        == parent_experiment_id
        for method_id in map(
            str, as_list(get_value(summary, "method_card_ids", None))
        )
    }


def _cost_tier(value: Any) -> str:
    tier = get_value(value, "cost_tier", value)
    normalized = _normalized(tier)
    return normalized if normalized in {"low", "medium", "high"} else "medium"


def _proposal(
    *,
    parent: ExperimentNodeView,
    family: str,
    card: Any,
    phase: str,
    reason_code: str,
    reason: str,
    component_experiment_ids: tuple[str, ...] = (),
) -> PolicyChoice:
    return PolicyChoice(
        action="propose",
        parent=parent,
        family=family,
        cost_tier=_cost_tier(get_value(card, "cost_tier", None)),
        phase=phase,
        reason_code=reason_code,
        reason=reason,
        method_card_id=str(get_value(card, "method_id", "")),
        component_experiment_ids=component_experiment_ids,
    )


def _blocked(reason_code: str, reason: str, *, phase: str = "playbook_gate") -> PolicyChoice:
    return PolicyChoice(
        action="blocked",
        parent=None,
        family=None,
        cost_tier="low",
        phase=phase,
        reason_code=reason_code,
        reason=reason,
    )


def _best_parent(eligible: Sequence[ExperimentNodeView]) -> ExperimentNodeView:
    return sorted(
        eligible,
        key=lambda node: (-_score(node), node.child_count, node.experiment_id),
    )[0]


def _best_experimental_parent(
    eligible: Sequence[ExperimentNodeView],
) -> ExperimentNodeView:
    """Choose the strongest measured research node before baseline fallback."""

    experimental = [
        node
        for node in eligible
        if not node.is_root and node.primary_score is not None
    ]
    return _best_parent(experimental or eligible)


def _same_family_refinement_choice(
    context: Any,
    parent: ExperimentNodeView,
    allowed: tuple[str, ...],
) -> PolicyChoice | None:
    """Try one legal refinement of the strongest experimental mechanism."""

    family = str(parent.family or "")
    if not family or family not in allowed:
        return None
    eligible = {
        str(get_value(card, "method_id", "")): card
        for card in eligible_method_cards(context, family)
    }
    if family == "ensemble":
        eligible.pop("ensemble_diverse_residual_candidate", None)
    attempted = _attempted_methods_for_parent(context, parent.experiment_id)
    parent_methods = set(parent.method_card_ids)
    ordered = list(_method_order(context, family))
    ordered.extend(sorted(set(eligible) - set(ordered)))

    # Prefer a distinct method within the same mechanism family. If the family
    # exposes only one method card, permit one materially different child from
    # that parent; duplicate-plan validation still rejects an identical plan.
    method_id = next(
        (
            item
            for item in ordered
            if item in eligible
            and item not in attempted
            and item not in parent_methods
        ),
        None,
    )
    if method_id is None:
        method_id = next(
            (
                item
                for item in ordered
                if item in eligible and item not in attempted
            ),
            None,
        )
    if method_id is None:
        return None
    return _proposal(
        parent=parent,
        family=family,
        card=eligible[method_id],
        phase="playbook",
        reason_code="SCORE_GUIDED_SAME_FAMILY_REFINEMENT",
        reason=(
            "Predictions changed without a trusted gain; refine the strongest "
            "eligible experimental path %s within family %s before introducing "
            "an unrelated mechanism." % (parent.experiment_id, family)
        ),
    )


def _depth_first_frontier(
    graph: GraphView,
    eligible: Sequence[ExperimentNodeView],
    limit: int,
) -> tuple[ExperimentNodeView, ...]:
    """Rank trusted branches by score, then depth, for deterministic backtracking."""

    # The first sort supplies the stable tie-break for equal score and depth:
    # the newest experiment ID stays at the front, matching stack-like DFS.
    newest_first = sorted(
        eligible,
        key=lambda node: node.experiment_id,
        reverse=True,
    )
    ranked = sorted(
        newest_first,
        key=lambda node: (
            _score(node),
            len(graph.ancestors_of(node.experiment_id)),
        ),
        reverse=True,
    )
    return tuple(ranked[:limit])


def _latest_parent(
    latest: Any, eligible: Sequence[ExperimentNodeView]
) -> ExperimentNodeView:
    latest_id = str(get_value(latest, "experiment_id", ""))
    return next(
        (node for node in eligible if node.experiment_id == latest_id),
        _best_parent(eligible),
    )


def _next_independent_choice(
    context: Any,
    eligible: Sequence[ExperimentNodeView],
    allowed: tuple[str, ...],
    latest_family: str,
    *,
    reason_code: str,
    reason: str,
    preferred_parent: ExperimentNodeView | None = None,
) -> PolicyChoice | None:
    choices = _independent_choices(
        context,
        eligible,
        allowed,
        latest_family,
        reason_code=reason_code,
        reason=reason,
        preferred_parent=preferred_parent,
    )
    return choices[0] if choices else None


def _independent_choices(
    context: Any,
    eligible: Sequence[ExperimentNodeView],
    allowed: tuple[str, ...],
    latest_family: str,
    *,
    reason_code: str,
    reason: str,
    preferred_parent: ExperimentNodeView | None = None,
) -> tuple[PolicyChoice, ...]:
    """Return every legal independent-family action in deterministic order."""

    tried = set(_family_history(context))
    parent = preferred_parent or _best_experimental_parent(eligible)
    ordered = [
        family
        for family in _family_order(context)
        if family in allowed and family != latest_family
    ]
    ordered.sort(key=lambda family: (family in tried, _family_order(context).index(family)))
    choices = []
    for family in ordered:
        card = _method_for_family(
            context,
            family,
            parent_experiment_id=parent.experiment_id,
        )
        if card is not None:
            choices.append(
                _proposal(
                    parent=parent,
                    family=family,
                    card=card,
                    phase="playbook",
                    reason_code=reason_code,
                    reason=reason,
                )
            )
    return tuple(choices)


def _no_op_choices(
    context: Any,
    eligible: Sequence[ExperimentNodeView],
    allowed: tuple[str, ...],
) -> tuple[PolicyChoice, ...] | None:
    """Expose bounded next actions after a terminal prediction no-op.

    ``None`` means the latest result is not a no-op. An empty tuple means it is
    a no-op but no legal next action remains. The no-op node itself is never a
    parent: a reimplementation branches from its last trusted parent.
    """

    history = as_list(get_value(context, "family_history", None))
    if not history:
        return None
    latest = history[-1]
    if _normalized(get_value(latest, "status", None)) != "no_op":
        return None
    verdict = _normalized(get_value(latest, "trust_verdict", None))
    contract = get_value(context, "contract_summary", None)
    no_op_threshold = _number(
        get_value(contract, "prediction_change_no_op_threshold", 0.001)
    )
    no_op_threshold = 0.001 if no_op_threshold is None else no_op_threshold
    prediction_change = _number(get_value(latest, "prediction_change", None))
    if verdict != "no_op" and not (
        prediction_change is not None and prediction_change <= no_op_threshold
    ):
        return None

    family = str(get_value(latest, "family", ""))
    choices = list(
        _independent_choices(
            context,
            eligible,
            allowed,
            family,
            reason_code="NO_OP_INDEPENDENT_MECHANISM",
            reason=(
                "The terminal no-op is research evidence rather than an adapter "
                "failure; test an independent mechanism from the trusted frontier."
            ),
        )
    )

    parent_id = str(get_value(latest, "parent_experiment_id", ""))
    parent = next(
        (node for node in eligible if node.experiment_id == parent_id),
        None,
    )
    latest_methods = tuple(
        str(item)
        for item in as_list(get_value(latest, "method_card_ids", None))
        if str(item)
    )
    same_mechanism_no_ops = 0
    for summary in history:
        summary_methods = {
            str(item)
            for item in as_list(get_value(summary, "method_card_ids", None))
        }
        summary_change = _number(get_value(summary, "prediction_change", None))
        summary_is_no_op = _normalized(
            get_value(summary, "trust_verdict", None)
        ) == "no_op" or (
            summary_change is not None and summary_change <= no_op_threshold
        )
        if (
            summary_is_no_op
            and str(get_value(summary, "parent_experiment_id", "")) == parent_id
            and str(get_value(summary, "family", "")) == family
            and bool(summary_methods.intersection(latest_methods))
        ):
            same_mechanism_no_ops += 1

    # Permit one planner-selected reimplementation after the first no-op. A
    # second no-op for the same parent/family/method retires that mechanism.
    if (
        parent is not None
        and family in allowed
        and len(latest_methods) == 1
        and same_mechanism_no_ops == 1
    ):
        eligible_cards = {
            str(get_value(card, "method_id", "")): card
            for card in eligible_method_cards(context, family)
        }
        card = eligible_cards.get(latest_methods[0])
        if card is not None:
            choices.append(
                _proposal(
                    parent=parent,
                    family=family,
                    card=card,
                    phase="no_op_reimplementation",
                    reason_code="NO_OP_REIMPLEMENT_MECHANISM",
                    reason=(
                        "The previous implementation produced identical predictions. "
                        "Reimplement or materially refine the same approved mechanism "
                        "once from its trusted parent; the duplicate-plan gate remains "
                        "binding."
                    ),
                )
            )
    return tuple(choices)


def _required_method_choice(
    context: Any,
    parent: ExperimentNodeView,
    family: str,
    method_id: str,
    *,
    reason_code: str,
    reason: str,
) -> PolicyChoice:
    if family not in _allowed_families(context):
        return _blocked(
            "REQUIRED_FAMILY_UNAVAILABLE",
            "Playbook family %s is not allowed by the frozen contract." % family,
        )
    card = _method_for_family(
        context,
        family,
        preferred=method_id,
        parent_experiment_id=parent.experiment_id,
    )
    if card is None:
        return _blocked(
            "REQUIRED_METHOD_UNAVAILABLE",
            "Playbook method %s is not eligible under the frozen method-card contract."
            % method_id,
        )
    return _proposal(
        parent=parent,
        family=family,
        card=card,
        phase="playbook",
        reason_code=reason_code,
        reason=reason,
    )


def _soft_prune_choice(
    context: Any,
    latest: Any,
    eligible: Sequence[ExperimentNodeView],
    allowed: tuple[str, ...],
) -> PolicyChoice | None:
    """Return one bounded refinement or ensemble action for a soft result."""

    search = classify_search_eligibility(latest, context)
    node = ExperimentNodeView.from_summary(latest)
    if node is None:
        return None
    refinement_ids = {
        str(item)
        for item in as_list(get_value(context, "refinement_frontier_ids", None))
    }
    ensemble_ids = {
        str(item)
        for item in as_list(get_value(context, "ensemble_candidate_ids", None))
    }
    refinement_authorized = (
        node.experiment_id in refinement_ids
        if has_value(context, "refinement_frontier_ids")
        else search.refinement_eligible
    )
    ensemble_authorized = (
        node.experiment_id in ensemble_ids
        if has_value(context, "ensemble_candidate_ids")
        else search.ensemble_eligible
    )
    method_ids = {
        str(item) for item in as_list(get_value(latest, "method_card_ids", None))
    }
    if refinement_authorized and "objective_pairwise_bpr" in method_ids:
        card = _method_for_family(
            context,
            "objective",
            preferred="objective_listwise_user_softmax",
            parent_experiment_id=node.experiment_id,
        )
        if "objective" in allowed and card is not None:
            return _proposal(
                parent=node,
                family="objective",
                card=card,
                phase="refinement",
                reason_code="SOFT_PRUNE_METRIC_TRADEOFF_REFINEMENT",
                reason=(
                    "The clean pairwise proxy exposed a component-metric trade-off; "
                    "allow exactly one documented listwise refinement from %s."
                    % node.experiment_id
                ),
            )

    if ensemble_authorized and "ensemble" in allowed:
        parent = _best_parent(eligible)
        card = _method_for_family(
            context,
            "ensemble",
            preferred="ensemble_diverse_residual_candidate",
            parent_experiment_id=parent.experiment_id,
        )
        if card is not None and node.experiment_id != parent.experiment_id:
            return _proposal(
                parent=parent,
                family="ensemble",
                card=card,
                phase="ensemble",
                reason_code="SOFT_PRUNE_DIVERSE_ENSEMBLE_TEST",
                reason=(
                    "Retain %s only as a clean diverse secondary component and "
                    "test one predeclared blend against trusted parent %s."
                    % (node.experiment_id, parent.experiment_id)
                ),
                component_experiment_ids=(node.experiment_id,),
            )
    return None


def _playbook_choice(
    context: Any,
    eligible: list[ExperimentNodeView],
    allowed: tuple[str, ...],
) -> PolicyChoice | None:
    history = as_list(get_value(context, "family_history", None))
    if not history:
        return None
    latest = history[-1]
    verdict = _normalized(get_value(latest, "trust_verdict", None))
    integrity = _normalized(get_value(latest, "integrity", None))
    stability = _normalized(get_value(latest, "stability", None))
    fidelity = _normalized(get_value(latest, "highest_completed_fidelity", None))
    population = _normalized(get_value(latest, "population", None))
    output_accepted = get_value(latest, "output_accepted", None)
    contract = get_value(context, "contract_summary", None)
    epsilon = _number(get_value(contract, "epsilon", 0.002))
    if epsilon is None:
        epsilon = 0.002
    no_op_threshold = _number(
        get_value(contract, "prediction_change_no_op_threshold", 0.0)
    ) or 0.0
    prediction_change = _number(get_value(latest, "prediction_change", None))
    parent_delta = _number(get_value(latest, "parent_delta", None))
    decision = _normalized(get_value(latest, "decision", None))
    parent_eligible = bool(get_value(latest, "parent_eligible", False))
    gauc_delta = _metric_delta(latest, "gauc")
    ndcg_delta = _metric_delta(latest, "ndcg@5", "ndcg")
    family = str(get_value(latest, "family", ""))
    status = _normalized(get_value(latest, "status", None))
    method_ids = {
        str(item) for item in as_list(get_value(latest, "method_card_ids", None))
    }
    is_pairwise = "objective_pairwise_bpr" in method_ids
    exploratory_full_public = (
        verdict == "inconclusive"
        and stability == "confirmed"
        and decision in {"accept", "accepted"}
        and parent_eligible
    )
    clean_full_public = (
        (verdict in {"accepted", "verified"} or exploratory_full_public)
        and integrity == "clean"
        and fidelity == "full"
        and population == "public_validation"
        and output_accepted is True
        and stability in {"single_seed", "confirmed", "not_applicable"}
        and prediction_change is not None
    )
    parent = _latest_parent(latest, eligible)

    if (
        status == "invalid"
        and get_value(latest, "primary_score", None) is None
        and get_value(latest, "metric_set", None) is None
    ):
        return _next_independent_choice(
            context,
            eligible,
            allowed,
            family,
            reason_code="OPERATIONAL_FAILURE_UNTESTED",
            reason=(
                "The latest experiment ended before verified evaluation; do not "
                "interpret it as research evidence and continue with an independent "
                "eligible mechanism."
            ),
        ) or _blocked(
            "NO_ELIGIBLE_METHOD",
            "The failed operational attempt produced no research result and no independent method remains.",
        )

    for rule in _rule_order(context):
        if rule == "output_rejected" and output_accepted is False:
            return _blocked(
                "OUTPUT_CHECK_REJECTED",
                "The latest output failed structural or contract validation and must recover.",
            )
        if rule == "suspicious_or_compromised" and (
            integrity == "compromised" or verdict == "suspicious"
        ):
            return _blocked(
                "SUSPICIOUS_RESULT_REQUIRES_QUARANTINE",
                "The latest evaluation is suspicious or integrity-compromised.",
            )
        if rule == "unstable" and stability == "unstable":
            return _blocked(
                "UNSTABLE_RESULT_REQUIRES_CONFIRMATION",
                "The latest result requires seed confirmation before branching.",
            )
        if rule == "non_public_or_incomplete" and (
            not clean_full_public
        ):
            terminal_clean_full = (
                output_accepted is True
                and fidelity == "full"
                and population == "public_validation"
                and integrity == "clean"
                and stability in {"single_seed", "confirmed", "not_applicable"}
                and prediction_change is not None
                and (
                    decision in {"prune", "reject"}
                    or verdict in {"negative", "inconclusive"}
                )
            )
            if terminal_clean_full:
                portfolio_choice = _soft_prune_choice(
                    context,
                    latest,
                    eligible,
                    allowed,
                )
                if portfolio_choice is not None:
                    return portfolio_choice
                return _next_independent_choice(
                    context,
                    eligible,
                    allowed,
                    family,
                    reason_code="TERMINAL_FULL_RESULT_REJECTED",
                    reason=(
                        "The clean full result was rejected as a checkpoint and has "
                        "no authorized portfolio action; move to an independent method."
                    ),
                ) or _blocked(
                    "NO_ELIGIBLE_METHOD", "No independent eligible method remains."
                )
            return _blocked(
                "RESULT_NOT_BRANCHABLE",
                "Only a completed public-validation result may drive a research branch.",
            )
        if rule == "promotion_required" and fidelity in {"smoke", "proxy"}:
            if decision in {"prune", "reject"}:
                portfolio_choice = _soft_prune_choice(
                    context,
                    latest,
                    eligible,
                    allowed,
                )
                if portfolio_choice is not None:
                    return portfolio_choice
                return _next_independent_choice(
                    context,
                    eligible,
                    allowed,
                    family,
                    reason_code="EARLY_FIDELITY_REJECTED",
                    reason=(
                        "The latest mechanism was terminally rejected before full "
                        "evaluation; move to the next independent method."
                        + _diagnostic_suffix(latest)
                    ),
                ) or _blocked(
                    "NO_ELIGIBLE_METHOD", "No independent eligible method remains."
                )
            return _blocked(
                "FIDELITY_PROMOTION_REQUIRED",
                "A smoke or proxy result must be promoted or rejected before branching.",
            )
        if not clean_full_public:
            continue
        if (
            rule == "pairwise_gauc_up_ndcg_down"
            and is_pairwise
            and gauc_delta is not None
            and ndcg_delta is not None
            and gauc_delta > epsilon
            and ndcg_delta < -epsilon
        ):
            return _required_method_choice(
                context,
                parent,
                "objective",
                "objective_listwise_user_softmax",
                reason_code="PAIRWISE_GAUC_UP_NDCG_DOWN",
                reason="Pairwise improved GAUC but hurt nDCG@5; test one listwise objective.",
            )
        if (
            rule == "pairwise_gauc_down_ndcg_up"
            and is_pairwise
            and gauc_delta is not None
            and ndcg_delta is not None
            and gauc_delta < -epsilon
            and ndcg_delta > epsilon
        ):
            return _required_method_choice(
                context,
                parent,
                "objective",
                "objective_listwise_user_softmax",
                reason_code="PAIRWISE_GAUC_DOWN_NDCG_UP",
                reason="Pairwise improved top-5 placement but hurt broad ordering; test a bounded hybrid/listwise objective.",
            )
        if (
            rule == "pairwise_both_up"
            and is_pairwise
            and gauc_delta is not None
            and ndcg_delta is not None
            and gauc_delta > epsilon
            and ndcg_delta > epsilon
        ):
            return _required_method_choice(
                context,
                parent,
                "objective",
                "objective_pairwise_bpr",
                reason_code="PAIRWISE_BOTH_METRICS_UP",
                reason="Confirm or atomically refine the successful pairwise mechanism.",
            )
        if (
            rule == "meaningful_no_gain"
            and parent_delta is not None
            and abs(parent_delta) <= epsilon
            and prediction_change is not None
            and prediction_change > no_op_threshold
        ):
            exploration_parent = _best_experimental_parent(eligible)
            refinement = _same_family_refinement_choice(
                context,
                exploration_parent,
                allowed,
            )
            if refinement is not None:
                return refinement
            exploration_family = str(exploration_parent.family or family)
            return _next_independent_choice(
                context,
                eligible,
                allowed,
                exploration_family,
                reason_code="MEANINGFUL_CHANGE_NO_GAIN",
                reason=(
                    "The strongest eligible experimental path has no remaining "
                    "same-family refinement; move to the next independent mechanism."
                ),
                preferred_parent=exploration_parent,
            ) or _blocked(
                "NO_ELIGIBLE_METHOD",
                "No independent eligible method remains.",
            )
        if (
            rule == "trusted_improvement"
            and parent_delta is not None
            and parent_delta > epsilon
        ):
            if family not in allowed:
                return _blocked(
                    "REQUIRED_FAMILY_UNAVAILABLE",
                    "The successful family is no longer allowed by the frozen contract.",
                )
            preferred = next(iter(method_ids), None) if len(method_ids) == 1 else None
            card = _method_for_family(
                context,
                family,
                preferred=preferred,
                parent_experiment_id=parent.experiment_id,
            )
            if card is None:
                return _blocked(
                    "REQUIRED_METHOD_UNAVAILABLE",
                    "No eligible method can confirm or refine the successful family.",
                )
            return _proposal(
                parent=parent,
                family=family,
                card=card,
                phase="playbook",
                reason_code="TRUSTED_FULL_IMPROVEMENT",
                reason="Deepen the same family after a trusted full improvement.",
            )
        if (
            rule == "trusted_regression"
            and parent_delta is not None
            and parent_delta < -epsilon
        ):
            return _next_independent_choice(
                context,
                eligible,
                allowed,
                family,
                reason_code="TRUSTED_FULL_REGRESSION",
                reason=(
                    "The tested mechanism regressed beyond tolerance; move to an "
                    "independent method." + _diagnostic_suffix(latest)
                ),
            ) or _blocked("NO_ELIGIBLE_METHOD", "No independent eligible method remains.")
    return None


class SearchPolicy:
    """Select a legal parent and method; optional rankers see legal choices only."""

    def __init__(
        self,
        frontier_limit: int = 3,
        legal_choice_ranker: LegalChoiceRanker | None = None,
    ):
        if frontier_limit < 1:
            raise ValueError("frontier_limit must be positive")
        self.frontier_limit = frontier_limit
        self.legal_choice_ranker = legal_choice_ranker or LinUCBLegalChoiceRanker()

    def _rank(self, candidates: Sequence[PolicyChoice], context: Any) -> PolicyChoice:
        if not candidates:
            raise ValueError("cannot rank an empty legal choice set")
        if self.legal_choice_ranker is None:
            return candidates[0]
        ranked = self.legal_choice_ranker(tuple(candidates), context)
        return ranked if ranked in candidates else candidates[0]

    def choose(self, context: Any) -> PolicyChoice:
        graph = GraphView.from_context(context)
        eligible = list(graph.eligible_parents())
        allowed = _allowed_families(context)
        history = _family_history(context)
        if not eligible:
            return _blocked(
                "NO_ELIGIBLE_PARENT",
                "No verified full-fidelity parent is available in the planner context.",
                phase="none",
            )
        if not allowed:
            return _blocked(
                "NO_LEGAL_FAMILY",
                "The frozen contract exposes no legal experiment family.",
                phase="none",
            )
        if not method_card_map(context):
            return _blocked(
                "NO_METHOD_CARDS",
                "The planner context contains no validated method cards.",
                phase="none",
            )
        if _playbook(context) is None:
            return _blocked(
                "PLAYBOOK_MISSING",
                "The planner context is missing the validated improvement playbook.",
                phase="none",
            )
        if not _playbook_is_valid(context):
            return _blocked(
                "PLAYBOOK_INVALID",
                "The planner context contains an invalid improvement playbook.",
                phase="none",
            )

        no_op_candidates = _no_op_choices(context, eligible, allowed)
        if no_op_candidates is not None:
            if no_op_candidates:
                return self._rank(no_op_candidates, context)
            return _blocked(
                "NO_ELIGIBLE_METHOD",
                "The latest candidate was a no-op and no legal reimplementation "
                "or independent method remains.",
            )

        routed = _playbook_choice(context, eligible, allowed)
        if routed is not None:
            return routed

        frontier = _depth_first_frontier(graph, eligible, self.frontier_limit)
        recent = set(history[-2:])
        for parent in frontier:
            depth: list[PolicyChoice] = []
            for family in allowed:
                card = _method_for_family(
                    context,
                    family,
                    parent_experiment_id=parent.experiment_id,
                )
                if card is None:
                    continue
                depth.append(
                    _proposal(
                        parent=parent,
                        family=family,
                        card=card,
                        phase="depth",
                        reason_code="SCORE_GUIDED_DEPTH_FIRST",
                        reason=(
                            "Continue depth-first from trusted branch %s using legal "
                            "method %s; backtrack only after this branch is exhausted."
                            % (parent.experiment_id, get_value(card, "method_id", ""))
                        ),
                    )
                )
            depth.sort(
                key=lambda choice: (
                    choice.family in recent,
                    allowed.index(choice.family),
                )
            )
            if depth:
                return self._rank(depth, context)
        return _blocked(
            "NO_ELIGIBLE_METHOD",
            "No candidate method satisfies status, data, prerequisite, prohibition and family gates.",
            phase="none",
        )
