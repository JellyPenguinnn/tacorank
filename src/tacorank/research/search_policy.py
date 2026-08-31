"""Deterministic, playbook-driven, score-guided depth-first search policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from .portfolio import (
    COMPOSITION_DISTILLATION_OBJECTIVES,
    COMPOSITION_LOSS_ALIGNED_REFINEMENTS,
    COMPOSITION_MAX_METHODS,
    HIGH_VALUE_FAMILIES,
)
from .search_eligibility import classify_search_eligibility


DEFAULT_RULE_ORDER = REQUIRED_RULE_ORDER
DEFAULT_FAMILY_ORDER = HIGH_VALUE_FAMILIES + (
    "sampling",
    "ensemble",
    "evaluation",
    "other",
)
DEFAULT_METHOD_ORDER = {
    "objective": (
        "objective_pairwise_bpr",
        "objective_weighted_cross_entropy",
        "objective_distill_softmax",
        "objective_loss_aligned_features",
        "objective_listwise_user_softmax",
    ),
    "temporal_history": (
        "temporal_causal_history_features",
        "temporal_history_compact",
        "temporal_deep_interest_network",
        "temporal_search_interest_model",
        "temporal_time_series_interest",
    ),
    "multitask": (
        "multitask_single_auxiliary",
        "multitask_shared_bottom",
        "multitask_gsu",
        "multitask_esu",
        "multitask_mmoe",
        "multitask_ple",
    ),
    "duration_bias": ("duration_bias_censored_watch_time",),
    "features": (
        "features_general_bounded_engineering",
        "temporal_drift_past_only",
    ),
    "model": (
        "model_field_aware_fm",
        "model_deep_cross_network",
        "model_lhuc",
        "model_compact_ranker",
    ),
    "ensemble": (
        "ensemble_parallel_round_synthesis",
        "ensemble_diverse_residual_candidate",
        "ensemble_confirmed_members",
    ),
    "evaluation": ("evaluation_random_exposure_robustness",),
}

# These are ordered preferences inside each compatibility slot.  The provider
# still receives every method card, including alternatives that are not chosen
# for this stack, so it can research and explain the trade-offs rather than
# treating the first card as universally optimal.
AGGRESSIVE_PRIMARY_OBJECTIVES = (
    "objective_pairwise_bpr",
    "objective_weighted_cross_entropy",
    "objective_listwise_user_softmax",
)
AGGRESSIVE_DISTILLATION_OBJECTIVES = ("objective_distill_softmax",)
AGGRESSIVE_FEATURE_METHODS = (
    "features_general_bounded_engineering",
    "features_tab_context_residual",
    "features_author_affinity_past_only",
    "temporal_drift_past_only",
)
AGGRESSIVE_INTEREST_METHODS = (
    "temporal_causal_history_features",
    "temporal_search_interest_model",
    "temporal_deep_interest_network",
    "temporal_time_series_interest",
    "temporal_history_compact",
)
AGGRESSIVE_SINGLE_TASK_BACKBONES = (
    "model_deep_cross_network",
    "model_compact_ranker",
    "model_field_aware_fm",
)
AGGRESSIVE_MULTITASK_BACKBONES = (
    "multitask_ple",
    "multitask_mmoe",
    "multitask_esu",
    "multitask_gsu",
    "multitask_shared_bottom",
    "multitask_single_auxiliary",
)


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
    method_card_ids: tuple[str, ...] = ()
    component_experiment_ids: tuple[str, ...] = ()
    batch_role: str | None = None
    hypothesis_group_id: str | None = None

    @property
    def choice_id(self) -> str:
        """Stable opaque identity for controller-side action validation."""

        parent = getattr(self.parent, "experiment_id", "baseline")
        components = ",".join(self.component_experiment_ids)
        methods = ",".join(self.selected_method_card_ids)
        return "choice_%s" % "_".join(
            str(item or "none")
            for item in (parent, self.family, methods, self.phase, components)
        ).replace("-", "_")

    @property
    def selected_method_card_ids(self) -> tuple[str, ...]:
        """Return the complete method identity, including compatible stacks."""

        if self.method_card_ids:
            return self.method_card_ids
        return (self.method_card_id,) if self.method_card_id else ()


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
    allow_repeated: bool = False,
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
    attempted = (
        set()
        if allow_repeated
        else _attempted_methods_for_parent(context, parent_experiment_id)
    )
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


def _ordered_eligible_method_cards(context: Any, family: str) -> tuple[Any, ...]:
    """Return every eligible card in the playbook's deterministic order."""

    by_id = {
        str(get_value(card, "method_id", "")): card
        for card in eligible_method_cards(context, family)
    }
    ordered = [
        by_id.pop(method_id)
        for method_id in _method_order(context, family)
        if method_id in by_id
    ]
    ordered.extend(by_id[method_id] for method_id in sorted(by_id))
    return tuple(ordered)


def _cost_tier(value: Any) -> str:
    tier = get_value(value, "cost_tier", value)
    normalized = _normalized(tier)
    return normalized if normalized in {"low", "medium", "high"} else "medium"


def _highest_cost_tier(cards: Sequence[Any]) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    tiers = [_cost_tier(get_value(card, "cost_tier", None)) for card in cards]
    return max(tiers, key=lambda tier: order[tier], default="medium")


def _aggressive_composition_enabled(context: Any) -> bool:
    value = get_value(
        get_value(context, "contract_summary", None),
        "aggressive_composition_enabled",
        False,
    )
    return value is True or str(value).strip().lower() == "true"


def _max_composed_methods(context: Any) -> int:
    value = get_value(
        get_value(context, "contract_summary", None),
        "max_composed_methods",
        COMPOSITION_MAX_METHODS,
    )
    try:
        return max(2, min(COMPOSITION_MAX_METHODS, int(value)))
    except (TypeError, ValueError):
        return COMPOSITION_MAX_METHODS


def _aggressive_composition_choice(
    context: Any,
    parent: ExperimentNodeView,
    allowed: tuple[str, ...],
) -> PolicyChoice | None:
    """Build the opt-in stack from compatible slots and legal cards."""

    if not _aggressive_composition_enabled(context) or "composition" not in allowed:
        return None
    if "objective" not in allowed or "features" not in allowed:
        return None

    def ordered_cards(family: str, method_ids: Sequence[str]) -> tuple[Any, ...]:
        candidates = {
            str(get_value(card, "method_id", "")): card
            for card in eligible_method_cards(context, family)
        }
        return tuple(
            candidates[method_id]
            for method_id in method_ids
            if method_id in candidates
        )

    primary = ordered_cards("objective", AGGRESSIVE_PRIMARY_OBJECTIVES)
    features = ordered_cards("features", AGGRESSIVE_FEATURE_METHODS)
    if not primary or not features:
        return None

    multitask = ordered_cards("multitask", AGGRESSIVE_MULTITASK_BACKBONES)
    single_task = ordered_cards("model", AGGRESSIVE_SINGLE_TASK_BACKBONES)
    if multitask:
        backbone = multitask[0]
    elif single_task:
        backbone = single_task[0]
    else:
        return None

    # Reserve the three essential roles first so a deliberately smaller
    # operator limit cannot produce a loss-only or feature-only stack.
    cards: list[Any] = [primary[0], features[0], backbone]

    def append_card(card: Any | None) -> None:
        if card is None:
            return
        method_id = str(get_value(card, "method_id", ""))
        if method_id and method_id not in {
            str(get_value(item, "method_id", "")) for item in cards
        }:
            cards.append(card)

    # Distillation is an additive teacher constraint; the other objective
    # cards remain alternatives because their losses replace one another.
    distillation = ordered_cards("objective", AGGRESSIVE_DISTILLATION_OBJECTIVES)
    append_card(distillation[0] if distillation else None)

    for card in features[1:]:
        append_card(card)

    if "temporal_history" in allowed:
        interest = ordered_cards("temporal_history", AGGRESSIVE_INTEREST_METHODS)
        append_card(interest[0] if interest else None)

    if not multitask and str(get_value(backbone, "method_id", "")) in {
        "model_deep_cross_network",
        "model_compact_ranker",
    }:
        adapter = ordered_cards("model", ("model_lhuc",))
        append_card(adapter[0] if adapter else None)

    if "duration_bias" in allowed:
        duration = ordered_cards(
            "duration_bias", ("duration_bias_censored_watch_time",)
        )
        append_card(duration[0] if duration else None)
    if "sampling" in allowed:
        sampler = ordered_cards("sampling", ("sampling_deterministic_coverage",))
        append_card(sampler[0] if sampler else None)

    # This is a later-stage additive refinement. It is only selected when the
    # pairwise-tested card is eligible and no second loss is being added.
    if not any(
        str(get_value(item, "method_id", ""))
        in COMPOSITION_DISTILLATION_OBJECTIVES
        for item in cards
    ):
        refinement = ordered_cards("objective", COMPOSITION_LOSS_ALIGNED_REFINEMENTS)
        append_card(refinement[0] if refinement else None)

    cards = cards[: _max_composed_methods(context)]
    if len(cards) < 3:
        return None
    method_ids = tuple(str(get_value(card, "method_id", "")) for card in cards)
    return _proposal(
        parent=parent,
        family="composition",
        card=cards[0],
        method_cards=cards,
        phase="composition",
        reason_code="AGGRESSIVE_COMPATIBLE_COMPOSITION",
        reason=(
            "Deep-dive one bounded compatible stack across the selected ranking "
            "objective, leakage-safe feature residuals, interest/model layers, "
            "and legal training add-ons. Treat same-slot cards as alternatives "
            "and resolve overlapping edits explicitly: %s."
            % ", ".join(method_ids)
        ),
    )


def _proposal(
    *,
    parent: ExperimentNodeView,
    family: str,
    card: Any,
    phase: str,
    reason_code: str,
    reason: str,
    component_experiment_ids: tuple[str, ...] = (),
    batch_role: str | None = None,
    hypothesis_group_id: str | None = None,
    method_cards: Sequence[Any] = (),
) -> PolicyChoice:
    selected_cards = tuple(method_cards) or (card,)
    method_ids = tuple(
        str(get_value(item, "method_id", "")) for item in selected_cards
    )
    return PolicyChoice(
        action="propose",
        parent=parent,
        family=family,
        cost_tier=_highest_cost_tier(selected_cards),
        phase=phase,
        reason_code=reason_code,
        reason=reason,
        method_card_id=method_ids[0],
        method_card_ids=method_ids,
        component_experiment_ids=component_experiment_ids,
        batch_role=batch_role,
        hypothesis_group_id=hypothesis_group_id,
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
    # A near-best exploratory result is useful evidence, but stacking another
    # mechanism on top of it compounds the very noise the confirmation ladder
    # is meant to control.  Prefer a trusted experimental parent whenever one
    # exists; use an exploratory parent only when it is the only measured path.
    trusted = [node for node in experimental if node.is_trusted]
    return _best_parent(trusted or experimental or eligible)


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
    latest_node = next(
        (node for node in eligible if node.experiment_id == latest_id),
        _best_parent(eligible),
    )
    if latest_node.is_exploratory_parent:
        trusted = [
            node
            for node in eligible
            if not node.is_root and node.is_trusted
        ]
        if trusted:
            return _best_parent(trusted)
    return latest_node


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
        parent = _best_experimental_parent(eligible)
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
        if rule == "suspicious_or_compromised" and integrity == "compromised":
            return _blocked(
                "SUSPICIOUS_RESULT_REQUIRES_QUARANTINE",
                "The latest evaluation is integrity-compromised and requires "
                "operator quarantine.",
            )
        if rule == "suspicious_or_compromised" and verdict == "suspicious":
            return _next_independent_choice(
                context,
                eligible,
                allowed,
                family,
                reason_code="SUSPICIOUS_RESULT_QUARANTINED",
                reason=(
                    "Quarantine the suspicious result as non-reward evidence and "
                    "continue from a verified eligible parent with an independent "
                    "method."
                ),
            ) or _blocked(
                "NO_ELIGIBLE_METHOD",
                "The suspicious result was quarantined and no independent eligible "
                "method remains.",
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

        # A fresh opt-in deployment gets one deliberately bounded interaction
        # test before the ordinary atomic playbook route.  Historical configs
        # have this flag unset, so their policy and action identities do not
        # change.
        if not history:
            composition = _aggressive_composition_choice(
                context, _best_experimental_parent(eligible), allowed
            )
            if composition is not None:
                return composition

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

    def legal_choices(self, context: Any) -> tuple[PolicyChoice, ...]:
        """Return the controller-approved open search set for one checkpoint.

        Mandatory safety/playbook routes intentionally collapse to one choice.
        Only the ordinary depth-search route exposes alternatives to the
        bounded agent, and every alternative is generated by the same legal
        parent, family, method, data, and cost gates as ``choose``.
        """

        selected = self.choose(context)
        if selected.action != "propose":
            return (selected,)
        if selected.reason_code != "SCORE_GUIDED_DEPTH_FIRST":
            return (selected,)
        choices = self._parallel_choices(context, direction_count=None)
        if not choices:
            return (selected,)
        by_id = {choice.choice_id: choice for choice in choices}
        if selected.choice_id not in by_id:
            choices = (selected,) + tuple(choices)
        return tuple(choices[:24])

    def choose_parallel_direction(
        self, context: Any, direction_index: int, direction_count: int
    ) -> PolicyChoice:
        """Choose one legal, independently testable lane for a parallel round.

        Each lane is pinned to a different legal parent/method identity.  A
        method card may be reused from a different eligible parent, but the
        same method cannot be proposed twice from one parent because that is a
        duplicate experiment under the schema-v1 identity contract.
        """

        if direction_index < 0 or direction_index >= direction_count:
            raise ValueError("parallel direction index is out of range")
        if direction_count in (2, 4):
            batch = self.choose_parallel_batch(context, direction_count)
            if direction_index < len(batch):
                return batch[direction_index]
        choices = self._parallel_choices(context, direction_count)
        if not choices:
            return self.choose(context)
        # A parallel round is sealed against one snapshot, so rank the legal
        # arms using only evidence already present in this run.  With no
        # history the original deterministic playbook order is retained.
        if _family_history(context) and self.legal_choice_ranker is not None:
            choices = self._rank_parallel_choices(choices, context)
        if direction_index >= len(choices):
            raise ValueError(
                "parallel direction index exceeds unique legal parent/method choices"
            )
        return replace(
            choices[direction_index],
            reason_code="PARALLEL_DIRECTION_%d_OF_%d"
            % (direction_index + 1, direction_count),
        )

    def choose_parallel_batch(
        self, context: Any, direction_count: int
    ) -> tuple[PolicyChoice, ...]:
        """Compose an adaptive portfolio while retaining legal action identity.

        Exploration spreads across families. Exploitation reserves two slots for
        sibling refinements of the strongest stable parent, then adds a metric-
        complementary arm and an independent challenger. Narrowing uses one
        conservative confirmation and one challenger.
        """

        if direction_count < 1:
            return ()
        choices = list(self._parallel_choices(context, direction_count))
        if not choices:
            return ()
        if _family_history(context) and self.legal_choice_ranker is not None:
            choices = list(self._rank_parallel_choices(choices, context))

        history = as_list(get_value(context, "family_history", None))
        stable = [
            item
            for item in history
            if _normalized(get_value(item, "trust_verdict", None)) in {"accepted", "verified"}
            and _normalized(get_value(item, "integrity", None)) == "clean"
            and _normalized(get_value(item, "highest_completed_fidelity", None)) == "full"
            and _normalized(get_value(item, "stability", None)) == "confirmed"
        ]
        selected: list[PolicyChoice] = []
        if direction_count == 2:
            parent = _best_experimental_parent(list(GraphView.from_context(context).eligible_parents()))
            confirmation = next(
                (item for item in choices if item.parent and item.parent.experiment_id == parent.experiment_id),
                choices[0],
            )
            selected.append(confirmation)
            selected.extend(item for item in choices if item not in selected)
        elif not stable:
            # Round-robin family diversity is more useful than four variants of
            # the first family in a cold-start round.
            for family in _family_order(context):
                item = next((candidate for candidate in choices if candidate.family == family), None)
                if item is not None and item not in selected:
                    selected.append(item)
                if len(selected) >= direction_count:
                    break
            selected.extend(item for item in choices if item not in selected)
        else:
            strongest = sorted(
                stable,
                key=lambda item: -(_number(get_value(item, "seed_mean", None)) or _number(get_value(item, "primary_score", None)) or float("-inf")),
            )[0]
            strongest_id = str(get_value(strongest, "experiment_id", ""))
            strongest_family = str(get_value(strongest, "family", ""))
            siblings = [
                item for item in choices
                if item.parent and item.parent.experiment_id == strongest_id
                and item.family == strongest_family
            ]
            if not siblings:
                siblings = [item for item in choices if item.family == strongest_family]
            selected.extend(siblings[:2])
            selected.extend(
                item for item in choices
                if item not in selected and item.family != strongest_family
            )
            selected.extend(item for item in choices if item not in selected)

        selected = selected[:direction_count]
        if len(selected) < min(direction_count, len(choices)):
            selected.extend(item for item in choices if item not in selected)
            selected = selected[:direction_count]
        if direction_count == 2:
            roles = ["confirmation", "challenger"]
        elif not stable:
            roles = ["exploration"] * direction_count
        else:
            sibling_count = min(
                2,
                sum(
                    1
                    for item in selected
                    if item.family == strongest_family
                    and item.parent
                    and item.parent.experiment_id == strongest_id
                ),
            )
            roles = ["sibling_refinement"] * sibling_count
            if len(roles) < direction_count:
                roles.append("complementary")
            if len(roles) < direction_count:
                roles.append("challenger")
            roles.extend(["challenger"] * max(0, direction_count - len(roles)))
        output = []
        for index, item in enumerate(selected):
            group = item.hypothesis_group_id
            if roles[index] == "sibling_refinement":
                group = "hg_%s_%s" % (item.parent.experiment_id if item.parent else "baseline", item.family)
            output.append(
                replace(item, batch_role=roles[index], hypothesis_group_id=group)
            )
        return tuple(output)

    def _rank_parallel_choices(
        self, choices: Sequence[PolicyChoice], context: Any
    ) -> tuple[PolicyChoice, ...]:
        """Rank all legal lanes without changing the legal action set."""

        remaining = list(choices)
        ranked = []
        while remaining:
            selected = self._rank(remaining, context)
            if selected not in remaining:
                selected = remaining[0]
            ranked.append(selected)
            remaining.remove(selected)
        return tuple(ranked)

    def parallel_direction_capacity(self, context: Any) -> int:
        """Return the number of unique legal parent/method lanes at a checkpoint."""

        return len(self._parallel_choices(context, direction_count=None))

    def _parallel_choices(
        self, context: Any, direction_count: int | None
    ) -> tuple[PolicyChoice, ...]:
        """Build the same legal lane set used by capacity and lane selection.

        Parallel planning is sealed against one planner snapshot.  Therefore
        the policy must account for identities already present in that
        snapshot before asking the provider for more directions.  The old
        implementation counted every eligible method card for only the best
        parent, even when that parent had already exhausted those cards.
        """

        graph = GraphView.from_context(context)
        eligible = list(graph.eligible_parents())
        allowed = _allowed_families(context)
        if not eligible or not allowed or not method_card_map(context):
            return ()

        # Keep the strongest parent first, then deterministically backtrack to
        # other verified parents once its method-card identities are spent.
        trusted_experimental = [
            node for node in eligible if not node.is_root and node.is_trusted
        ]
        exploratory_experimental = [
            node
            for node in eligible
            if not node.is_root and not node.is_trusted
        ]
        roots = [node for node in eligible if node.is_root]
        parents = []
        for group in (trusted_experimental, exploratory_experimental, roots):
            parents.extend(
                sorted(
                    group,
                    key=lambda node: (
                        -_score(node),
                        node.child_count,
                        node.experiment_id,
                    ),
                )
            )
        choices: list[PolicyChoice] = []
        if not _family_history(context):
            composition = _aggressive_composition_choice(
                context, _best_experimental_parent(eligible), allowed
            )
            if composition is not None:
                choices.append(composition)
        for parent in parents:
            attempted = _attempted_methods_for_parent(
                context, parent.experiment_id
            )
            for family in allowed:
                if family == "ensemble":
                    continue
                for card in _ordered_eligible_method_cards(context, family):
                    method_id = str(get_value(card, "method_id", ""))
                    if method_id in attempted:
                        continue
                    choices.append(
                        _proposal(
                            parent=parent,
                            family=family,
                            card=card,
                            phase="parallel_round",
                            reason_code="PARALLEL_DIRECTION_%d_OF_%d"
                            % (
                                len(choices) + 1,
                                direction_count or len(choices) + 1,
                            ),
                            reason=(
                                "Produce atomic direction using the distinct legal "
                                "parent/method identity %s/%s."
                                % (parent.experiment_id, method_id)
                            ),
                        )
                    )
        return tuple(choices)

    def choose_synthesis(
        self, context: Any, component_experiment_ids: Sequence[str]
    ) -> PolicyChoice:
        """Create one agent-implemented alignment candidate from round winners."""

        graph = GraphView.from_context(context)
        eligible_by_id = {
            node.experiment_id: node for node in graph.eligible_parents()
        }
        components = [
            eligible_by_id[experiment_id]
            for experiment_id in component_experiment_ids
            if experiment_id in eligible_by_id
        ]
        if len(components) < 2:
            return _blocked(
                "INSUFFICIENT_PARALLEL_IMPROVEMENTS",
                "Synthesis requires at least two independently accepted round members.",
                phase="synthesis",
            )
        parent = _best_parent(components)
        card = _method_for_family(
            context,
            "ensemble",
            preferred="ensemble_parallel_round_synthesis",
            parent_experiment_id=parent.experiment_id,
            allow_repeated=True,
        )
        if card is None:
            return _blocked(
                "SYNTHESIS_METHOD_UNAVAILABLE",
                "The parallel synthesis method is not legal at this checkpoint.",
                phase="synthesis",
            )
        secondary = tuple(
            node.experiment_id
            for node in components
            if node.experiment_id != parent.experiment_id
        )
        return _proposal(
            parent=parent,
            family="ensemble",
            card=card,
            phase="synthesis",
            reason_code="PARALLEL_ROUND_SYNTHESIS",
            reason=(
                "Align every independently accepted round improvement on the best "
                "member, resolve interaction conflicts, and produce one gated candidate."
            ),
            component_experiment_ids=secondary,
        )
