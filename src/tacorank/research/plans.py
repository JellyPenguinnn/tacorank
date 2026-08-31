"""First-class adaptive research plans above atomic experiment proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .graph_view import as_list, get_value


@dataclass(frozen=True)
class ResearchPlanDefinition:
    plan_id: str
    research_question: str
    families: tuple[str, ...]
    method_card_ids: tuple[str, ...]
    maximum_experiments: int
    conditional_followups: Mapping[str, str]


RESEARCH_PLANS: tuple[ResearchPlanDefinition, ...] = (
    ResearchPlanDefinition(
        plan_id="objective_alignment",
        research_question=(
            "Does within-user ranking loss outperform the copied pointwise FM?"
        ),
        families=("objective",),
        method_card_ids=(
            "objective_pairwise_bpr",
            "objective_loss_aligned_features",
            "objective_listwise_user_softmax",
        ),
        maximum_experiments=4,
        conditional_followups={
            "clear_gain": "seed_confirmation_then_ablation",
            "gauc_up_ndcg_down": "objective_listwise_user_softmax",
            "no_op": "implementation_diagnosis",
            "confirmed_regression": "close_after_second_distinct_test",
        },
    ),
    ResearchPlanDefinition(
        plan_id="behavioral_history",
        research_question=(
            "Does strictly past candidate-conditioned behavior add ranking signal?"
        ),
        families=("temporal_history", "features"),
        method_card_ids=(
            "temporal_history_compact",
            "features_history_affinity",
            "features_author_affinity_past_only",
            "temporal_drift_past_only",
        ),
        maximum_experiments=4,
        conditional_followups={
            "clear_gain": "recency_or_component_ablation",
            "within_noise": "one_stronger_controlled_history_test",
            "no_op": "implementation_diagnosis",
            "confirmed_regression": "switch_history_mechanism_once",
        },
    ),
    ResearchPlanDefinition(
        plan_id="auxiliary_learning",
        research_question=(
            "Can legal click or censored watch-time supervision improve long-view ranking?"
        ),
        families=("multitask", "duration_bias"),
        method_card_ids=(
            "multitask_single_auxiliary",
            "duration_bias_censored_watch_time",
        ),
        maximum_experiments=4,
        conditional_followups={
            "clear_gain": "auxiliary_weight_ablation",
            "primary_hurt": "close_negative_transfer",
            "no_op": "implementation_diagnosis",
        },
    ),
    ResearchPlanDefinition(
        plan_id="temporal_robustness",
        research_question=(
            "Can past-only drift and context interactions improve chronological generalization?"
        ),
        families=("features", "sampling", "evaluation"),
        method_card_ids=(
            "features_tab_context_residual",
            "sampling_deterministic_coverage",
            "evaluation_random_exposure_robustness",
        ),
        maximum_experiments=3,
        conditional_followups={
            "clear_gain": "time_slice_confirmation",
            "drift_conflict": "simplify_or_close",
            "confirmed_regression": "close_plan",
        },
    ),
    ResearchPlanDefinition(
        plan_id="model_and_ensemble",
        research_question=(
            "Do confirmed signals require nonlinear modeling or complementary combination?"
        ),
        families=("model", "ensemble"),
        method_card_ids=(
            "model_compact_ranker",
            "ensemble_diverse_residual_candidate",
            "ensemble_confirmed_members",
            "ensemble_parallel_round_synthesis",
        ),
        maximum_experiments=5,
        conditional_followups={
            "two_confirmed_diverse_members": "ensemble_confirmed_members",
            "redundant": "close_plan",
            "confirmed_regression": "backtrack_to_best_member",
        },
    ),
)


def plan_for_method(method_id: str) -> Optional[ResearchPlanDefinition]:
    return next(
        (plan for plan in RESEARCH_PLANS if method_id in plan.method_card_ids),
        None,
    )


def plan_progress(context: Any, plan: ResearchPlanDefinition) -> dict[str, Any]:
    """Derive plan status from verified ledger projections, never mutable state."""

    epsilon = float(
        get_value(get_value(context, "contract_summary", None), "epsilon", 0.002)
        or 0.002
    )
    history = []
    for summary in as_list(get_value(context, "family_history", None)):
        explicit_plan = str(get_value(summary, "plan_id", "") or "")
        methods = {
            str(item)
            for item in as_list(get_value(summary, "method_card_ids", None))
        }
        if explicit_plan == plan.plan_id or methods.intersection(
            plan.method_card_ids
        ):
            history.append(summary)
    valid = [
        item
        for item in history
        if bool(get_value(item, "execution_conformant", False))
        and str(get_value(item, "highest_completed_fidelity", "")).lower()
        in {"fidelity.full", "full"}
        and str(get_value(item, "integrity", "")).lower()
        in {"integrity.clean", "clean"}
    ]
    improvements = [
        item
        for item in valid
        if bool(get_value(item, "best_eligible", False))
        or float(get_value(item, "parent_delta", 0.0) or 0.0) > epsilon
    ]
    regressions = [
        item
        for item in valid
        if float(get_value(item, "parent_delta", 0.0) or 0.0) < -epsilon
    ]
    if len(history) >= plan.maximum_experiments:
        status = "exhausted"
    elif len(regressions) >= 2 and not improvements:
        status = "falsified"
    elif history:
        status = "active"
    else:
        status = "unstarted"
    return {
        "plan_id": plan.plan_id,
        "research_question": plan.research_question,
        "families": list(plan.families),
        "method_card_ids": list(plan.method_card_ids),
        "maximum_experiments": plan.maximum_experiments,
        "experiments_attempted": len(history),
        "valid_full_experiments": len(valid),
        "confirmed_improvements": len(improvements),
        "confirmed_regressions": len(regressions),
        "status": status,
        "conditional_followups": dict(plan.conditional_followups),
    }


def method_is_plan_eligible(context: Any, method_id: str) -> bool:
    plan = plan_for_method(method_id)
    if plan is None:
        return True
    return plan_progress(context, plan)["status"] not in {
        "exhausted",
        "falsified",
    }
