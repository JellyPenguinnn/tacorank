from types import SimpleNamespace

from tacorank.research.graph_view import ExperimentNodeView
from tacorank.research.linucb import LinUCBLegalChoiceRanker
from tacorank.research.search_policy import PolicyChoice

from .conftest import make_summary


def _choice(context, family, method):
    parent = ExperimentNodeView.from_summary(context.baseline)
    return PolicyChoice(
        action="propose",
        parent=parent,
        family=family,
        cost_tier="medium",
        phase="depth",
        reason_code="TEST",
        reason="test",
        method_card_id=method,
    )


def test_linucb_prefers_untried_legal_arm_after_clean_negative(planner_context):
    planner_context.family_history = [
        make_summary(
            "exp_0001",
            parent_experiment_id="exp_0000",
            family="objective",
            parent_delta=-0.02,
            output_accepted=True,
            integrity="clean",
        )
    ]
    objective = _choice(planner_context, "objective", "objective_pairwise_bpr")
    history = _choice(
        planner_context, "temporal_history", "temporal_history_compact"
    )

    selected = LinUCBLegalChoiceRanker()([objective, history], planner_context)

    assert selected is history


def test_linucb_never_returns_choice_outside_supplied_legal_set(planner_context):
    choices = [
        _choice(planner_context, "objective", "objective_pairwise_bpr"),
        _choice(planner_context, "model", "model_compact_ranker"),
    ]

    selected = LinUCBLegalChoiceRanker()(choices, planner_context)

    assert selected in choices


def test_linucb_learns_method_specific_negative_feedback(planner_context):
    planner_context.historical_feedback = [
        SimpleNamespace(
            source_run_id="run_prior",
            experiment_id="exp_0042",
            parent_experiment_id="baseline",
            family="objective",
            method_card_ids=["objective_pairwise_bpr"],
            parent_stable_primary_score=0.5946,
            stable_primary_score=0.5746,
            risk_adjusted_reward=-0.02,
            seed_count=3,
            seed_stderr=0.0002,
            stability="confirmed",
            trust_verdict="negative",
            integrity="clean",
            trust_flags=[],
        )
    ]
    pairwise = _choice(planner_context, "objective", "objective_pairwise_bpr")
    listwise = _choice(
        planner_context, "objective", "objective_listwise_user_softmax"
    )

    selected = LinUCBLegalChoiceRanker()(
        [pairwise, listwise], planner_context
    )

    assert selected is listwise


def test_linucb_uses_risk_adjusted_positive_method_feedback(planner_context):
    planner_context.historical_feedback = [
        SimpleNamespace(
            source_run_id="run_prior",
            experiment_id="exp_0042",
            parent_experiment_id="baseline",
            family="objective",
            method_card_ids=["objective_pairwise_bpr"],
            parent_stable_primary_score=0.5946,
            stable_primary_score=0.6046,
            risk_adjusted_reward=0.009,
            seed_count=3,
            seed_stderr=0.0005,
            stability="confirmed",
            trust_verdict="accepted",
            integrity="clean",
            trust_flags=[],
        ),
        SimpleNamespace(
            source_run_id="run_prior",
            experiment_id="exp_0043",
            parent_experiment_id="baseline",
            family="objective",
            method_card_ids=["objective_listwise_user_softmax"],
            parent_stable_primary_score=0.5946,
            stable_primary_score=0.5906,
            risk_adjusted_reward=-0.005,
            seed_count=3,
            seed_stderr=0.0005,
            stability="confirmed",
            trust_verdict="negative",
            integrity="clean",
            trust_flags=[],
        ),
    ]
    pairwise = _choice(planner_context, "objective", "objective_pairwise_bpr")
    listwise = _choice(
        planner_context, "objective", "objective_listwise_user_softmax"
    )

    selected = LinUCBLegalChoiceRanker()(
        [pairwise, listwise], planner_context
    )

    assert selected is pairwise
