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
        phase="breadth",
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
