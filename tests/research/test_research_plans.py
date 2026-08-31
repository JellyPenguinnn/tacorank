from types import SimpleNamespace

from tacorank.research.plans import RESEARCH_PLANS, plan_for_method, plan_progress


def _context(history):
    return SimpleNamespace(
        contract_summary=SimpleNamespace(epsilon=0.002),
        family_history=history,
    )


def _result(experiment_id: str, delta: float):
    return SimpleNamespace(
        experiment_id=experiment_id,
        plan_id="objective_alignment",
        method_card_ids=["objective_pairwise_bpr"],
        execution_conformant=True,
        highest_completed_fidelity="full",
        integrity="clean",
        parent_delta=delta,
        best_eligible=delta > 0.002,
    )


def test_plan_catalog_maps_atomic_methods_to_research_questions() -> None:
    plan = plan_for_method("objective_pairwise_bpr")

    assert plan is not None
    assert plan.plan_id == "objective_alignment"
    assert plan.maximum_experiments > 1
    assert "ranking loss" in plan.research_question
    assert len(RESEARCH_PLANS) == 5


def test_plan_falsifies_after_two_distinct_confirmed_regressions() -> None:
    plan = plan_for_method("objective_pairwise_bpr")
    assert plan is not None

    progress = plan_progress(
        _context([_result("exp_0001", -0.003), _result("exp_0002", -0.004)]),
        plan,
    )

    assert progress["confirmed_regressions"] == 2
    assert progress["confirmed_improvements"] == 0
    assert progress["status"] == "falsified"


def test_one_confirmed_gain_keeps_plan_active_for_ablation() -> None:
    plan = plan_for_method("objective_pairwise_bpr")
    assert plan is not None

    progress = plan_progress(
        _context([_result("exp_0001", 0.003), _result("exp_0002", -0.003)]),
        plan,
    )

    assert progress["confirmed_improvements"] == 1
    assert progress["status"] == "active"
