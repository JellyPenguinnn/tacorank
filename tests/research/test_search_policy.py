from types import SimpleNamespace

from tacorank.research.search_policy import SearchPolicy

from .conftest import make_summary


def test_policy_probes_untried_high_value_family_from_baseline(planner_context):
    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.phase == "breadth"
    assert choice.family == "objective"
    assert choice.parent.experiment_id == "exp_0000"


def test_policy_returns_to_trusted_frontier_with_family_diversity():
    root = make_summary("exp_0000", score=0.5946)
    best = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        score=0.61,
        parent_eligible=True,
        child_count=2,
    )
    other = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0000",
        commit_sha="c" * 40,
        family="model",
        score=0.60,
        parent_eligible=True,
        child_count=0,
    )
    context = SimpleNamespace(
        context_id="ctx_planner_000002",
        run_id="run_20260829_b",
        contract_summary=SimpleNamespace(allowed_families=["objective", "model", "ensemble"]),
        baseline=root,
        current_best=best,
        eligible_frontier=[root, best, other],
        family_history=[best, other, best],
    )

    choice = SearchPolicy().choose(context)

    assert choice.phase == "depth"
    assert choice.parent.experiment_id == "exp_0001"
    assert choice.family == "ensemble"


def test_playbook_routes_pairwise_gauc_up_ndcg_down_to_objective():
    root = make_summary("exp_0000", score=0.5946)
    pairwise = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        score=0.601,
        parent_eligible=True,
        metric_deltas={"gauc": 0.01, "ndcg@5": -0.004, "primary": 0.003},
        method_card_ids=["objective_pairwise_bpr"],
    )
    context = SimpleNamespace(
        contract_summary=SimpleNamespace(
            allowed_families=["objective", "temporal_history"],
            epsilon=0.002,
        ),
        eligible_frontier=[root, pairwise],
        family_history=[pairwise],
    )

    choice = SearchPolicy().choose(context)

    assert choice.phase == "playbook"
    assert choice.reason_code == "PAIRWISE_GAUC_UP_NDCG_DOWN"
    assert choice.parent.experiment_id == "exp_0001"
    assert choice.family == "objective"
    assert choice.method_card_id == "objective_listwise_user_softmax"
