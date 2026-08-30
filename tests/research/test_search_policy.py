from types import SimpleNamespace

import pytest

from tacorank.research.search_policy import PolicyChoice, SearchPolicy

from .conftest import make_summary


def test_policy_probes_untried_high_value_family_from_baseline(planner_context):
    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.phase == "breadth"
    assert choice.family == "objective"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.method_card_id == "objective_pairwise_bpr"


def context_with_latest(planner_context, latest, *, allowed_families=None):
    root = make_summary("exp_0000", score=0.5946)
    contract = SimpleNamespace(**vars(planner_context.contract_summary))
    if allowed_families is not None:
        contract.allowed_families = allowed_families
    frontier = [root]
    if latest.parent_eligible:
        frontier.append(latest)
    return SimpleNamespace(
        contract_summary=contract,
        baseline=root,
        current_best=latest if latest.parent_eligible else root,
        eligible_frontier=frontier,
        family_history=[latest],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )


def test_policy_returns_to_trusted_frontier_with_family_diversity(planner_context):
    root = make_summary("exp_0000", score=0.5946)
    best = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        score=0.61,
        parent_eligible=True,
        child_count=2,
        parent_delta=None,
    )
    other = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0000",
        commit_sha="c" * 40,
        family="model",
        score=0.60,
        parent_eligible=True,
        child_count=0,
        parent_delta=None,
    )
    prior_ensemble = make_summary(
        "exp_0003",
        parent_experiment_id="exp_0000",
        commit_sha="d" * 40,
        family="ensemble",
        parent_eligible=False,
        parent_delta=None,
    )
    contract = SimpleNamespace(**vars(planner_context.contract_summary))
    contract.allowed_families = ["objective", "model", "ensemble"]
    context = SimpleNamespace(
        context_id="ctx_planner_000002",
        run_id="run_20260829_b",
        contract_summary=contract,
        baseline=root,
        current_best=best,
        eligible_frontier=[root, best, other],
        family_history=[prior_ensemble, best, other, best],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )

    choice = SearchPolicy().choose(context)

    assert choice.phase == "depth"
    assert choice.parent.experiment_id == "exp_0001"
    assert choice.family == "ensemble"


def test_playbook_routes_pairwise_gauc_up_ndcg_down_to_objective(planner_context):
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
    contract = SimpleNamespace(**vars(planner_context.contract_summary))
    contract.allowed_families = ["objective", "temporal_history"]
    context = SimpleNamespace(
        contract_summary=contract,
        eligible_frontier=[root, pairwise],
        family_history=[pairwise],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )

    choice = SearchPolicy().choose(context)

    assert choice.phase == "playbook"
    assert choice.reason_code == "PAIRWISE_GAUC_UP_NDCG_DOWN"
    assert choice.parent.experiment_id == "exp_0001"
    assert choice.family == "objective"
    assert choice.method_card_id == "objective_listwise_user_softmax"


def test_playbook_cannot_reintroduce_contract_disallowed_family(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=True,
        metric_deltas={"gauc": 0.01, "ndcg@5": -0.004},
        method_card_ids=["objective_pairwise_bpr"],
    )
    context = context_with_latest(
        planner_context,
        latest,
        allowed_families=["temporal_history"],
    )

    choice = SearchPolicy().choose(context)

    assert choice.action == "blocked"
    assert choice.reason_code == "REQUIRED_FAMILY_UNAVAILABLE"


@pytest.mark.parametrize(
    ("overrides", "reason_code"),
    [
        ({"output_accepted": False}, "OUTPUT_CHECK_REJECTED"),
        (
            {"trust_verdict": "suspicious", "integrity": "compromised"},
            "SUSPICIOUS_RESULT_REQUIRES_QUARANTINE",
        ),
        ({"prediction_change": 0.0001}, "NO_OP_REQUIRES_RECOVERY"),
        ({"output_accepted": None}, "RESULT_NOT_BRANCHABLE"),
        ({"integrity": None}, "RESULT_NOT_BRANCHABLE"),
        ({"stability": None}, "RESULT_NOT_BRANCHABLE"),
        ({"population": None}, "RESULT_NOT_BRANCHABLE"),
        ({"prediction_change": None}, "RESULT_NOT_BRANCHABLE"),
        ({"stability": "unstable"}, "UNSTABLE_RESULT_REQUIRES_CONFIRMATION"),
        (
            {"fidelity": "proxy", "population": "internal_proxy"},
            "FIDELITY_PROMOTION_REQUIRED",
        ),
        ({"population": "unbiased_audit"}, "RESULT_NOT_BRANCHABLE"),
    ],
)
def test_playbook_blocks_unbranchable_results(
    planner_context, overrides, reason_code
):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=False,
        method_card_ids=["objective_pairwise_bpr"],
        **overrides,
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.action == "blocked"
    assert choice.reason_code == reason_code


def test_playbook_branches_after_terminal_proxy_prune(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.09,
        method_card_ids=["objective_pairwise_bpr"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.action == "propose"
    assert choice.reason_code == "EARLY_FIDELITY_REJECTED"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.family == "temporal_history"
    assert choice.method_card_id == "temporal_history_compact"


@pytest.mark.parametrize(
    ("metric_deltas", "reason_code"),
    [
        (
            {"gauc": -0.01, "ndcg@5": 0.01},
            "PAIRWISE_GAUC_DOWN_NDCG_UP",
        ),
        (
            {"gauc": 0.01, "ndcg@5": 0.01},
            "PAIRWISE_BOTH_METRICS_UP",
        ),
    ],
)
def test_playbook_handles_remaining_pairwise_metric_shapes(
    planner_context, metric_deltas, reason_code
):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=True,
        metric_deltas=metric_deltas,
        method_card_ids=["objective_pairwise_bpr"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.reason_code == reason_code
    assert choice.family == "objective"
    expected = (
        "objective_pairwise_bpr"
        if reason_code == "PAIRWISE_BOTH_METRICS_UP"
        else "objective_listwise_user_softmax"
    )
    assert choice.method_card_id == expected


def test_playbook_moves_meaningful_pairwise_no_gain_to_history(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=True,
        parent_delta=0.0,
        prediction_change=0.5,
        metric_deltas={"gauc": 0.0, "ndcg@5": 0.0},
        method_card_ids=["objective_pairwise_bpr"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.reason_code == "MEANINGFUL_CHANGE_NO_GAIN"
    assert choice.family == "temporal_history"
    assert choice.method_card_id == "temporal_history_compact"


def test_playbook_deepens_trusted_improvement(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="temporal_history",
        parent_eligible=True,
        parent_delta=0.01,
        method_card_ids=["temporal_history_compact"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.reason_code == "TRUSTED_FULL_IMPROVEMENT"
    assert choice.parent.experiment_id == "exp_0001"
    assert choice.method_card_id == "temporal_history_compact"


def test_playbook_abandons_trusted_regression(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=False,
        parent_delta=-0.01,
        method_card_ids=["objective_pairwise_bpr"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.reason_code == "TRUSTED_FULL_REGRESSION"
    assert choice.family == "temporal_history"


def test_optional_ranker_cannot_inject_an_illegal_choice(planner_context):
    illegal = PolicyChoice(
        action="propose",
        parent=None,
        family="forbidden",
        cost_tier="high",
        phase="external",
        reason_code="ILLEGAL",
        reason="Illegal external choice.",
        method_card_id="unknown",
    )
    policy = SearchPolicy(legal_choice_ranker=lambda choices, context: illegal)

    choice = policy.choose(planner_context)

    assert choice.family == "objective"
    assert choice.method_card_id == "objective_pairwise_bpr"


def test_policy_fails_closed_when_contract_has_no_allowed_families(planner_context):
    planner_context.contract_summary.allowed_families = []

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "blocked"
    assert choice.reason_code == "NO_LEGAL_FAMILY"


def test_policy_fails_closed_without_validated_playbook(planner_context):
    context = SimpleNamespace(
        **{
            key: value
            for key, value in vars(planner_context).items()
            if key != "playbook"
        }
    )

    choice = SearchPolicy().choose(context)

    assert choice.action == "blocked"
    assert choice.reason_code == "PLAYBOOK_MISSING"


def test_policy_fails_closed_when_playbook_safety_order_is_invalid(planner_context):
    rules = list(planner_context.playbook.rule_order)
    rules[0], rules[-1] = rules[-1], rules[0]
    planner_context.playbook = SimpleNamespace(
        schema_version="1.0",
        rule_order=rules,
        family_order=planner_context.playbook.family_order,
        method_order=planner_context.playbook.method_order,
    )

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "blocked"
    assert choice.reason_code == "PLAYBOOK_INVALID"


def test_policy_blocks_when_no_method_is_eligible(planner_context):
    planner_context.contract_summary.allowed_families = ["objective"]
    planner_context.contract_summary.allowed_data = ["public_validation"]

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "blocked"
    assert choice.reason_code == "NO_ELIGIBLE_METHOD"
