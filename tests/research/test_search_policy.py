from types import SimpleNamespace

import pytest

from tacorank.research.search_policy import PolicyChoice, SearchPolicy

from .conftest import make_summary


def test_policy_starts_score_guided_depth_first_from_baseline(planner_context):
    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.phase == "depth"
    assert choice.reason_code == "SCORE_GUIDED_DEPTH_FIRST"
    assert choice.family == "model"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.method_card_id == "model_lgbm_causal_history"


def test_clean_evaluator_baseline_does_not_imply_executable_parent_parity(
    planner_context,
):
    planner_context.contract_summary.research_capabilities = []

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "blocked"
    assert choice.reason_code == "NO_ELIGIBLE_METHOD"


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
    # Family diversity would reach for ensemble here, but the generic
    # depth route cannot name component_experiment_ids, and an ensemble
    # without components fails Person 1 validation with
    # ENSEMBLE_COMPONENT_REQUIRED. Only the soft-portfolio route can
    # propose one, so diversity falls through to the next allowed family.
    assert choice.family == "objective"
    assert choice.component_experiment_ids == ()


def test_policy_deepens_best_branch_before_untried_baseline_family(planner_context):
    root = make_summary("exp_0000", score=0.5946)
    best = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        score=0.61,
        parent_eligible=True,
        parent_delta=None,
    )
    contract = SimpleNamespace(**vars(planner_context.contract_summary))
    contract.allowed_families = ["model"]
    context = SimpleNamespace(
        contract_summary=contract,
        baseline=root,
        current_best=best,
        eligible_frontier=[root, best],
        family_history=[best],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )

    choice = SearchPolicy().choose(context)

    assert choice.reason_code == "SCORE_GUIDED_DEPTH_FIRST"
    assert choice.parent.experiment_id == "exp_0001"
    assert choice.family == "model"


def test_policy_prefers_deeper_branch_when_trusted_scores_tie(planner_context):
    root = make_summary("exp_0000", score=0.61)
    child = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        score=0.61,
        parent_eligible=True,
        parent_delta=None,
    )
    grandchild = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0001",
        commit_sha="c" * 40,
        family="objective",
        score=0.61,
        parent_eligible=True,
        parent_delta=None,
    )
    contract = SimpleNamespace(**vars(planner_context.contract_summary))
    contract.allowed_families = ["model"]
    context = SimpleNamespace(
        contract_summary=contract,
        baseline=root,
        current_best=grandchild,
        eligible_frontier=[root, child, grandchild],
        family_history=[child, grandchild],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )

    choice = SearchPolicy().choose(context)

    assert choice.parent.experiment_id == "exp_0002"


def test_policy_backtracks_only_after_best_branch_is_exhausted(planner_context):
    root = make_summary("exp_0000", score=0.5946)
    best = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        score=0.61,
        parent_eligible=True,
        parent_delta=None,
    )
    pairwise = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0001",
        family="objective",
        parent_eligible=False,
        method_card_ids=["objective_pairwise_bpr"],
    )
    listwise = make_summary(
        "exp_0003",
        parent_experiment_id="exp_0001",
        family="objective",
        parent_eligible=False,
        parent_delta=None,
        method_card_ids=["objective_listwise_user_softmax"],
    )
    # The branch is only exhausted once every objective card has been spent on
    # it. The family gained objective_lambdarank_ndcg, so leaving it unattempted
    # would test continuation rather than backtracking.
    lambdarank = make_summary(
        "exp_0004",
        parent_experiment_id="exp_0001",
        family="objective",
        parent_eligible=False,
        parent_delta=None,
        method_card_ids=["objective_lambdarank_ndcg"],
    )
    contract = SimpleNamespace(**vars(planner_context.contract_summary))
    contract.allowed_families = ["objective"]
    context = SimpleNamespace(
        contract_summary=contract,
        baseline=root,
        current_best=best,
        eligible_frontier=[root, best],
        family_history=[best, pairwise, listwise, lambdarank],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )

    choice = SearchPolicy().choose(context)

    assert choice.reason_code == "SCORE_GUIDED_DEPTH_FIRST"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.method_card_id == "objective_pairwise_bpr"


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


def test_playbook_continues_after_no_op_with_independent_choice(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=False,
        trust_verdict="no_op",
        stability="not_applicable",
        prediction_change=0.0,
        method_card_ids=["objective_pairwise_bpr"],
        status="no_op",
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.action == "propose"
    assert choice.reason_code == "NO_OP_INDEPENDENT_MECHANISM"
    assert choice.family == "model"
    assert choice.method_card_id == "model_lgbm_causal_history"


def test_no_op_tree_ranker_can_choose_one_same_mechanism_reimplementation(
    planner_context,
):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=False,
        trust_verdict="no_op",
        stability="not_applicable",
        prediction_change=0.0,
        method_card_ids=["objective_pairwise_bpr"],
        status="no_op",
    )
    seen = []

    def choose_reimplementation(choices, context):
        seen.extend(choices)
        return next(
            choice
            for choice in choices
            if choice.reason_code == "NO_OP_REIMPLEMENT_MECHANISM"
        )

    choice = SearchPolicy(
        legal_choice_ranker=choose_reimplementation
    ).choose(context_with_latest(planner_context, latest))

    assert {
        candidate.reason_code for candidate in seen
    } >= {
        "NO_OP_REIMPLEMENT_MECHANISM",
        "NO_OP_INDEPENDENT_MECHANISM",
    }
    assert choice.phase == "no_op_reimplementation"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.family == "objective"
    assert choice.method_card_id == "objective_pairwise_bpr"


def test_second_same_mechanism_no_op_retires_reimplementation(planner_context):
    first = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=False,
        trust_verdict="no_op",
        stability="not_applicable",
        prediction_change=0.0,
        method_card_ids=["objective_pairwise_bpr"],
        status="no_op",
    )
    latest = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0000",
        family="objective",
        parent_eligible=False,
        trust_verdict="no_op",
        stability="not_applicable",
        prediction_change=0.0,
        method_card_ids=["objective_pairwise_bpr"],
        status="no_op",
    )
    context = context_with_latest(planner_context, latest)
    context.family_history = [first, latest]
    seen = []

    def capture(choices, context):
        seen.extend(choices)
        return choices[0]

    choice = SearchPolicy(legal_choice_ranker=capture).choose(context)

    assert choice.action == "propose"
    assert choice.reason_code == "NO_OP_INDEPENDENT_MECHANISM"
    assert not any(
        candidate.reason_code == "NO_OP_REIMPLEMENT_MECHANISM"
        for candidate in seen
    )


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
    assert choice.family == "model"
    assert choice.method_card_id == "model_lgbm_causal_history"


def test_soft_pairwise_tradeoff_gets_one_bounded_listwise_child(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.001,
        metric_deltas={"GAUC": 0.006, "nDCG@5": -0.008},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
        method_card_ids=["objective_pairwise_bpr"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.action == "propose"
    assert choice.phase == "refinement"
    assert choice.reason_code == "SOFT_PRUNE_METRIC_TRADEOFF_REFINEMENT"
    assert choice.parent.experiment_id == "exp_0001"
    assert choice.method_card_id == "objective_listwise_user_softmax"


def test_soft_diverse_result_can_enter_bounded_ensemble_test(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="temporal_history",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.004,
        metric_deltas={"GAUC": -0.003, "nDCG@5": -0.005},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
        method_card_ids=["temporal_history_compact"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.action == "propose"
    assert choice.phase == "ensemble"
    assert choice.reason_code == "SOFT_PRUNE_DIVERSE_ENSEMBLE_TEST"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.method_card_id == "ensemble_diverse_residual_candidate"
    assert choice.component_experiment_ids == ("exp_0001",)


def test_soft_full_result_can_enter_bounded_ensemble_test(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="temporal_history",
        fidelity="full",
        population="public_validation",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="single_seed",
        parent_delta=-0.004,
        metric_deltas={"GAUC": -0.003, "nDCG@5": -0.005},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
        method_card_ids=["temporal_history_compact"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.action == "propose"
    assert choice.phase == "ensemble"
    assert choice.reason_code == "SOFT_PRUNE_DIVERSE_ENSEMBLE_TEST"
    assert choice.component_experiment_ids == ("exp_0001",)


def test_explicit_empty_soft_portfolios_are_authoritative(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="temporal_history",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.004,
        metric_deltas={"GAUC": -0.003, "nDCG@5": -0.005},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
        method_card_ids=["temporal_history_compact"],
    )
    context = context_with_latest(planner_context, latest)
    context.refinement_frontier_ids = []
    context.ensemble_candidate_ids = []

    choice = SearchPolicy().choose(context)

    assert choice.reason_code == "EARLY_FIDELITY_REJECTED"
    assert choice.phase == "playbook"
    assert choice.family != "ensemble"


def test_severe_proxy_regression_is_not_refined_or_ensembled(planner_context):
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
        metric_deltas={"GAUC": -0.08, "nDCG@5": -0.10},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.2,
        method_card_ids=["objective_pairwise_bpr"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.phase == "playbook"
    assert choice.reason_code == "EARLY_FIDELITY_REJECTED"
    assert choice.family == "model"
    assert choice.component_experiment_ids == ()


def test_failed_method_is_not_retried_from_same_parent(planner_context):
    pairwise = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.02,
        method_card_ids=["objective_pairwise_bpr"],
    )
    listwise = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0000",
        family="objective",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.02,
        method_card_ids=["objective_listwise_user_softmax"],
    )
    context = context_with_latest(
        planner_context,
        listwise,
        allowed_families=["objective"],
    )
    context.family_history = [pairwise, listwise]

    choice = SearchPolicy().choose(context)

    assert choice.action == "blocked"
    assert choice.reason_code == "NO_ELIGIBLE_METHOD"


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


def test_playbook_refines_meaningful_pairwise_no_gain_in_family(planner_context):
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

    assert choice.reason_code == "SCORE_GUIDED_SAME_FAMILY_REFINEMENT"
    assert choice.family == "objective"
    # A generic no-gain refinement follows method_order, which now offers the
    # metric-aligned objective before listwise. The specific
    # pairwise_gauc_up_ndcg_down rule still routes to listwise directly,
    # because it names the card rather than taking the ordered next one.
    assert choice.method_card_id == "objective_lambdarank_ndcg"


def test_directionally_positive_parent_prevents_premature_search_stop(
    planner_context,
):
    latest = make_summary(
        "exp_0006",
        parent_experiment_id="exp_0000",
        family="objective",
        score=0.6022787269104787,
        parent_eligible=True,
        parent_delta=0.0008099705575197,
        prediction_change=0.95,
        method_card_ids=["objective_listwise_user_softmax"],
    )

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.action == "propose"
    assert choice.reason_code == "SCORE_GUIDED_SAME_FAMILY_REFINEMENT"
    assert choice.parent.experiment_id == "exp_0006"
    assert choice.family == "objective"
    assert choice.method_card_id == "objective_pairwise_bpr"


def test_near_best_exploratory_parent_continues_depth_first(planner_context):
    root = make_summary("exp_0000", score=0.601468756352959)
    exploratory = make_summary(
        "exp_0003",
        parent_experiment_id="exp_0000",
        family="duration_bias",
        score=0.6014212941699442,
        decision="accept",
        parent_eligible=True,
        trust_verdict="inconclusive",
        stability="confirmed",
        parent_delta=-0.0000474621830148,
        prediction_change=0.8,
        method_card_ids=["duration_bias_censored_watch_time"],
    )
    context = SimpleNamespace(
        contract_summary=SimpleNamespace(**vars(planner_context.contract_summary)),
        baseline=root,
        current_best=root,
        eligible_frontier=[root, exploratory],
        family_history=[exploratory],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )

    choice = SearchPolicy().choose(context)

    assert choice.action == "propose"
    assert choice.reason_code == "SCORE_GUIDED_SAME_FAMILY_REFINEMENT"
    assert choice.parent.experiment_id == "exp_0003"
    assert choice.family == "duration_bias"
    # The parent already spent duration_bias_censored_watch_time, so the
    # same-family refinement picks the family's other card. Before the family
    # had a second card this route had nothing to offer and the search had to
    # leave the direction after a single attempt.
    assert choice.method_card_id == "duration_bias_quantile_deconfounded"


def test_meaningful_no_gain_backtracks_to_highest_scoring_experimental_path(
    planner_context,
):
    root = make_summary("baseline", score=0.601468756352959)
    stronger = make_summary(
        "exp_002",
        parent_experiment_id="baseline",
        family="temporal_history",
        score=0.6013885105993917,
        decision="accept",
        parent_eligible=True,
        trust_verdict="inconclusive",
        stability="confirmed",
        parent_delta=-0.00008024575356735397,
        prediction_change=0.033800606841780816,
        method_card_ids=["temporal_history_compact"],
        child_count=1,
    )
    latest_weaker = make_summary(
        "exp_003",
        parent_experiment_id="exp_002",
        family="duration_bias",
        score=0.6012304244722566,
        decision="accept",
        parent_eligible=True,
        trust_verdict="inconclusive",
        stability="confirmed",
        parent_delta=-0.00015808612713508197,
        prediction_change=0.9831317198920815,
        method_card_ids=["duration_bias_censored_watch_time"],
    )
    context = SimpleNamespace(
        contract_summary=SimpleNamespace(**vars(planner_context.contract_summary)),
        baseline=root,
        current_best=root,
        eligible_frontier=[root, stronger, latest_weaker],
        family_history=[stronger, latest_weaker],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )

    choice = SearchPolicy().choose(context)

    assert choice.action == "propose"
    assert choice.reason_code == "SCORE_GUIDED_SAME_FAMILY_REFINEMENT"
    assert choice.parent.experiment_id == "exp_002"
    assert choice.family == "temporal_history"
    # exp_002 already spent temporal_history_compact, so refining its own
    # family now takes the family's other card instead of leaving the branch
    # after a single attempt.
    assert choice.method_card_id == "temporal_history_target_attention"


def test_meaningful_no_gain_switches_family_after_best_path_refinement(
    planner_context,
):
    root = make_summary("baseline", score=0.601468756352959)
    strongest = make_summary(
        "exp_002",
        parent_experiment_id="baseline",
        family="temporal_history",
        score=0.6013885105993917,
        parent_eligible=True,
        trust_verdict="inconclusive",
        parent_delta=-0.00008024575356735397,
        prediction_change=0.03,
        method_card_ids=["temporal_history_compact"],
        child_count=1,
    )
    refinement = make_summary(
        "exp_003",
        parent_experiment_id="exp_002",
        family="temporal_history",
        score=0.60130,
        parent_eligible=False,
        decision="reject",
        trust_verdict="negative",
        parent_delta=-0.0000885105993917,
        prediction_change=0.02,
        method_card_ids=["temporal_history_compact"],
        status="rejected",
    )
    # The family only switches once every temporal_history card has been spent
    # on exp_002. Leaving the second one unattempted would test same-family
    # refinement instead.
    attention = make_summary(
        "exp_005",
        parent_experiment_id="exp_002",
        family="temporal_history",
        score=0.60129,
        parent_eligible=False,
        decision="reject",
        trust_verdict="negative",
        parent_delta=-0.0000985105993917,
        prediction_change=0.02,
        method_card_ids=["temporal_history_target_attention"],
        status="rejected",
    )
    latest = make_summary(
        "exp_004",
        parent_experiment_id="exp_002",
        family="duration_bias",
        score=0.60135,
        parent_eligible=True,
        trust_verdict="inconclusive",
        parent_delta=-0.0000385105993917,
        prediction_change=0.8,
        method_card_ids=["duration_bias_censored_watch_time"],
    )
    context = SimpleNamespace(
        contract_summary=SimpleNamespace(**vars(planner_context.contract_summary)),
        baseline=root,
        current_best=root,
        eligible_frontier=[root, strongest, latest],
        family_history=[strongest, refinement, attention, latest],
        method_cards=planner_context.method_cards,
        playbook=planner_context.playbook,
    )

    choice = SearchPolicy().choose(context)

    assert choice.action == "propose"
    assert choice.reason_code == "MEANINGFUL_CHANGE_NO_GAIN"
    assert choice.parent.experiment_id == "exp_002"
    assert choice.family != "temporal_history"


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
    assert choice.family == "model"


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

    assert choice.family == "model"
    assert choice.method_card_id == "model_lgbm_causal_history"


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
