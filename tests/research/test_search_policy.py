from types import SimpleNamespace

import pytest

from tacorank.research.search_policy import PolicyChoice, SearchPolicy

from .conftest import make_summary


def test_policy_starts_score_guided_depth_first_from_baseline(planner_context):
    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.phase == "depth"
    assert choice.reason_code == "SCORE_GUIDED_DEPTH_FIRST"
    assert choice.family == "objective"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.method_card_id == "objective_pairwise_bpr"


def test_campaign_exhausts_first_family_before_second(planner_context):
    planner_context.contract_summary.allowed_families = [
        "objective",
        "temporal_history",
    ]
    planner_context.research_campaign = SimpleNamespace(
        campaign_id="depth_test",
        family_order=["objective", "temporal_history"],
        family_budgets={"objective": 2, "temporal_history": 2},
        family_method_card_ids={
            "objective": ["objective_pairwise_bpr"],
            "temporal_history": ["temporal_history_compact"],
        },
        family_directives={
            "objective": "Adapt objective experiments from prior evidence.",
            "temporal_history": "Adapt history experiments from prior evidence.",
        },
    )

    first = SearchPolicy().choose(planner_context)
    assert first.phase == "campaign_depth"
    assert first.variant_id == "objective_01"

    attempted = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        decision="reject",
        parent_eligible=False,
        method_card_ids=["objective_pairwise_bpr"],
    )
    attempted.status = "rejected"
    attempted.campaign_id = "depth_test"
    attempted.variant_id = "objective_01"
    planner_context.family_history = [attempted]

    second = SearchPolicy().choose(planner_context)
    assert second.variant_id == "objective_02"
    assert second.family == "objective"
    assert second.parent.experiment_id == "exp_0000"
    assert second.implementation_parent.experiment_id == "exp_0001"

    attempted_two = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0000",
        family="objective",
        decision="reject",
        parent_eligible=False,
        method_card_ids=["objective_pairwise_bpr"],
    )
    attempted_two.status = "rejected"
    attempted_two.campaign_id = "depth_test"
    attempted_two.variant_id = "objective_02"
    planner_context.family_history.append(attempted_two)

    third = SearchPolicy().choose(planner_context)
    assert third.variant_id == "temporal_history_01"
    assert third.family == "temporal_history"


def test_campaign_exposes_all_currently_eligible_methods_to_agent(planner_context):
    planner_context.contract_summary.allowed_families = ["objective"]
    planner_context.research_campaign = SimpleNamespace(
        campaign_id="adaptive_depth",
        family_order=["objective"],
        family_budgets={"objective": 25},
        family_method_card_ids={
            "objective": [
                "objective_pairwise_bpr",
                "objective_listwise_user_softmax",
            ]
        },
        family_directives={"objective": "Adapt from prior evidence."},
    )
    prior = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        method_card_ids=["objective_pairwise_bpr"],
    )
    prior.campaign_id = "adaptive_depth"
    prior.variant_id = "objective_01"
    planner_context.family_history = [prior]

    choice = SearchPolicy().choose(planner_context)

    assert choice.variant_id == "objective_02"
    assert choice.method_card_id is None
    assert choice.allowed_method_card_ids == (
        "objective_pairwise_bpr",
        "objective_listwise_user_softmax",
    )


def test_campaign_continues_after_quarantining_suspicious_slot(planner_context):
    planner_context.contract_summary.allowed_families = ["objective"]
    planner_context.research_campaign = SimpleNamespace(
        campaign_id="depth_test",
        family_order=["objective"],
        family_budgets={"objective": 2},
        family_method_card_ids={"objective": ["objective_pairwise_bpr"]},
        family_directives={"objective": "Adapt from evidence."},
    )
    suspicious = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        decision="invalid",
        parent_eligible=False,
        trust_verdict="suspicious",
        integrity="compromised",
        method_card_ids=["objective_pairwise_bpr"],
    )
    suspicious.status = "invalid"
    suspicious.campaign_id = "depth_test"
    suspicious.variant_id = "objective_01"
    planner_context.family_history = [suspicious]

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.variant_id == "objective_02"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.implementation_parent.experiment_id == "exp_0000"


def test_campaign_refines_retained_implementation_from_trusted_score_parent(
    planner_context,
):
    planner_context.contract_summary.allowed_families = ["objective"]
    planner_context.research_campaign = SimpleNamespace(
        campaign_id="depth_test",
        family_order=["objective"],
        family_budgets={"objective": 2},
        family_method_card_ids={"objective": ["objective_pairwise_bpr"]},
        family_directives={"objective": "Adapt from evidence."},
    )
    retained = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        decision="retain",
        parent_eligible=False,
        trust_verdict="inconclusive",
        integrity="clean",
        method_card_ids=["objective_pairwise_bpr"],
    )
    retained.status = "retained"
    retained.campaign_id = "depth_test"
    retained.variant_id = "objective_01"
    planner_context.family_history = [retained]

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.variant_id == "objective_02"
    assert choice.parent.experiment_id == "exp_0000"
    assert choice.implementation_parent.experiment_id == "exp_0001"
    assert choice.implementation_parent.parent_commit_sha == "b" * 40


def test_campaign_enumerates_all_fifty_slots_before_exhaustion(planner_context):
    planner_context.contract_summary.allowed_families = [
        "objective",
        "temporal_history",
    ]
    planner_context.research_campaign = SimpleNamespace(
        campaign_id="objective_temporal_50",
        family_order=["objective", "temporal_history"],
        family_budgets={"objective": 25, "temporal_history": 25},
        family_method_card_ids={
            "objective": ["objective_pairwise_bpr"],
            "temporal_history": ["temporal_history_compact"],
        },
        family_directives={
            "objective": "Adapt objective parameters.",
            "temporal_history": "Adapt temporal parameters.",
        },
        minimum_family_full_evaluations=25,
        family_convergence_patience=25,
    )
    history = []
    for slot in range(1, 51):
        planner_context.family_history = history
        choice = SearchPolicy().choose(planner_context)
        expected_family = "objective" if slot <= 25 else "temporal_history"
        expected_sequence = slot if slot <= 25 else slot - 25
        assert choice.action == "propose"
        assert choice.family == expected_family
        assert choice.variant_id == "%s_%02d" % (
            expected_family,
            expected_sequence,
        )
        summary = make_summary(
            "exp_%03d" % slot,
            parent_experiment_id="exp_0000",
            commit_sha=("%040x" % slot),
            family=expected_family,
            decision="reject",
            parent_eligible=False,
            method_card_ids=[choice.method_card_id],
        )
        summary.status = "rejected"
        summary.campaign_id = "objective_temporal_50"
        summary.variant_id = choice.variant_id
        history.append(summary)

    planner_context.family_history = history
    exhausted = SearchPolicy().choose(planner_context)
    assert exhausted.action == "blocked"
    assert exhausted.reason_code == "CAMPAIGN_EXHAUSTED"


def test_adaptive_campaign_starts_with_objective_screening(
    planner_context,
):
    planner_context.contract_summary.allowed_families = [
        "features",
        "objective",
        "temporal_history",
    ]
    planner_context.contract_summary.allowed_data.extend(
        [
            "time_ms",
            "hourmin",
            "item_tags",
            "upload_date",
            "point_in_time_history_features",
        ]
    )
    planner_context.contract_summary.research_capabilities.extend(
        ["strict_temporal_cutoff", "history_affinity_features_legal"]
    )
    planner_context.research_campaign = SimpleNamespace(
        campaign_id="objective_temporal_features_50_v3",
        family_order=["objective", "temporal_history", "features"],
        family_budgets={"objective": 27, "temporal_history": 15, "features": 8},
        family_method_card_ids={
            "features": ["features_history_affinity"],
            "objective": [
                "objective_pairwise_bpr",
                "objective_listwise_user_softmax",
            ],
            "temporal_history": ["temporal_history_compact"],
        },
        family_directives={
            "features": "Adapt point-in-time history affinity.",
            "objective": "Adapt objective parameters.",
            "temporal_history": "Adapt temporal parameters.",
        },
        minimum_family_full_evaluations=5,
        family_convergence_patience=5,
    )

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.family == "objective"
    assert choice.allowed_method_card_ids == ("objective_pairwise_bpr",)
    assert choice.variant_id == "objective_01"


def test_deep_campaign_does_not_advance_after_five_non_improving_full_trials(
    planner_context,
):
    planner_context.contract_summary.allowed_families = [
        "objective",
        "temporal_history",
    ]
    planner_context.research_campaign = SimpleNamespace(
        campaign_id="objective_temporal_50_v4",
        family_order=["objective", "temporal_history"],
        family_budgets={"objective": 25, "temporal_history": 25},
        family_method_card_ids={
            "objective": ["objective_pairwise_bpr"],
            "temporal_history": ["temporal_history_compact"],
        },
        family_directives={
            "objective": "Adapt objective parameters.",
            "temporal_history": "Adapt temporal parameters.",
        },
        minimum_family_full_evaluations=25,
        family_convergence_patience=25,
    )
    history = []
    for slot in range(1, 6):
        summary = make_summary(
            "exp_%03d" % slot,
            family="objective",
            decision="reject",
            parent_eligible=False,
            best_eligible=False,
            method_card_ids=["objective_pairwise_bpr"],
        )
        summary.status = "rejected"
        summary.campaign_id = "objective_temporal_50_v4"
        summary.variant_id = "objective_%02d" % slot
        history.append(summary)
    planner_context.family_history = history

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.family == "objective"
    assert choice.variant_id == "objective_06"


def test_adaptive_campaign_improvement_resets_family_patience(planner_context):
    planner_context.contract_summary.allowed_families = [
        "objective",
        "temporal_history",
        "features",
    ]
    planner_context.research_campaign = SimpleNamespace(
        campaign_id="objective_temporal_features_50_v3",
        family_order=["objective", "temporal_history", "features"],
        family_budgets={"objective": 27, "temporal_history": 15, "features": 8},
        family_method_card_ids={
            "objective": ["objective_pairwise_bpr"],
            "temporal_history": ["temporal_history_compact"],
            "features": ["features_history_affinity"],
        },
        family_directives={
            "objective": "Adapt objective parameters.",
            "temporal_history": "Adapt temporal parameters.",
            "features": "Adapt point-in-time history affinity.",
        },
        minimum_family_full_evaluations=5,
        family_convergence_patience=5,
    )
    history = []
    for slot in range(1, 7):
        summary = make_summary(
            "exp_%03d" % slot,
            family="objective",
            decision="accept" if slot == 2 else "reject",
            parent_eligible=slot == 2,
            best_eligible=slot == 2,
            method_card_ids=["objective_pairwise_bpr"],
        )
        summary.status = "accepted" if slot == 2 else "rejected"
        summary.campaign_id = "objective_temporal_features_50_v3"
        summary.variant_id = "objective_%02d" % slot
        history.append(summary)
    planner_context.family_history = history

    choice = SearchPolicy().choose(planner_context)

    assert choice.action == "propose"
    assert choice.family == "objective"
    assert choice.variant_id == "objective_07"


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
    assert choice.family == "ensemble"


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
    loss_aligned_features = make_summary(
        "exp_0004",
        parent_experiment_id="exp_0001",
        family="objective",
        parent_eligible=False,
        parent_delta=None,
        method_card_ids=["objective_loss_aligned_features"],
    )
    contract = SimpleNamespace(**vars(planner_context.contract_summary))
    contract.allowed_families = ["objective"]
    context = SimpleNamespace(
        contract_summary=contract,
        baseline=root,
        current_best=best,
        eligible_frontier=[root, best],
        family_history=[best, pairwise, loss_aligned_features, listwise],
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


def test_suspicious_candidate_is_quarantined_without_blocking_search(planner_context):
    latest = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        decision="invalid",
        parent_eligible=False,
        trust_verdict="suspicious",
        integrity="compromised",
        method_card_ids=["objective_pairwise_bpr"],
    )
    latest.status = "invalid"

    choice = SearchPolicy().choose(context_with_latest(planner_context, latest))

    assert choice.action == "propose"
    assert choice.reason_code == "SUSPICIOUS_CANDIDATE_QUARANTINED"
    assert choice.parent.experiment_id == "exp_0000"


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
    assert choice.family == "temporal_history"
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
    assert choice.method_card_id == "objective_loss_aligned_features"


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
    assert choice.method_card_id == "duration_bias_censored_watch_time"


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
    assert choice.method_card_id == "temporal_history_compact"


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
        family_history=[strongest, refinement, latest],
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
