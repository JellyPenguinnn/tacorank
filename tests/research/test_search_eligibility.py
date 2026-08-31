from tacorank.research.search_eligibility import (
    PruneDisposition,
    classify_search_eligibility,
)

from .conftest import make_summary


def test_severe_regression_is_hard_pruned(planner_context):
    result = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.02,
        metric_deltas={"GAUC": -0.02, "nDCG@5": -0.02},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.4,
    )

    eligibility = classify_search_eligibility(result, planner_context)

    assert eligibility.disposition == PruneDisposition.HARD
    assert not eligibility.refinement_eligible
    assert not eligibility.ensemble_eligible
    assert "SEVERE_PRIMARY_REGRESSION" in eligibility.reasons


def test_clean_mild_diverse_regression_is_soft_ensemble_candidate(planner_context):
    result = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.001,
        metric_deltas={"GAUC": -0.0008, "nDCG@5": -0.0012},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
    )

    eligibility = classify_search_eligibility(result, planner_context)

    assert eligibility.disposition == PruneDisposition.SOFT
    assert not eligibility.branch_eligible
    assert not eligibility.best_checkpoint_eligible
    assert not eligibility.refinement_eligible
    assert eligibility.ensemble_eligible


def test_regression_beyond_one_epsilon_has_no_portfolio_followup(planner_context):
    result = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
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
    )

    eligibility = classify_search_eligibility(result, planner_context)

    assert eligibility.disposition == PruneDisposition.HARD
    assert not eligibility.refinement_eligible
    assert not eligibility.ensemble_eligible
    assert "SEVERE_PRIMARY_REGRESSION" in eligibility.reasons


def test_metric_tradeoff_allows_one_refinement(planner_context):
    result = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
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
        child_count=0,
    )

    eligibility = classify_search_eligibility(result, planner_context)

    assert eligibility.refinement_eligible
    result.child_count = 1
    assert not classify_search_eligibility(result, planner_context).refinement_eligible


def test_canonical_parent_and_checkpoint_flags_remain_authoritative(planner_context):
    result = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        parent_eligible=True,
        best_eligible=True,
    )

    eligibility = classify_search_eligibility(result, planner_context)

    assert eligibility.disposition == PruneDisposition.FRONTIER
    assert eligibility.branch_eligible
    assert eligibility.best_checkpoint_eligible
    assert not eligibility.refinement_eligible
    assert not eligibility.ensemble_eligible


def test_safety_evidence_overrides_inconsistent_positive_flags(planner_context):
    result = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        parent_eligible=True,
        best_eligible=True,
        trust_verdict="suspicious",
        integrity="compromised",
    )

    eligibility = classify_search_eligibility(result, planner_context)

    assert eligibility.disposition == PruneDisposition.HARD
    assert not eligibility.branch_eligible
    assert not eligibility.best_checkpoint_eligible
    assert "INTEGRITY_UNTRUSTED" in eligibility.reasons


def test_no_op_is_neutral_evidence_not_a_controller_prune(planner_context):
    result = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        parent_eligible=False,
        best_eligible=False,
        trust_verdict="no_op",
        stability="not_applicable",
        prediction_change=0.0,
        status="no_op",
    )

    eligibility = classify_search_eligibility(result, planner_context)

    assert eligibility.disposition == PruneDisposition.NULL
    assert not eligibility.branch_eligible
    assert not eligibility.best_checkpoint_eligible
