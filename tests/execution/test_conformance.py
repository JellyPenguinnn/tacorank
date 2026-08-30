from __future__ import annotations

from tacorank.execution.conformance import (
    ExecutionConformanceError,
    verify_execution_receipt,
)
from tacorank.schemas import CostEstimate, ExperimentSpec, Fidelity, TrialType


def _spec(formulation: str = "bpr") -> ExperimentSpec:
    if formulation == "bpr":
        parameters = {
            "formulation": "bpr",
            "embedding_dim": 16,
            "learning_rate": 0.01,
            "epochs": 3,
            "negative_count": 4,
            "l2": 0.001,
            "residual_scale": 0.1,
            "max_train_rows": 100000,
        }
        implementation_id = "objective_bpr_v2"
    else:
        parameters = {
            "formulation": "listwise",
            "embedding_dim": 16,
            "learning_rate": 0.01,
            "epochs": 3,
            "l2": 0.001,
            "residual_scale": 0.1,
            "max_train_rows": 100000,
            "listwise_strategy": "full_observed",
        }
        implementation_id = "objective_listwise_full_v2"
    return ExperimentSpec(
        run_id="run_test",
        experiment_id="exp_001",
        parent_experiment_id="baseline",
        implementation_parent_experiment_id="baseline",
        parent_commit_sha="a" * 40,
        context_id="ctx_test",
        hypothesis="Test one verified ranking objective.",
        family="objective",
        change_summary="Configure one verified objective.",
        expected_mechanism="Improve within-user ranking.",
        success_criteria="Positive paired confidence interval.",
        falsification_condition="No trusted improvement.",
        estimated_cost=CostEstimate(
            llm_tokens_upper_bound=0,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=0,
            cost_tier="low",
        ),
        campaign_id="campaign_1",
        variant_id="objective_01",
        variant_instruction="Use the exact typed configuration.",
        variant_parameters=parameters,
        method_card_ids=["objective_pairwise_bpr"],
        duplicate_key="objective_01",
        target_stage="objective",
        target_files=["solution/experiment_config.py"],
        fidelity_plan=[Fidelity.SMOKE, Fidelity.PROXY, Fidelity.FULL],
        trial_type=TrialType.CONFIGURATION,
        implementation_id=implementation_id,
        implementation_sha256="b" * 64,
        active_parameter_names=sorted(parameters),
    )


def test_bpr_receipt_proves_every_declared_negative_was_used() -> None:
    spec = _spec()
    receipt = {
        "implementation_id": spec.implementation_id,
        "implementation_sha256": spec.implementation_sha256,
        "effective_parameters": dict(spec.variant_parameters),
        "training_semantics": {
            "informative_user_count": 5,
            "positive_count": 10,
            "negative_count": 4,
            "pair_count": 40,
            "negatives_per_positive": 4.0,
        },
    }

    evidence = verify_execution_receipt(spec, receipt)

    assert evidence["implementation_conformant"] == 1.0
    assert evidence["effective_negative_count"] == 4.0
    assert evidence["training_pair_count"] == 40.0


def test_bpr_receipt_rejects_an_ignored_negative_count() -> None:
    spec = _spec()
    receipt = {
        "implementation_id": spec.implementation_id,
        "implementation_sha256": spec.implementation_sha256,
        "effective_parameters": dict(spec.variant_parameters),
        "training_semantics": {
            "informative_user_count": 5,
            "positive_count": 10,
            "negative_count": 4,
            "pair_count": 10,
            "negatives_per_positive": 1.0,
        },
    }

    try:
        verify_execution_receipt(spec, receipt)
    except ExecutionConformanceError as error:
        assert "negative count" in str(error)
    else:
        raise AssertionError("ignored negative_count was accepted")


def test_configuration_receipt_rejects_changed_implementation() -> None:
    spec = _spec()
    receipt = {
        "implementation_id": spec.implementation_id,
        "implementation_sha256": "c" * 64,
        "effective_parameters": dict(spec.variant_parameters),
        "training_semantics": {},
    }

    try:
        verify_execution_receipt(spec, receipt)
    except ExecutionConformanceError as error:
        assert "hash-bound" in str(error)
    else:
        raise AssertionError("changed implementation was accepted")


def test_listwise_receipt_rejects_sampled_lists() -> None:
    spec = _spec("listwise")
    receipt = {
        "implementation_id": spec.implementation_id,
        "implementation_sha256": spec.implementation_sha256,
        "effective_parameters": dict(spec.variant_parameters),
        "training_semantics": {
            "listwise_strategy": "sampled_negatives",
            "informative_user_count": 5,
            "positive_count": 10,
            "list_count": 10,
            "list_row_count": 50,
            "normalized_positive_target_mass": 1.0,
        },
    }

    try:
        verify_execution_receipt(spec, receipt)
    except ExecutionConformanceError as error:
        assert "complete observed lists" in str(error)
    else:
        raise AssertionError("sampled listwise execution was accepted")
