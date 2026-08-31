from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tacorank.schemas import (
    ArtifactKind,
    ArtifactRef,
    CostEstimate,
    CostTier,
    ExperimentSpec,
    Fidelity,
    MetricSet,
    PlannerAction,
    PlannerOutput,
    payload_artifacts,
)


def valid_spec(**updates):
    values = dict(
        run_id="run_1",
        experiment_id="exp_1",
        parent_experiment_id="baseline",
        parent_commit_sha="a" * 40,
        context_id="ctx_1",
        hypothesis="A bounded feature cross improves ranking.",
        family="feature_cross",
        change_summary="Add a feature cross.",
        target_stage="features",
        target_files=["solution/model.py"],
        fidelity_plan=[Fidelity.SMOKE, Fidelity.PROXY, Fidelity.FULL],
        expected_mechanism="Better interactions.",
        success_criteria="Primary score improves.",
        falsification_condition="No improvement.",
        estimated_cost=CostEstimate(
            llm_tokens_upper_bound=10,
            wall_time_seconds_upper_bound=10,
            gpu_seconds_upper_bound=0,
            cost_tier=CostTier.LOW,
        ),
        duplicate_key="feature_cross:v1",
    )
    values.update(updates)
    return ExperimentSpec(**values)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        valid_spec(unknown="not allowed")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_metrics_are_rejected(value):
    with pytest.raises(ValidationError):
        MetricSet(metrics={"primary": value}, primary_metric_name="primary", primary_score=value)


@pytest.mark.parametrize("path", ["/tmp/a", "../a", "a/../b", "a\\b", "./a"])
def test_artifact_paths_must_be_normalized_relative(path):
    with pytest.raises(ValidationError):
        ArtifactRef(
            artifact_id="a",
            kind=ArtifactKind.OTHER,
            path=path,
            sha256="0" * 64,
            size_bytes=0,
        )


def test_content_addressed_artifacts_allow_distinct_paths_for_same_bytes():
    digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    log = ArtifactRef(
        artifact_id="sha256-" + digest,
        kind=ArtifactKind.LOG,
        path="runs/run_1/artifacts/exp_1/attempt_001/execution.log",
        sha256=digest,
        size_bytes=0,
        content_type="text/plain; charset=utf-8",
    )
    telemetry = ArtifactRef(
        artifact_id="sha256-" + digest,
        kind=ArtifactKind.OTHER,
        path="runs/run_1/artifacts/exp_1/attempt_001/telemetry.jsonl",
        sha256=digest,
        size_bytes=0,
        content_type="application/x-ndjson",
    )

    assert payload_artifacts({"log": log, "telemetry": telemetry}) == [log, telemetry]


def test_content_addressed_artifacts_reject_same_id_for_different_content():
    first = ArtifactRef(
        artifact_id="shared",
        kind=ArtifactKind.LOG,
        path="artifacts/first.log",
        sha256="a" * 64,
        size_bytes=1,
    )
    second = ArtifactRef(
        artifact_id="shared",
        kind=ArtifactKind.LOG,
        path="artifacts/second.log",
        sha256="b" * 64,
        size_bytes=1,
    )

    with pytest.raises(ValueError, match="conflicting payload artifacts"):
        payload_artifacts({"first": first, "second": second})


def test_planner_discriminator_invariant():
    with pytest.raises(ValidationError):
        PlannerOutput(
            action=PlannerAction.PROPOSE,
            spec=None,
            reason_code="bad",
            reason="Missing proposal.",
        )
    with pytest.raises(ValidationError):
        PlannerOutput(
            action=PlannerAction.BLOCKED,
            spec=valid_spec(),
            reason_code="bad",
            reason="Blocked cannot carry a proposal.",
        )


def test_fidelity_plan_rejects_duplicate_or_reverse_stages():
    with pytest.raises(ValidationError):
        valid_spec(fidelity_plan=[Fidelity.SMOKE, Fidelity.PROXY, Fidelity.SMOKE])
    with pytest.raises(ValidationError):
        valid_spec(fidelity_plan=[Fidelity.SMOKE, Fidelity.SMOKE])


def test_ensemble_component_ids_are_typed_and_unique():
    spec = valid_spec(
        family="ensemble",
        component_experiment_ids=["exp_0001", "exp_0002"],
    )

    assert spec.component_experiment_ids == ["exp_0001", "exp_0002"]
    with pytest.raises(ValidationError):
        valid_spec(component_experiment_ids=["exp_0001", "exp_0001"])
    with pytest.raises(ValidationError):
        valid_spec(component_experiment_ids=["bad/component"])


def test_versioned_contract_fixtures():
    fixture_root = Path(__file__).parents[1] / "fixtures"
    valid = json.loads((fixture_root / "valid/experiment_spec.json").read_text())
    assert ExperimentSpec.model_validate(valid).schema_version == "1.0"
    invalid = json.loads(
        (fixture_root / "invalid/planner_output_missing_spec.json").read_text()
    )
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(invalid)
