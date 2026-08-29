from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tacorank.schemas import (
    ArtifactRef,
    EvaluationResult,
    LessonCandidate,
    MonitorDirective,
    PatchCheckResult,
    RecoveryDecision,
    RecoveryPolicyContext,
    ResourceDelta,
    RunResult,
    TelemetrySample,
)


HASH = "a" * 64


def artifact(kind: str = "log") -> ArtifactRef:
    return ArtifactRef(
        artifact_id="art_1",
        kind=kind,
        path="artifacts/run/exp/file.jsonl",
        sha256=HASH,
        size_bytes=1,
        content_type="application/jsonl",
    )


def resources() -> ResourceDelta:
    return ResourceDelta(
        llm_input_tokens=0,
        llm_output_tokens=0,
        token_measurement="none",
        wall_time_ms=1,
        cpu_time_ms=1,
        gpu_time_ms=0,
        gpu_count=0,
        peak_rss_mb=10,
        peak_gpu_memory_mb=None,
        manual_interventions=0,
    )


def test_artifact_rejects_unsafe_path_bad_hash_and_extra_fields() -> None:
    base = artifact().model_dump()
    for update in (
        {"path": "../secret"},
        {"path": "C:/absolute/file"},
        {"path": "artifacts\\file"},
        {"sha256": "A" * 64},
        {"unexpected": True},
    ):
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate(base | update)


def test_resource_and_telemetry_numbers_are_bounded_and_finite() -> None:
    with pytest.raises(ValidationError):
        ResourceDelta.model_validate(resources().model_dump() | {"gpu_time_ms": -1})

    sample = dict(
        timestamp=datetime.now(timezone.utc),
        run_id="run_1",
        experiment_id="exp_1",
        attempt=1,
        elapsed_ms=1,
        process_alive=True,
        last_output_age_ms=0,
        cpu_percent=10.0,
        rss_mb=1,
        gpu_utilization_percent=None,
        gpu_memory_mb=None,
        loss=0.5,
        gradient_norm=1.0,
        disk_free_mb=100,
        recent_output_tail=None,
    )
    TelemetrySample(**sample)
    assert TelemetrySample(**(sample | {"loss": float("nan")})).loss != 0
    assert TelemetrySample(**(sample | {"gradient_norm": float("inf")})).gradient_norm == float("inf")
    with pytest.raises(ValidationError):
        TelemetrySample(**(sample | {"cpu_percent": float("nan")}))
    with pytest.raises(ValidationError):
        TelemetrySample(**(sample | {"timestamp": datetime.now()}))


def test_monitor_directive_fields_follow_action() -> None:
    MonitorDirective(action="continue", reason_code=None, summary=None)
    MonitorDirective(action="terminate", reason_code="HANG", summary="No progress detected.")
    with pytest.raises(ValidationError):
        MonitorDirective(action="terminate", reason_code="HANG", summary=None)


def test_patch_gate_receipt_and_violation_invariants() -> None:
    accepted = dict(
        run_id="run_1",
        experiment_id="exp_1",
        attempt=1,
        patch_commit_sha="deadbeef",
        diff_sha256=HASH,
        accepted=True,
        receipt_id="receipt_1",
        receipt_artifact=artifact("verification_receipt"),
        checks=[{"name": "protected_paths", "status": "pass", "details": None}],
        violations=[],
    )
    PatchCheckResult(**accepted)
    with pytest.raises(ValidationError):
        PatchCheckResult(**(accepted | {"receipt_id": None}))
    with pytest.raises(ValidationError):
        PatchCheckResult(
            **(
                accepted
                | {
                    "accepted": False,
                    "receipt_id": None,
                    "receipt_artifact": None,
                    "violations": [],
                }
            )
        )


def test_run_failure_requires_complete_fingerprint_evidence() -> None:
    failed = dict(
        run_id="run_1",
        experiment_id="exp_1",
        attempt=1,
        fidelity="proxy",
        patch_commit_sha="deadbeef",
        outcome="code_error",
        exit_code=1,
        error_class="NameError",
        error_fingerprint=HASH,
        error_summary="Candidate symbol was undefined.",
        log_artifact=artifact(),
        telemetry_artifact=None,
        checkpoint_artifact=None,
        prediction_artifact=None,
        resource_delta=resources(),
    )
    RunResult(**failed)
    with pytest.raises(ValidationError):
        RunResult(**(failed | {"error_fingerprint": None}))
    with pytest.raises(ValidationError):
        RunResult(**(failed | {"outcome": "success"}))


def test_no_op_evaluation_is_a_typed_recovery_input() -> None:
    result = EvaluationResult(
        run_id="run_1",
        experiment_id="exp_1",
        attempt=1,
        population="public_validation",
        fidelity="full",
        seed=0,
        public_query_index=1,
        evaluator_sha256=HASH,
        contract_sha256=HASH,
        metric_set={
            "metrics": {"ndcg": 0.5},
            "primary_metric_name": "primary",
            "primary_score": 0.5,
        },
        baseline_delta=0.0,
        parent_delta=0.0,
        previous_best_delta=0.0,
        prediction_change={"spearman_vs_parent": 1.0, "changed_row_fraction": 0.0},
        trust={
            "verdict": "no_op",
            "stability": "not_applicable",
            "integrity": "clean",
            "flags": [],
        },
    )
    assert result.trust.verdict.value == "no_op"
    with pytest.raises(ValidationError):
        type(result.metric_set).model_validate(
            result.metric_set.model_dump() | {"primary_score": float("inf")}
        )


def test_recovery_budget_and_operational_lesson_are_bounded() -> None:
    context = RecoveryPolicyContext(
        run_id="run_1",
        experiment_id="exp_1",
        original_experiment_spec={"hypothesis": "Keep the mechanism fixed."},
        current_patch_commit_sha="deadbeef",
        failure_event_id="evt_000027",
        attempt_history=[],
        prior_error_fingerprints=[HASH],
        repair_attempts_used=1,
        max_repair_attempts=2,
        same_commit_retries_used=0,
        remaining_run_budget={"execution_attempts": 1, "wall_time_ms": 30_000},
        allowed_runtime_adjustments={"batch_size": [64, 32]},
        contract_summary="Protected evaluator and fixed data boundary.",
    )
    assert context.max_repair_attempts == 2
    with pytest.raises(ValidationError):
        RecoveryPolicyContext.model_validate(
            context.model_dump() | {"repair_attempts_used": 2, "max_repair_attempts": 1}
        )

    lesson = LessonCandidate(
        origin="operational",
        category="implementation_constraint",
        tags=["wiring"],
        summary="The configured feature path is not wired into prediction.",
        applicability="Runs using this feature path.",
        avoid_when="The wiring smoke test passes.",
        confidence=0.8,
        source_event_ids=["evt_000027"],
        source_commit_shas=["deadbeef"],
    )
    RecoveryDecision(
        run_id="run_1",
        experiment_id="exp_1",
        failure_event_id="evt_000027",
        repair_attempt=2,
        action="abandon",
        reason_code="REPEATED_NO_OP",
        instructions="Stop recovery; preserve the evidence for planning.",
        same_error_count=2,
        remaining_repair_budget=0,
        lesson_candidate=lesson,
    )
    with pytest.raises(ValidationError):
        LessonCandidate.model_validate(lesson.model_dump() | {"category": "research_result"})
