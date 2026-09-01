from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tacorank.schemas import (
    ExperimentSpec,
    RecoveryDecision,
    RecoveryPolicyContext,
    TelemetrySample,
)


HASH = "a" * 64


def experiment_spec():
    return ExperimentSpec.model_construct(
        run_id="run_1",
        experiment_id="exp_1",
        hypothesis="Keep the hypothesis fixed.",
        expected_mechanism="Exercise the intended feature path.",
        target_files=["solution/model.py"],
    )


def test_person4_consumes_canonical_recovery_context_names():
    context = RecoveryPolicyContext(
        run_id="run_1",
        experiment_id="exp_1",
        original_experiment_spec=experiment_spec(),
        current_patch_commit_sha="deadbeef",
        failure_event_id="evt_1",
        repair_attempts_used=0,
        max_repair_attempts=2,
        same_commit_retries_used=0,
        remaining_repair_budget=2,
        previous_error_fingerprints=[HASH],
        contract_summary="Frozen test contract.",
    )

    assert context.previous_error_fingerprints == [HASH]
    with pytest.raises(ValidationError):
        RecoveryPolicyContext.model_validate(
            context.model_dump() | {"remaining_repair_budget": 1}
        )
    with pytest.raises(ValidationError):
        RecoveryPolicyContext.model_validate(
            context.model_dump()
            | {"allowed_runtime_adjustments": {"shell": "unsafe"}}
        )
    with pytest.raises(ValidationError):
        RecoveryPolicyContext(
            run_id="run_1",
            experiment_id="exp_1",
            original_experiment_spec=experiment_spec(),
            current_patch_commit_sha="deadbeef",
            failure_event_id="evt_1",
            repair_attempts_used=0,
            max_repair_attempts=2,
            same_commit_retries_used=0,
            remaining_repair_budget=2,
            prior_error_fingerprints=[HASH],
            contract_summary="Frozen test contract.",
        )


def test_recovery_context_accepts_every_granted_same_commit_retry():
    # run_20260831T074812Z_35160 stopped fail-closed because the policy
    # granted a second same-commit retry while this schema still capped the
    # counter at 1. The schema bound must track the policy constant.
    from tacorank.recovery.policy import MAX_SAME_COMMIT_RETRIES

    context = RecoveryPolicyContext(
        run_id="run_1",
        experiment_id="exp_1",
        original_experiment_spec=experiment_spec(),
        current_patch_commit_sha="deadbeef",
        failure_event_id="evt_1",
        repair_attempts_used=0,
        max_repair_attempts=2,
        same_commit_retries_used=MAX_SAME_COMMIT_RETRIES,
        remaining_repair_budget=2,
        contract_summary="Frozen test contract.",
    )
    assert context.same_commit_retries_used == MAX_SAME_COMMIT_RETRIES
    with pytest.raises(ValidationError):
        RecoveryPolicyContext.model_validate(
            context.model_dump()
            | {"same_commit_retries_used": MAX_SAME_COMMIT_RETRIES + 1}
        )


def test_canonical_recovery_decision_never_allows_attempt_zero():
    values = dict(
        run_id="run_1",
        experiment_id="exp_1",
        failure_event_id="evt_1",
        action="abandon",
        reason_code="BUDGET_EXHAUSTED",
        instructions="Stop.",
        same_error_count=1,
        remaining_repair_budget=0,
    )

    RecoveryDecision(repair_attempt=1, **values)
    with pytest.raises(ValidationError):
        RecoveryDecision(repair_attempt=0, **values)
    with pytest.raises(ValidationError):
        RecoveryDecision(
            repair_attempt=1,
            action="adjust_approved_runtime_setting",
            runtime_adjustments={"batch_size": -1},
            **{key: value for key, value in values.items() if key != "action"},
        )


def test_person4_telemetry_uses_the_canonical_shape():
    sample = TelemetrySample(
        timestamp=datetime.now(timezone.utc),
        run_id="run_1",
        experiment_id="exp_1",
        attempt=1,
        elapsed_ms=1,
        process_alive=True,
        last_output_age_ms=0,
        cpu_percent=10.0,
        rss_mb=1,
        disk_free_mb=100,
        recent_output_tail="",
    )

    assert sample.recent_output_tail == ""
