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
