import asyncio
from types import SimpleNamespace

import pytest

from tacorank.recovery.classifier import classify_failure
from tacorank.recovery.classifier import TRANSIENT_CODING_ERROR_CODES
from tacorank.recovery.fingerprints import fingerprint_failure
from tacorank.recovery.policy import RecoveryManager
from tacorank.orchestrator.router import Harness
from tacorank.safety.path_policy import DELIBERATE_INTEGRITY_CODES, ViolationCode
from tacorank.schemas import RecoveryAction


def run_failure(outcome="code_error", summary="candidate failure", upstream_hash=None):
    return SimpleNamespace(
        outcome=outcome,
        error_class=outcome,
        error_summary=summary,
        error_fingerprint=upstream_hash,
    )


def test_failure_evidence_preserves_terminal_exception_after_bounding() -> None:
    result = SimpleNamespace(
        accepted=False,
        checks=[SimpleNamespace(name="smoke_test", status="fail")],
        violations=[
            SimpleNamespace(
                code="SMOKE_FAILURE",
                message=(
                    "isolated import traceback "
                    + "frame " * 300
                    + "ModuleNotFoundError: No module named 'numpy'"
                ),
            )
        ],
    )

    classification = classify_failure(result)

    assert len(classification.evidence) <= 800
    assert classification.evidence.startswith("smoke_test SMOKE_FAILURE")
    assert "ModuleNotFoundError: No module named 'numpy'" in classification.evidence


def context(*, remaining=2, previous=(), retries=0, adjustments=None):
    return SimpleNamespace(
        run_id="run-1",
        experiment_id="exp-1",
        original_experiment_spec=SimpleNamespace(
            hypothesis="Pairwise training improves within-user ordering",
            expected_mechanism="more informative preference gradients",
            target_files=["solution/train.py"],
        ),
        current_patch_commit_sha="abc123",
        failure_event_id="evt-failure-1",
        attempt_history=[],
        repair_attempts_used=2 - remaining,
        max_repair_attempts=2,
        same_commit_retries_used=retries,
        remaining_repair_budget=remaining,
        previous_error_fingerprints=list(previous),
        remaining_run_budget={"experiments": 1},
        allowed_runtime_adjustments=adjustments or {},
        contract_summary="Frozen contract; protected evaluator and data.",
    )


def decide(result, ctx=None, event_id="evt-failure-1"):
    return asyncio.run(
        RecoveryManager().decide(event_id, result, ctx or context())
    )


def test_candidate_code_failure_gets_focused_first_repair():
    decision = decide(run_failure("code_error", "NameError in solution/train.py"))

    assert decision.action == RecoveryAction.TRAE_REPAIR
    assert decision.repair_attempt == 1
    assert decision.remaining_repair_budget == 1
    assert "original hypothesis" in decision.instructions.lower()
    assert "Pairwise training" in decision.instructions
    assert "abc123" in decision.instructions
    assert "solution/train.py" in decision.instructions
    assert "before invoking any edit tool" in decision.instructions.lower()
    assert "REPAIR_PLAN:" in decision.instructions


def test_repair_budget_is_hard_capped_and_attempt_is_never_zero():
    decision = decide(run_failure(), context(remaining=0))

    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "REPAIR_BUDGET_EXHAUSTED"
    assert decision.repair_attempt == 2
    assert decision.remaining_repair_budget == 0


def test_supplied_hash_cannot_bypass_normalized_fingerprint_limit():
    first = run_failure(upstream_hash="a" * 64)
    second = run_failure(upstream_hash="b" * 64)
    fingerprint = classify_failure(first).fingerprint

    assert classify_failure(second).fingerprint == fingerprint
    decision = decide(second, context(previous=[fingerprint]))
    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "REPEATED_ERROR_FINGERPRINT"


def test_distinct_same_type_exception_messages_have_distinct_fingerprints():
    shape = fingerprint_failure("ValueError", "ValueError: shape mismatch")
    column = fingerprint_failure("ValueError", "ValueError: missing column")
    assert shape != column


def test_transient_initial_coding_failure_gets_one_bounded_retry():
    result = run_failure("code_error", "Trae could not start")
    result.failure_stage = "coding"
    result.error_class = "TRAE_LAUNCH_FAILED"
    ctx = context()
    ctx.failure_stage = "coding"
    first = decide(result, ctx)
    assert first.action == RecoveryAction.RETRY_SAME_COMMIT
    assert first.reason_code == "TRANSIENT_CODING_RETRY"

    second_ctx = context(
        previous=[classify_failure(result).fingerprint], retries=1
    )
    second_ctx.failure_stage = "coding"
    second = decide(result, second_ctx)
    assert second.action == RecoveryAction.ABANDON


def test_integrity_registry_covers_path_data_and_output_boundaries():
    expected = {
        ViolationCode.PROTECTED_PATH_MODIFIED.value,
        ViolationCode.PATH_TRAVERSAL.value,
        ViolationCode.SYMLINK_ESCAPE.value,
        ViolationCode.SUBMODULE_ESCAPE.value,
        ViolationCode.HIDDEN_LABEL_ACCESS.value,
        ViolationCode.FUTURE_INFORMATION_LEAKAGE.value,
        ViolationCode.UNAPPROVED_NETWORK.value,
        ViolationCode.SECRET_DETECTED.value,
        ViolationCode.OUTPUT_PROTECTED_DATA.value,
    }
    assert expected.issubset(DELIBERATE_INTEGRITY_CODES)
    assert "TRAE_LAUNCH_FAILED" in TRANSIENT_CODING_ERROR_CODES


def test_disk_quota_failure_abandons_without_same_commit_retry():
    result = run_failure("infrastructure_error", "No space left on device")
    result.error_class = "DISK_SPACE_EXHAUSTED"
    classification = classify_failure(result)
    assert classification.disk_quota_failure
    decision = decide(result)
    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "DISK_QUOTA_EXHAUSTED"


@pytest.mark.parametrize("dimension", ["wall_time_seconds", "token", "gpu_seconds"])
def test_exhausted_run_budget_abandons_before_any_work(dimension):
    ctx = context()
    ctx.remaining_run_budget = {dimension: 0}
    decision = decide(run_failure("infrastructure_error"), ctx)
    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "RUN_%s_BUDGET_EXHAUSTED" % dimension.upper()


def test_harness_recomputes_prior_fingerprints_and_excludes_current_failure():
    prior = run_failure(upstream_hash="a" * 64)
    current = run_failure(upstream_hash="b" * 64)
    prior.experiment_id = current.experiment_id = "exp-1"
    harness = Harness.__new__(Harness)
    harness.events = lambda: [
        SimpleNamespace(
            event_id="evt-prior",
            payload=SimpleNamespace(result=prior),
        ),
        SimpleNamespace(
            event_id="evt-current",
            payload=SimpleNamespace(result=current),
        ),
    ]

    fingerprints = harness._previous_failure_fingerprints(
        "exp-1", "evt-current"
    )

    assert fingerprints == [classify_failure(prior).fingerprint]
    assert fingerprints != ["a" * 64]


@pytest.mark.parametrize("outcome", ["infrastructure_error", "hang", "timeout"])
def test_transient_execution_failure_gets_one_exact_retry(outcome):
    result = run_failure(outcome, "execution stopped after progress")
    first = decide(result)
    fingerprint = classify_failure(result).fingerprint
    second = decide(result, context(previous=[fingerprint]))

    assert first.action == RecoveryAction.RETRY_SAME_COMMIT
    assert first.repair_attempt == 1
    assert first.remaining_repair_budget == 2
    assert second.action == RecoveryAction.ABANDON


def test_same_commit_retry_is_global_across_different_fingerprints():
    second_failure = run_failure("hang", "different worker failure")
    decision = decide(second_failure, context(retries=1))

    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "SAME_COMMIT_RETRY_EXHAUSTED"


def test_timeout_without_progress_does_not_receive_exact_retry():
    decision = decide(run_failure("timeout", "deadline exceeded"))
    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "TIMEOUT_WITHOUT_PROGRESS"


def test_approved_oom_adjustment_is_structured_and_reachable():
    decision = decide(
        run_failure("oom", "CUDA out of memory"),
        context(adjustments={"batch_size": {"next_value": 32}}),
    )
    assert decision.action == RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING
    assert decision.runtime_adjustments == {"batch_size": 32}


def test_timeout_profile_adjustment_is_reachable():
    decision = decide(
        run_failure("timeout", "deadline exceeded after checkpoint progress"),
        context(adjustments={"timeout_profile": {"next_value": "extended"}}),
    )
    assert decision.action == RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING
    assert decision.runtime_adjustments == {"timeout_profile": "extended"}


def test_identical_adjustment_is_not_repeated_after_a_different_failure():
    ctx = context(
        adjustments={"batch_size": {"next_value": 32}},
    )
    ctx.current_runtime_settings = {"batch_size": 32}
    ctx.attempt_history = [
        {
            "action": "adjust_approved_runtime_setting",
            "runtime_adjustments": {"batch_size": 32},
        }
    ]
    decision = decide(run_failure("oom", "different memory summary"), ctx)
    assert decision.action == RecoveryAction.ROLLBACK


def test_oom_without_contract_approved_adjustment_rolls_back():
    assert decide(run_failure("oom", "CUDA out of memory")).action == RecoveryAction.ROLLBACK


@pytest.mark.parametrize(
    "code,message",
    [
        ("UNAPPROVED_NETWORK", "unauthorized network access"),
        ("TARGET_LABEL_ACCESS", "candidate read target labels"),
    ],
)
def test_deliberate_integrity_codes_abandon_without_repair(code, message):
    result = SimpleNamespace(
        accepted=False,
        checks=[],
        violations=[SimpleNamespace(code=code, message=message)],
    )
    decision = decide(result)

    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "INTEGRITY_VIOLATION"


def test_first_hang_does_not_claim_repeated_hangs_when_budget_is_exhausted():
    decision = decide(run_failure("hang", "worker stopped"), context(remaining=0))

    assert decision.action == RecoveryAction.RETRY_SAME_COMMIT
    assert decision.lesson_candidate is None


def test_first_noop_does_not_claim_a_focused_repair_occurred():
    result = SimpleNamespace(
        trust=SimpleNamespace(verdict="no_op", flags=["predictions unchanged"])
    )
    decision = decide(result, context(remaining=0))

    assert decision.action == RecoveryAction.ABANDON
    assert decision.lesson_candidate is None


def test_repeated_hang_emits_evidence_linked_lesson():
    result = run_failure("hang", "worker stopped")
    fingerprint = classify_failure(result).fingerprint
    decision = decide(result, context(previous=[fingerprint]))

    assert decision.lesson_candidate is not None
    assert decision.lesson_candidate.source_event_ids == ["evt-failure-1"]
    assert "repeatedly" in decision.lesson_candidate.summary


def test_non_noop_evaluation_is_not_a_recovery_input():
    result = SimpleNamespace(trust=SimpleNamespace(verdict="valid", flags=[]))
    with pytest.raises(ValueError):
        decide(result)
