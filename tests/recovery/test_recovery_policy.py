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
    assert first.reason_code == "TRANSIENT_CODING_WORKER_RETRY"
    assert "Give Trae this exact diagnostic" in first.instructions
    assert "DIAGNOSIS" in first.instructions

    second_ctx = context(
        previous=[classify_failure(result).fingerprint], retries=1
    )
    second_ctx.failure_stage = "coding"
    second = decide(result, second_ctx)
    assert second.action == RecoveryAction.ABANDON


def test_correctable_coding_protocol_failure_reissues_owner_with_diagnostic():
    result = run_failure("code_error", "must-patch task produced no Git diff")
    result.failure_stage = "coding"
    result.error_class = "NO_PATCH"
    ctx = context()
    ctx.failure_stage = "coding"

    decision = decide(result, ctx)

    assert decision.action == RecoveryAction.RETRY_SAME_COMMIT
    assert decision.reason_code == "TRANSIENT_CODING_WORKER_RETRY"
    assert "must-patch task produced no Git diff" in decision.instructions
    assert "Give Trae this exact diagnostic" in decision.instructions
    assert "REPAIR_PLAN" in decision.instructions
    assert decision.remaining_repair_budget == 2


def test_coding_configuration_failure_is_not_given_to_candidate_agent():
    result = run_failure("code_error", "pinned runtime identity does not match")
    result.failure_stage = "coding"
    result.error_class = "TRAE_RUNTIME_IDENTITY_MISMATCH"
    ctx = context()
    ctx.failure_stage = "coding"

    decision = decide(result, ctx)

    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "CODING_WORKER_FAILURE"
    assert decision.remaining_repair_budget == 2


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


def test_malformed_solution_verifier_is_terminal_after_internal_retries():
    result = run_failure(
        "code_error",
        "solution verifier finding keys differ from the required schema",
    )
    result.failure_stage = "coding"
    result.error_class = "SOLUTION_VERIFIER_MALFORMED"
    ctx = context()
    ctx.failure_stage = "coding"

    classification = classify_failure(result)
    decision = decide(result, ctx)

    assert classification.owner == "solution_verifier"
    assert not classification.owner_retryable
    assert not classification.trae_repairable
    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "SOLUTION_VERIFIER_RETRY_EXHAUSTED"
    assert "Preserve the candidate trajectory" in decision.instructions
    assert "do not rerun Trae" in decision.instructions
    assert decision.remaining_repair_budget == 2


def test_repeated_malformed_solution_verifier_does_not_consume_repair_budget():
    result = run_failure(
        "code_error",
        "solution verifier finding keys differ from the required schema",
    )
    result.failure_stage = "coding"
    result.error_class = "SOLUTION_VERIFIER_MALFORMED"
    fingerprint = classify_failure(result).fingerprint
    ctx = context(previous=[fingerprint], retries=1)
    ctx.failure_stage = "coding"

    decision = decide(result, ctx)

    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "SOLUTION_VERIFIER_RETRY_EXHAUSTED"
    assert decision.remaining_repair_budget == 2


def test_verifier_provider_exhaustion_does_not_rerun_coding_worker():
    result = run_failure(
        "code_error",
        "solution verifier provider request failed after bounded retries",
    )
    result.failure_stage = "coding"
    result.error_class = "TRAE_PROVIDER_UNAVAILABLE"
    ctx = context()
    ctx.failure_stage = "coding"

    classification = classify_failure(result)
    decision = decide(result, ctx)

    assert classification.owner == "solution_verifier"
    assert not classification.owner_retryable
    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "SOLUTION_VERIFIER_RETRY_EXHAUSTED"
    assert "do not rerun Trae" in decision.instructions
    assert decision.remaining_repair_budget == 2


def test_escaped_patch_gate_exception_is_not_sent_to_trae():
    result = run_failure("code_error", "receipt store returned inconsistent bytes")
    result.failure_stage = "patch_gate"
    result.error_class = "RuntimeError"
    ctx = context()
    ctx.failure_stage = "patch_gate"

    classification = classify_failure(result)
    decision = decide(result, ctx)

    assert classification.owner == "patch_gate"
    assert classification.control_plane_failure
    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "CONTROL_PLANE_INVARIANT_FAILURE"
    assert "do not ask Trae" in decision.instructions


def test_transient_evaluator_provider_failure_retries_evaluator_only():
    result = run_failure(
        "infrastructure_error", "evaluation provider is temporarily unavailable"
    )
    result.failure_stage = "evaluation"
    result.error_class = "EVALUATOR_PROVIDER_UNAVAILABLE"
    ctx = context()
    ctx.failure_stage = "evaluation"

    classification = classify_failure(result)
    decision = decide(result, ctx)

    assert classification.owner == "evaluator"
    assert classification.owner_retryable
    assert decision.action == RecoveryAction.RETRY_SAME_COMMIT
    assert decision.reason_code == "TRANSIENT_EVALUATOR_RETRY"
    assert "Retry only the evaluator stage" in decision.instructions
    assert decision.remaining_repair_budget == 2


def test_contract_identity_mismatch_is_control_plane_not_code_repair():
    result = SimpleNamespace(
        accepted=False,
        checks=[],
        violations=[
            SimpleNamespace(
                code="CONTRACT_HASH_MISMATCH",
                message="protected manifest differs from sealed identity",
            )
        ],
    )

    classification = classify_failure(result)
    decision = decide(result)

    assert classification.control_plane_failure
    assert not classification.trae_repairable
    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "CONTROL_PLANE_INVARIANT_FAILURE"


def test_typed_gate_a_interface_rejection_routes_to_grounded_trae_repair():
    result = SimpleNamespace(
        accepted=False,
        checks=[SimpleNamespace(name="interface", status="fail")],
        violations=[
            SimpleNamespace(
                code="INTERFACE_MISMATCH",
                message="candidate callable signature is incompatible",
            )
        ],
    )

    classification = classify_failure(result)
    decision = decide(result)

    assert classification.failure_class == "contract_error"
    assert classification.trae_repairable
    assert not classification.control_plane_failure
    assert decision.action == RecoveryAction.TRAE_REPAIR
    assert "candidate callable signature is incompatible" in decision.instructions


def test_typed_output_contract_rejection_routes_to_grounded_trae_repair():
    result = SimpleNamespace(
        accepted=False,
        checks={"header": "fail", "row_count": "pass"},
        violations=[],
    )

    classification = classify_failure(result)
    decision = decide(result)

    assert classification.failure_class == "output_contract"
    assert classification.trae_repairable
    assert not classification.control_plane_failure
    assert decision.action == RecoveryAction.TRAE_REPAIR
    assert "header" in decision.instructions


def test_candidate_interface_failure_remains_a_grounded_trae_repair():
    result = run_failure(
        "interface_error", "candidate emitted the wrong prediction columns"
    )

    classification = classify_failure(result)
    decision = decide(result)

    assert classification.owner == "execution_runner"
    assert classification.trae_repairable
    assert decision.action == RecoveryAction.TRAE_REPAIR
    assert "wrong prediction columns" in decision.instructions
    assert "DIAGNOSIS:" in decision.instructions


@pytest.mark.parametrize(
    "outcome,summary",
    [
        ("code_error", "NameError in candidate implementation"),
        ("interface_error", "candidate output columns do not match"),
        ("contract_error", "candidate command contract does not match"),
        ("numerical_error", "candidate scores contain NaN"),
    ],
)
def test_typed_candidate_defects_are_the_only_general_trae_repairs(
    outcome, summary
):
    decision = decide(run_failure(outcome, summary))

    assert decision.action == RecoveryAction.TRAE_REPAIR
    assert summary in decision.instructions
    assert decision.remaining_repair_budget == 1


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
def test_candidate_integrity_codes_restart_once_from_trusted_parent(code, message):
    result = SimpleNamespace(
        accepted=False,
        checks=[],
        violations=[SimpleNamespace(code=code, message=message)],
    )
    decision = decide(result)

    assert decision.action == RecoveryAction.RESTART_FROM_TRUSTED_PARENT
    assert decision.reason_code == "CANDIDATE_INTEGRITY_CLEAN_RESTART"
    assert decision.remaining_repair_budget == 1
    assert "Do not edit, bypass, or weaken" in decision.instructions


def test_repeated_candidate_integrity_violation_abandons_only_experiment():
    result = SimpleNamespace(
        accepted=False,
        checks=[],
        violations=[
            SimpleNamespace(
                code="PROTECTED_PATH_MODIFIED",
                message="candidate patch touches a protected path",
            )
        ],
    )
    fingerprint = classify_failure(result).fingerprint
    decision = decide(result, context(previous=[fingerprint], remaining=1))

    assert decision.action == RecoveryAction.ABANDON
    assert decision.reason_code == "CANDIDATE_INTEGRITY_RETRY_EXHAUSTED"


def test_secret_detection_remains_terminal_integrity_violation():
    result = SimpleNamespace(
        accepted=False,
        checks=[],
        violations=[
            SimpleNamespace(
                code="SECRET_DETECTED",
                message="credential material was detected",
            )
        ],
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

    assert decision.action == RecoveryAction.RETURN_TO_PLANNER
    assert decision.reason_code == "NO_OP_RECOVERY_EXHAUSTED"
    assert decision.lesson_candidate is None


def test_first_noop_gets_one_scoped_trae_wiring_repair():
    result = SimpleNamespace(
        trust=SimpleNamespace(
            verdict="no_op", flags=["NO_PREDICTION_CHANGE"]
        )
    )

    decision = decide(result)

    assert decision.action == RecoveryAction.TRAE_REPAIR
    assert decision.reason_code == "REPAIRABLE_NO_OP_WIRING"
    assert "solution/train.py" in decision.instructions
    assert "starting with the current diff" in decision.instructions
    assert "Do not survey setup files" in decision.instructions
    assert "task_done immediately" in decision.instructions
    assert "Frozen contract; protected evaluator and data" not in decision.instructions


def test_repeated_noop_returns_evidence_to_planner_without_second_repair():
    result = SimpleNamespace(
        trust=SimpleNamespace(
            verdict="no_op", flags=["NO_PREDICTION_CHANGE"]
        )
    )
    fingerprint = classify_failure(result).fingerprint

    decision = decide(result, context(remaining=1, previous=[fingerprint]))

    assert decision.action == RecoveryAction.RETURN_TO_PLANNER
    assert decision.reason_code == "NO_OP_RECOVERY_EXHAUSTED"
    assert decision.remaining_repair_budget == 1


def test_exhausted_noop_repair_worker_returns_evidence_to_planner():
    ctx = context(remaining=1)
    ctx.failure_stage = "coding"
    ctx.attempt_history = [
        {
            "action": "trae_repair",
            "reason_code": "REPAIRABLE_NO_OP_WIRING",
        }
    ]

    decision = decide(
        run_failure(
            "code_error",
            "TRAE_STEP_LIMIT_EXCEEDED after bounded no-op repair",
        ),
        ctx,
    )

    assert decision.action == RecoveryAction.RETURN_TO_PLANNER
    assert decision.reason_code == "NO_OP_REPAIR_WORKER_EXHAUSTED"
    assert decision.remaining_repair_budget == 1


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
