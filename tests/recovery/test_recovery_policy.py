from types import SimpleNamespace

import pytest

from tacorank.recovery.classifier import classify_failure
from tacorank.recovery.fingerprints import fingerprint_failure
from tacorank.recovery.policy import RecoveryManager
from tacorank.schemas import RecoveryAction, RecoveryPolicyContext


def context(**changes):
    values = dict(
        run_id="run-1",
        experiment_id="exp-1",
        original_experiment_spec={
            "hypothesis": "Pairwise training improves within-user ordering",
            "expected_mechanism": "more informative preference gradients",
        },
        current_patch_commit_sha="abc123",
        failure_event_id="evt-failure-1",
        attempt_history=[],
        prior_error_fingerprints=[],
        repair_attempts_used=0,
        max_repair_attempts=2,
        same_commit_retries_used=0,
        remaining_run_budget={"runs": 3},
        allowed_runtime_adjustments={},
        contract_summary="candidate training files only",
    )
    values.update(changes)
    return RecoveryPolicyContext(**values)


def run_failure(outcome, summary="failure evidence", fingerprint=None):
    return SimpleNamespace(
        outcome=outcome,
        error_class="RuntimeError",
        error_summary=summary,
        error_fingerprint=fingerprint,
    )


def test_fingerprint_ignores_timestamp_address_temp_path_and_line_number():
    first = fingerprint_failure(
        "code_error",
        '2026-08-29T10:00:00Z File "C:\\temp\\a\\solution\\train.py", line 19, in fit\nValueError: bad 0xabc',
    )
    second = fingerprint_failure(
        "code_error",
        '2026-08-29T11:10:12Z File "C:\\temp\\b\\solution\\train.py", line 91, in fit\nValueError: bad 0xdef',
    )
    assert first == second
    assert len(first) == 64


def test_candidate_code_failure_gets_focused_first_repair():
    decision = RecoveryManager().decide(
        run_failure("code_error", "NameError in solution/train.py"), context()
    )
    assert decision.action is RecoveryAction.TRAE_REPAIR
    assert decision.repair_attempt == 1
    assert decision.remaining_repair_budget == 1
    assert "original hypothesis remains" in decision.instructions
    assert "First explain the fault briefly" in decision.instructions
    assert "do not edit protected" in decision.instructions


def test_same_fingerprint_twice_abandons_before_another_repair():
    result = run_failure("interface_error", "shape mismatch")
    fingerprint = classify_failure(result).fingerprint
    decision = RecoveryManager().decide(
        result,
        context(prior_error_fingerprints=[fingerprint], repair_attempts_used=1),
    )
    assert decision.action is RecoveryAction.ABANDON
    assert decision.same_error_count == 2
    assert decision.repair_attempt == 1


def test_repair_budget_is_hard_capped_at_two():
    decision = RecoveryManager().decide(
        run_failure("code_error", "new syntax error"),
        context(repair_attempts_used=2),
    )
    assert decision.action is RecoveryAction.ABANDON
    assert decision.repair_attempt == 2
    assert decision.remaining_repair_budget == 0


def test_infrastructure_and_hang_get_only_one_exact_retry():
    first = RecoveryManager().decide(run_failure("infrastructure_error"), context())
    assert first.action is RecoveryAction.RETRY_SAME_COMMIT
    second = RecoveryManager().decide(
        run_failure("hang", "heartbeat stale"),
        context(same_commit_retries_used=1),
    )
    assert second.action is RecoveryAction.ABANDON
    assert second.lesson_candidate.category.value == "process_rule"


def test_oom_uses_only_named_allowlisted_adjustment():
    decision = RecoveryManager().decide(
        run_failure("oom", "CUDA out of memory"),
        context(allowed_runtime_adjustments={"batch_size": 32, "shell": "rm -rf /"}),
    )
    assert decision.action is RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING
    assert "batch_size=32" in decision.instructions
    assert "shell" not in decision.instructions


def test_oom_without_legal_adjustment_rolls_back():
    decision = RecoveryManager().decide(run_failure("oom"), context())
    assert decision.action is RecoveryAction.ROLLBACK


def test_repeated_oom_emits_evidence_linked_resource_lesson():
    result = run_failure("oom", "CUDA out of memory")
    decision = RecoveryManager().decide(
        result,
        context(prior_error_fingerprints=[classify_failure(result).fingerprint]),
    )
    lesson = decision.lesson_candidate
    assert decision.action is RecoveryAction.ABANDON
    assert lesson.category.value == "resource_constraint"
    assert lesson.source_event_ids == ["evt-failure-1"]
    assert lesson.source_commit_shas == ["abc123"]


def test_timeout_does_not_expand_unapproved_budget():
    decision = RecoveryManager().decide(
        run_failure("timeout", "steady checkpoint progress"), context()
    )
    assert decision.action is RecoveryAction.ABANDON
    assert decision.reason_code == "TIMEOUT_TOO_COSTLY"


def test_explicitly_exhausted_structured_run_budget_abandons():
    decision = RecoveryManager().decide(
        run_failure("code_error"),
        context(remaining_run_budget={"proxy_runs": 0, "full_runs": 0}),
    )
    assert decision.action is RecoveryAction.ABANDON
    assert decision.reason_code == "RUN_BUDGET_EXHAUSTED"


def test_gate_b_alignment_failure_routes_to_focused_repair():
    output = SimpleNamespace(
        accepted=False,
        checks={"row_alignment": "fail", "finite_scores": "pass"},
        violations=[SimpleNamespace(code="ROW_ALIGNMENT", message="row order differs")],
    )
    decision = RecoveryManager().decide(output, context())
    assert decision.action is RecoveryAction.TRAE_REPAIR
    assert "Gate B output-contract checks" in decision.instructions


def test_non_noop_evaluation_is_not_a_recovery_input():
    accepted = SimpleNamespace(trust=SimpleNamespace(verdict="negative", flags=[]))
    with pytest.raises(ValueError, match="only an evaluation verdict of no_op"):
        RecoveryManager().decide(accepted, context())


def test_first_noop_repairs_wiring_and_repeated_noop_reflects_sparsely():
    result = SimpleNamespace(trust=SimpleNamespace(verdict="no_op", flags=["unchanged_scores"]))
    first = RecoveryManager().decide(result, context())
    assert first.action is RecoveryAction.TRAE_REPAIR
    assert "wiring smoke test" in first.instructions
    assert first.lesson_candidate is None

    second = RecoveryManager().decide(
        result,
        context(
            prior_error_fingerprints=[classify_failure(result).fingerprint],
            repair_attempts_used=1,
        ),
    )
    assert second.action is RecoveryAction.ABANDON
    assert second.lesson_candidate.category.value == "implementation_constraint"
    assert "hypothesis as falsified" in second.lesson_candidate.avoid_when


def test_hidden_access_abandons_and_emits_integrity_lesson():
    gate_result = SimpleNamespace(
        accepted=False,
        checks=[],
        violations=[SimpleNamespace(code="HIDDEN_ACCESS", message="attempted hidden label access")],
    )
    decision = RecoveryManager().decide("evt-legacy", gate_result, context())
    assert decision.failure_event_id == "evt-legacy"
    assert decision.action is RecoveryAction.ABANDON
    assert decision.lesson_candidate.category.value == "integrity_warning"
    assert decision.lesson_candidate.confidence == pytest.approx(0.99)


def test_one_off_syntax_failure_produces_no_operational_lesson():
    decision = RecoveryManager().decide(
        run_failure("code_error", "SyntaxError: invalid syntax"), context()
    )
    assert decision.action is RecoveryAction.TRAE_REPAIR
    assert decision.lesson_candidate is None
