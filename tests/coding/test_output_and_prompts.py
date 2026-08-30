from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tacorank.coding.output_parser import TrajectoryParseError, parse_trajectory_bytes
from tacorank.coding.prompts import (
    PromptContractError,
    build_coding_prompt,
    build_repair_prompt,
    build_solution_revision_prompt,
)
from tacorank.coding.redaction import REDACTED, SecretRedactor


def _trajectory(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "task": "edit candidate",
        "start_time": "2026-01-01T00:00:00",
        "end_time": "2026-01-01T00:00:01",
        "provider": "fake-provider",
        "model": "fake-model",
        "max_steps": 4,
        "llm_interactions": [
            {
                "provider": "fake-provider",
                "model": "fake-model",
                "response": {
                    "content": "done",
                    "usage": {
                        "input_tokens": 8,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 2,
                        "cache_read_input_tokens": 1,
                        "reasoning_tokens": None,
                    },
                }
            },
            {
                "provider": "fake-provider",
                "model": "fake-model",
                "response": {
                    "content": "verified",
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            },
        ],
        "agent_steps": [
            {"step_number": 1, "state": "thinking"},
            {"step_number": 2, "state": "completed"},
        ],
        "success": True,
        "final_result": "patched",
        "execution_time": 1.0,
    }
    value.update(overrides)
    return value


def test_trajectory_parser_accounts_provider_usage_without_double_counting() -> None:
    parsed = parse_trajectory_bytes(json.dumps(_trajectory()).encode())
    assert parsed.steps_used == 2
    assert parsed.usage.input_tokens == 13
    assert parsed.usage.output_tokens == 5
    assert parsed.usage.cache_creation_input_tokens == 2
    assert parsed.usage.cache_read_input_tokens == 1
    assert parsed.usage.reasoning_tokens == 0


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update(llm_interactions=[]), "TOKEN_USAGE_MISSING"),
        (
            lambda value: value["llm_interactions"][0]["response"].pop("usage"),
            "TOKEN_USAGE_MISSING",
        ),
        (
            lambda value: value.update(
                agent_steps=[{"step_number": 2}, {"step_number": 1}]
            ),
            "TRAJECTORY_MALFORMED",
        ),
        (
            lambda value: value.update(agent_steps=[{"step_number": 5}]),
            "STEP_LIMIT_EXCEEDED",
        ),
    ],
)
def test_trajectory_parser_rejects_missing_or_inconsistent_evidence(
    mutation: object, code: str
) -> None:
    value = _trajectory()
    mutation(value)
    with pytest.raises(TrajectoryParseError) as failure:
        parse_trajectory_bytes(json.dumps(value).encode())
    assert failure.value.code == code


def test_redactor_removes_explicit_and_structured_credentials() -> None:
    secret = "super-secret-value"
    redactor = SecretRedactor([secret])
    source = (
        f"token={secret}\nAuthorization: Bearer abcdefghijklmnop\n"
        "https://user:password@example.invalid/path\n"
        "-----BEGIN PRIVATE KEY-----\nmaterial\n-----END PRIVATE KEY-----"
    )
    redacted = redactor.redact(source)
    assert secret not in redacted
    assert "abcdefghijklmnop" not in redacted
    assert "user:password@" not in redacted
    assert "material" not in redacted
    assert REDACTED in redacted
    assert secret not in repr(redactor)

    document = redactor.redact_json_bytes(
        json.dumps(
            {
                "api_key": secret,
                "nested": {"message": f"credential was {secret}"},
            }
        ).encode()
    )
    assert secret.encode() not in document
    assert json.loads(document)["api_key"] == REDACTED


def _coding_inputs(secret: str = "") -> tuple[SimpleNamespace, dict[str, object]]:
    spec: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": "run1",
        "experiment_id": "exp1",
        "parent_commit_sha": "b" * 40,
        "hypothesis": "Add a bounded candidate feature.",
        "target_files": ["solution/model.py"],
    }
    context = SimpleNamespace(
        context_id="ctx1",
        run_id="run1",
        experiment_id="exp1",
        contract_sha256="a" * 64,
        experiment_spec=spec,
        parent_commit_sha="b" * 40,
        target_interface_excerpts={"entrypoint": "predict(rows)"},
        editable_roots=("solution",),
        protected_paths=("contract", "runs"),
        allowed_command_ids=("candidate_smoke",),
        selected_method_cards=(),
        active_lessons=({"summary": f"Never print {secret}"},) if secret else (),
        coding_invariants=(
            "FM scores are unconstrained ranking scores; never clip them.",
        ),
        prior_result_summaries=(
            {
                "experiment_id": "exp0",
                "parent_delta": -0.04,
                "diagnostic_metrics": {"spearman_vs_fm_baseline": 0.67},
            },
        ),
        step_limit=4,
        token_limit=50,
        wall_time_limit_seconds=10,
        context_artifact={"path": "artifacts/context.json"},
    )
    return context, spec


def test_coding_prompt_is_exact_bounded_and_credential_free() -> None:
    secret = "credential-123456"
    context, spec = _coding_inputs(secret)
    prompt = build_coding_prompt(context, spec, redactor=SecretRedactor([secret]))
    assert secret not in prompt
    assert "candidate_smoke" in prompt
    assert '"hypothesis": "Add a bounded candidate feature."' in prompt
    assert "Do not choose or reinterpret the hypothesis" in prompt
    assert "max_provider_tokens: `50`" in prompt
    assert '"authoritative_target_files": [\n    "solution/model.py"\n  ]' in prompt
    assert "do not list the repository root" in prompt
    assert "controller-owned post-patch checks" in prompt
    assert "unconstrained real-valued ranking scores" in prompt
    assert '"parent_delta": -0.04' in prompt
    assert '"spearman_vs_fm_baseline": 0.67' in prompt
    assert "then call task_done" in prompt

    changed_spec = dict(spec, hypothesis="different")
    with pytest.raises(PromptContractError, match="differs"):
        build_coding_prompt(context, changed_spec)


def test_coding_prompt_records_an_unbounded_cumulative_token_limit() -> None:
    context, spec = _coding_inputs()
    context.token_limit = None

    prompt = build_coding_prompt(context, spec)

    assert "max_provider_tokens: null" in prompt
    assert "does not impose a cumulative trajectory token limit" in prompt


def test_coding_owner_retry_prompt_includes_exact_error_and_bounded_correction() -> None:
    context, spec = _coding_inputs()
    context.owner_retry_error_summary = "Trae returned NO_PATCH"
    context.owner_retry_instructions = (
        "Retry only the coding_worker stage against the same assignment."
    )

    prompt = build_coding_prompt(context, spec)

    assert "## Bounded owner retry" in prompt
    assert "Trae returned NO_PATCH" in prompt
    assert "Retry only the coding_worker stage" in prompt
    assert "proposed_correction as a bounded diagnostic" in prompt


def test_repair_prompt_preserves_original_hypothesis_and_exact_failure() -> None:
    _, spec = _coding_inputs()
    context = SimpleNamespace(
        context_id="repair-ctx",
        run_id="run1",
        experiment_id="exp1",
        repair_attempt=2,
        original_experiment_spec=spec,
        current_patch_commit_sha="c" * 40,
        accepted_patch_receipt_id="receipt-1",
        failure_class="code_error",
        error_fingerprint="fingerprint",
        error_summary="NameError in candidate",
        relevant_trace_tail="candidate.py:4",
        failed_checks=({"name": "smoke", "status": "fail"},),
        previous_repair_fingerprints=(),
        recovery_instructions="Import the missing symbol.",
        remaining_repair_budget=1,
        editable_roots=("solution",),
        protected_paths=("contract",),
    )
    prompt = build_repair_prompt(
        context,
        {"action": "trae_repair", "instructions": "fix import"},
        step_limit=3,
        token_limit=40,
        wall_time_limit_seconds=8,
        allowed_command_ids=("candidate_smoke",),
    )
    assert "Preserve the original hypothesis" in prompt
    assert "NameError in candidate" in prompt
    assert "authoritative observations" in prompt
    assert "proposed fix, not a proven diagnosis" in prompt
    assert "confirm that the proposed fix explains the supplied evidence" in prompt
    assert '"max_provider_tokens": 40' in prompt

    with pytest.raises(PromptContractError, match="not a code repair action"):
        build_repair_prompt(
            context,
            {"action": "retry_same_commit"},
            step_limit=3,
            token_limit=40,
            wall_time_limit_seconds=8,
            allowed_command_ids=("candidate_smoke",),
        )

    # restart_from_trusted_parent recodes from the parent commit instead of
    # editing the rejected candidate, but it is the same bounded repair task
    # and builds its prompt here. Rejecting it turned every clean restart into
    # CODING_WORKER_FAILURE and abandoned the experiment.
    restart_prompt = build_repair_prompt(
        context,
        {"action": "restart_from_trusted_parent"},
        step_limit=3,
        token_limit=40,
        wall_time_limit_seconds=8,
        allowed_command_ids=("candidate_smoke",),
    )
    assert "Preserve the original hypothesis" in restart_prompt
    assert "NameError in candidate" in restart_prompt


def test_repair_prompt_records_an_unbounded_cumulative_token_limit() -> None:
    _, spec = _coding_inputs()
    context = SimpleNamespace(
        context_id="repair-ctx",
        run_id="run1",
        experiment_id="exp1",
        repair_attempt=2,
        original_experiment_spec=spec,
        current_patch_commit_sha="c" * 40,
        accepted_patch_receipt_id="receipt-1",
        failure_class="code_error",
        error_fingerprint="fingerprint",
        error_summary="NameError in candidate",
        relevant_trace_tail="candidate.py:4",
        failed_checks=({"name": "smoke", "status": "fail"},),
        previous_repair_fingerprints=(),
        recovery_instructions="Import the missing symbol.",
        remaining_repair_budget=1,
        editable_roots=("solution",),
        protected_paths=("contract",),
    )

    prompt = build_repair_prompt(
        context,
        {"action": "trae_repair", "instructions": "fix import"},
        step_limit=3,
        token_limit=None,
        wall_time_limit_seconds=8,
        allowed_command_ids=("candidate_smoke",),
    )

    assert '"max_provider_tokens": null' in prompt
    assert "does not impose a cumulative trajectory token limit" in prompt


def test_solution_revision_prompt_is_grounded_and_bounded() -> None:
    context, spec = _coding_inputs()
    verification = {
        "accepted": False,
        "summary": "The approved residual is missing.",
        "findings": [
            {
                "code": "MECHANISM_MISSING",
                "severity": "error",
                "path": "solution/model.py",
                "message": "The entrypoint returns the parent score unchanged.",
            }
        ],
        "required_changes": ["Implement the approved bounded residual."],
    }

    prompt = build_solution_revision_prompt(
        context,
        spec,
        verification,
        review_attempt=1,
        max_review_attempts=5,
        step_limit=32,
        wall_time_limit_seconds=600,
    )

    assert '"max_review_attempts": 5' in prompt
    assert '"authoritative_target_files": [\n    "solution/model.py"\n  ]' in prompt
    assert "Implement the approved bounded residual." in prompt
    assert "Do not add smoke, test, helper, or alternate entrypoint files" in prompt
    assert "call task_done immediately" in prompt

    with pytest.raises(PromptContractError, match="requires a rejected verification"):
        build_solution_revision_prompt(
            context,
            spec,
            {**verification, "accepted": True},
            review_attempt=1,
            max_review_attempts=5,
            step_limit=32,
            wall_time_limit_seconds=600,
        )


def test_gate_a_repair_has_explicitly_absent_receipt() -> None:
    _, spec = _coding_inputs()
    context = SimpleNamespace(
        context_id="repair-gate-a",
        run_id="run1",
        experiment_id="exp1",
        repair_attempt=2,
        original_experiment_spec=spec,
        current_patch_commit_sha="c" * 40,
        accepted_patch_receipt_id=None,
        failure_class="contract_error",
        error_fingerprint="gate-a-fingerprint",
        error_summary="candidate modified a protected path",
        relevant_trace_tail="protected path: contract/frozen.json",
        failed_checks=({"name": "protected_path", "status": "fail"},),
        previous_repair_fingerprints=(),
        recovery_instructions="Remove only the protected-path change.",
        remaining_repair_budget=1,
        editable_roots=("solution",),
        protected_paths=("contract",),
    )
    prompt = build_repair_prompt(
        context,
        {"action": "trae_repair", "instructions": "remove protected change"},
        step_limit=3,
        token_limit=40,
        wall_time_limit_seconds=8,
        allowed_command_ids=("candidate_smoke",),
    )
    assert "- accepted_patch_receipt_id: null" in prompt
    assert "receipt-none" not in prompt


def test_post_acceptance_repair_requires_receipt() -> None:
    _, spec = _coding_inputs()
    context = SimpleNamespace(
        context_id="repair-execution",
        run_id="run1",
        experiment_id="exp1",
        repair_attempt=2,
        original_experiment_spec=spec,
        current_patch_commit_sha="c" * 40,
        accepted_patch_receipt_id=None,
        failure_class="code_error",
        error_fingerprint="execution-fingerprint",
        error_summary="candidate raised",
        relevant_trace_tail="candidate.py:4",
        failed_checks=(),
        previous_repair_fingerprints=(),
        recovery_instructions="Fix the candidate exception.",
        remaining_repair_budget=1,
        editable_roots=("solution",),
        protected_paths=("contract",),
    )
    with pytest.raises(PromptContractError, match="requires accepted_patch_receipt_id"):
        build_repair_prompt(
            context,
            {"action": "trae_repair", "instructions": "fix exception"},
            step_limit=3,
            token_limit=40,
            wall_time_limit_seconds=8,
            allowed_command_ids=("candidate_smoke",),
        )
