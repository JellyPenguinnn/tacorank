from __future__ import annotations

import asyncio

import pytest

from tacorank.context.builder import ContextBuildError
from tacorank.context.redaction import redact


def test_planner_context_is_byte_deterministic_and_immutable(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    first = harness.context_builder.build_planner(harness.events())
    second = harness.context_builder.build_planner(harness.events())
    assert first.context_id == second.context_id
    assert first.content.encode() == second.content.encode()
    assert first.artifact.sha256 == second.artifact.sha256
    assert first.estimated_tokens <= harness.config.context_token_limit
    assert first.contract_summary.editable_paths == harness.config.editable_roots


def test_mandatory_context_cannot_be_silently_truncated(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    with pytest.raises(ContextBuildError, match="mandatory"):
        harness.context_builder.build_planner(harness.events(), max_tokens=1)


def test_secret_redaction():
    redacted, count = redact("api_key=sk-abcdefghijklmnopqrstu")
    assert "sk-" not in redacted
    assert "[REDACTED]" in redacted
    assert count >= 1


def test_coder_context_contains_the_real_worker_contract(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    context = next(
        event.payload.context
        for event in harness.events()
        if event.payload.type == "context.created"
        and event.payload.context.role == "coder"
    )
    assert context.experiment_spec.experiment_id == context.experiment_id
    assert context.parent_commit_sha == context.experiment_spec.parent_commit_sha
    assert context.contract_sha256 == harness.verified_contract.contract_sha256
    assert context.editable_roots == ["solution"]
    assert context.allowed_command_ids == harness.config.command_ids
    assert context.context_artifact == context.artifact
    assert context.step_limit == harness.config.coding_step_limit
    assert context.token_limit == harness.config.coding_token_limit


def test_execution_attempts_are_unique_across_fidelities(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    requests = [
        event.payload.request
        for event in harness.events()
        if event.payload.type == "execution.started"
    ]
    assert [request.fidelity.value for request in requests] == [
        "smoke",
        "proxy",
        "full",
    ]
    assert [request.attempt for request in requests] == [1, 2, 3]
