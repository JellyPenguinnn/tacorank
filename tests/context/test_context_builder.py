from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tacorank.context.builder import ContextBuildError
from tacorank.context.redaction import redact
from tacorank.schemas import ArtifactKind


def test_planner_context_is_byte_deterministic_and_immutable(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    first = harness.context_builder.build_planner(harness.events())
    second = harness.context_builder.build_planner(harness.events())
    assert first.context_id == second.context_id
    assert first.content.encode() == second.content.encode()
    assert first.artifact.sha256 == second.artifact.sha256
    assert first.estimated_tokens <= harness.config.context_token_limit
    assert first.contract_summary.editable_paths == harness.config.editable_roots
    assert (
        first.contract_summary.allowed_families
        == harness.config.allowed_research_families
    )
    assert first.contract_summary.allowed_data == harness.config.allowed_research_data
    assert first.contract_summary.data_manifest_sha256 == harness.config.data_manifest_sha256
    assert first.contract_summary.evaluator_sha256 == harness.config.evaluator_sha256
    assert first.playbook.rule_order[0] == "output_rejected"
    assert first.playbook.method_order["objective"][0] == "objective_pairwise_bpr"
    assert first.refinement_frontier_ids == []
    assert first.ensemble_candidate_ids == []
    pairwise = next(
        card for card in first.method_cards if card.method_id == "objective_pairwise_bpr"
    )
    assert "within_user_positive_negative_pairs" in pairwise.prerequisites
    assert "train_interactions" in pairwise.allowed_data
    assert pairwise.implementation_targets == ["solution/candidate.py"]
    assert first.target_interface_excerpts == {
        "solution/candidate.py": harness.config.target_interface_excerpts[
            "solution/candidate.py"
        ]
    }


def test_planner_history_preserves_complete_evaluation_evidence(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())

    context = harness.context_builder.build_planner(harness.events())
    latest = context.family_history[-1]

    assert latest.output_accepted is True
    assert latest.output_checks["schema"].value == "pass"
    assert latest.output_violations == []
    assert latest.population.value == "public_validation"
    assert latest.highest_completed_fidelity.value == "full"
    assert latest.prediction_change == 0.1
    assert latest.prediction_spearman_vs_parent == 0.9
    assert latest.diagnostic_metrics == {
        "spearman_vs_fm_baseline": 0.8,
        "user_rankable_fraction": 1.0,
    }
    assert latest.trust_flags == []
    assert latest.parent_eligible is True
    assert latest.best_eligible is True
    assert context.refinement_frontier_ids == []
    assert context.ensemble_candidate_ids == []


def test_mandatory_context_cannot_be_silently_truncated(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    with pytest.raises(ContextBuildError, match="mandatory"):
        harness.context_builder.build_planner(harness.events(), max_tokens=1)


def test_planner_context_rejects_missing_configured_entrypoint(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    (harness.config.repository_root / "solution/candidate.py").unlink()

    with pytest.raises(ContextBuildError, match="target interface file is unavailable"):
        harness.context_builder.build_planner(harness.events())


def test_secret_redaction():
    redacted, count = redact("api_key=sk-abcdefghijklmnopqrstu")
    assert "sk-" not in redacted
    assert "[REDACTED]" in redacted
    assert count >= 1


def test_empty_verified_failure_log_uses_typed_error_summary(harness) -> None:
    artifact = harness.context_builder.artifact_store.write(
        artifact_id="empty-log",
        kind=ArtifactKind.LOG,
        relative_path="runs/run_test/empty-execution.log",
        content=b"",
        content_type="text/plain; charset=utf-8",
    )

    trace = harness.context_builder._trace_tail(
        SimpleNamespace(log_artifact=artifact),
        "missing required output roles: prediction",
    )

    assert trace == "missing required output roles: prediction"


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
