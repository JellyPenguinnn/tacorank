from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from tacorank.context.builder import (
    ContextBuildError,
    _execution_conformant,
    _mentions_protected_validation_arm,
    _planner_primary_score,
    _planner_parent_metric_deltas,
    _planner_primary_score,
)
from tacorank.context.redaction import redact
from tacorank.git.refs import read_blob_at_commit
from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.schemas import (
    ArtifactKind,
    CostEstimate,
    LiteratureEvidence,
    ResearchProposal,
)


def test_planner_context_is_byte_deterministic_and_immutable(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    first = harness.context_builder.build_planner(harness.events())
    second = harness.context_builder.build_planner(harness.events())
    assert first.context_id == second.context_id
    assert first.content.encode() == second.content.encode()
    assert first.artifact.sha256 == second.artifact.sha256
    assert first.estimated_tokens <= harness.config.context_token_limit
    assert first.contract_summary.editable_paths == []
    assert first.contract_summary.protected_paths == []
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
    assert [plan.plan_id for plan in first.research_plans] == [
        "objective_alignment",
        "behavioral_history",
        "auxiliary_learning",
        "temporal_robustness",
        "model_and_ensemble",
    ]
    assert all(plan.status == "unstarted" for plan in first.research_plans)
    pairwise = next(
        card for card in first.method_cards if card.method_id == "objective_pairwise_bpr"
    )
    assert "within_user_positive_negative_pairs" in pairwise.prerequisites
    assert "train_interactions" in pairwise.allowed_data
    assert pairwise.implementation_targets == []
    assert first.target_interface_excerpts == {}
    assert "Authorized implementation interfaces" not in first.content
    assert "solution/candidate.py" not in first.content
    assert "target_files" not in first.content
    assert "target_stage" not in first.content
    assert "fidelity_plan" not in first.content
    assert "implementation_targets" not in first.content
    assert "commit_sha" not in first.content


def test_planner_uses_training_conformance_and_aggregate_seed_score():
    evaluation = SimpleNamespace(
        diagnostic_metrics={"training_implementation_conformant": 1.0},
        trust=SimpleNamespace(seed_mean=0.6123),
    )
    metric_set = SimpleNamespace(primary_score=0.6010)

    assert _execution_conformant(evaluation) is True
    assert _planner_primary_score(evaluation, metric_set) == 0.6123


def test_planner_history_preserves_complete_evaluation_evidence(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())

    context = harness.context_builder.build_planner(harness.events())
    latest = context.family_history[-1]
    proposed = next(
        event.payload.spec
        for event in harness.events()
        if event.payload.type == "experiment.proposed"
    )

    assert latest.output_accepted is True
    assert latest.output_checks["schema"].value == "pass"
    assert latest.output_violations == []
    assert latest.population.value == "public_validation"
    assert latest.highest_completed_fidelity.value == "full"
    assert latest.prediction_change == 0.1
    assert latest.prediction_spearman_vs_parent == 0.9
    assert latest.diagnostic_metrics["spearman_vs_fm_baseline"] == 0.8
    assert latest.diagnostic_metrics["user_rankable_fraction"] == 1.0
    assert latest.trust_flags == []
    assert "validation_arm_gap" not in latest.diagnostic_metrics
    assert "val_b_parent_delta" not in latest.diagnostic_metrics
    assert "val_a_parent_delta" not in latest.diagnostic_metrics
    assert latest.diagnostic_metrics["temporal_delta_slope"] == -0.003
    assert latest.metric_deltas == {
        "gauc": pytest.approx(0.02),
        "ndcg@5": pytest.approx(0.02),
        "primary": pytest.approx(0.02),
    }
    assert latest.diagnostic_worst_slice == "user_history.cold"
    assert "Cohort weakness" in latest.failure_hypotheses[0]
    assert "contract v1" in latest.diagnostic_limitations[0]
    assert latest.decision_reason_code == "fake_trusted_result"
    assert latest.parent_eligible is True
    assert latest.best_eligible is True
    assert context.refinement_frontier_ids == []
    assert context.ensemble_candidate_ids == []
    assert proposed.target_stage == proposed.family
    assert proposed.target_files == ["solution/experiment_config.py"]
    assert proposed.trial_type.value == "implementation"
    assert [fidelity.value for fidelity in proposed.fidelity_plan] == [
        "smoke",
        "proxy",
        "full",
    ]


@pytest.mark.parametrize(
    "value",
    ["val_b_parent_delta", "Val-B regression", "validation arm gap"],
)
def test_planner_redacts_all_protected_validation_arm_spellings(value):
    assert _mentions_protected_validation_arm(value)


def test_planner_metric_deltas_are_authoritative_and_route_safe() -> None:
    candidate = SimpleNamespace(metrics={"GAUC": 0.64, "nDCG@5": 0.52})
    full_parent = SimpleNamespace(metrics={"GAUC": 0.67, "nDCG@5": 0.54})
    protected = SimpleNamespace(
        parent_metric_deltas={"GAUC": -0.031, "nDCG@5": -0.021}
    )

    assert _planner_parent_metric_deltas(
        evaluation=protected,
        metric_set=candidate,
        parent_metrics=full_parent,
        same_route=False,
    ) == {"GAUC": -0.031, "nDCG@5": -0.021}

    legacy = SimpleNamespace(parent_metric_deltas={})
    assert _planner_parent_metric_deltas(
        evaluation=legacy,
        metric_set=candidate,
        parent_metrics=full_parent,
        same_route=False,
    ) == {}
    assert _planner_parent_metric_deltas(
        evaluation=legacy,
        metric_set=candidate,
        parent_metrics=full_parent,
        same_route=True,
    ) == pytest.approx({"GAUC": -0.03, "nDCG@5": -0.02})


def test_planner_parent_ranking_uses_confirmed_seed_mean() -> None:
    metric_set = SimpleNamespace(primary_score=0.6007)
    confirmed = SimpleNamespace(trust=SimpleNamespace(seed_mean=0.60063))
    unconfirmed = SimpleNamespace(trust=SimpleNamespace(seed_mean=None))

    assert _planner_primary_score(confirmed, metric_set) == 0.60063
    assert _planner_primary_score(unconfirmed, metric_set) == 0.6007
    assert _planner_primary_score(None, None) is None


def test_planner_context_separates_active_lessons_from_experiment_history(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())

    context = harness.context_builder.build_planner(
        harness.events(), max_tokens=3_000
    )

    assert len(context.family_history) == 1
    assert len(context.active_lessons) == 1
    lesson = context.active_lessons[0]
    assert lesson.lesson_id == "lesson_001"
    assert lesson.tags == ["feature_cross", "confirmed"]
    assert "source_commit_shas" not in lesson.model_dump()
    assert "Applicable active lesson" in context.content


def test_mandatory_context_cannot_be_silently_truncated(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    with pytest.raises(ContextBuildError, match="mandatory"):
        harness.context_builder.build_planner(harness.events(), max_tokens=1)


def test_implementation_binding_rejects_missing_configured_entrypoint(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    (harness.config.repository_root / "solution/experiment_config.py").unlink()
    context = harness.context_builder.build_planner(harness.events())
    values = {
        "run_id": context.run_id,
        "experiment_id": "exp_001",
        "parent_experiment_id": "baseline",
        "parent_commit_sha": harness.config.baseline_commit_sha,
        "context_id": context.context_id,
        "hypothesis": "Pairwise ranking may improve within-user ordering.",
        "family": "objective",
        "change_summary": "Test bounded pairwise preference learning.",
        "expected_mechanism": "Improve relative positive-negative ordering.",
        "success_criteria": "Trusted primary score improves beyond epsilon.",
        "falsification_condition": "No stable gain over the parent.",
        "estimated_cost": CostEstimate(
            llm_tokens_upper_bound=500,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=0,
            cost_tier="medium",
        ),
        "method_card_ids": ["objective_pairwise_bpr"],
        "evidence_event_ids": list(context.source_event_ids),
    }
    values["duplicate_key"] = compute_duplicate_key(values)
    proposal = ResearchProposal(**values)

    with pytest.raises(
        ContextBuildError, match="implementation target interface file is unavailable"
    ):
        harness.context_builder.bind_implementation(proposal)


def test_controller_binds_codeblind_proposal_to_coder_contract(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    context = harness.context_builder.build_planner(harness.events())
    literature = LiteratureEvidence(
        evidence_id="lit_paper_001",
        paper_id="W1234567890",
        title="Bayesian Personalized Ranking from Implicit Feedback",
        abstract="Pairwise ranking optimizes relative preference ordering.",
        year=2009,
        authors=["Steffen Rendle"],
        venue="UAI",
        citation_count=1000,
        influential_citation_count=100,
        url="https://openalex.org/W1234567890",
        query="Bayesian personalized ranking recommender systems",
    )
    values = {
        "run_id": context.run_id,
        "experiment_id": "exp_001",
        "parent_experiment_id": "baseline",
        "parent_commit_sha": harness.config.baseline_commit_sha,
        "context_id": context.context_id,
        "hypothesis": "Pairwise ranking may improve within-user ordering.",
        "family": "objective",
        "change_summary": "Test bounded pairwise preference learning.",
        "expected_mechanism": "Improve relative positive-negative ordering.",
        "success_criteria": "Trusted primary score improves beyond epsilon.",
        "falsification_condition": "No stable gain over the parent.",
        "estimated_cost": CostEstimate(
            llm_tokens_upper_bound=500,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=0,
            cost_tier="medium",
        ),
        "method_card_ids": ["objective_pairwise_bpr"],
        "evidence_event_ids": list(context.source_event_ids),
        "literature_evidence": [literature],
    }
    values["duplicate_key"] = compute_duplicate_key(values)

    spec = harness.context_builder.bind_implementation(ResearchProposal(**values))

    assert spec.target_stage == "objective"
    assert spec.target_files == [
        "solution/candidate.py",
        "solution/losses.py",
        "solution/official_fm.py",
        "solution/train.py",
    ]
    assert spec.trial_type.value == "implementation"
    assert [fidelity.value for fidelity in spec.fidelity_plan] == [
        "smoke",
        "proxy",
        "full",
    ]


def test_controller_hash_binds_verified_configuration_capability(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    context = harness.context_builder.build_planner(harness.events())
    parameters = {
        "formulation": "bpr",
        "embedding_dim": 16,
        "learning_rate": 0.01,
        "epochs": 3,
        "negative_count": 4,
        "l2": 0.001,
        "residual_scale": 0.1,
        "max_train_rows": 100000,
    }
    values = {
        "run_id": context.run_id,
        "experiment_id": "exp_001",
        "parent_experiment_id": "baseline",
        "parent_commit_sha": harness.config.baseline_commit_sha,
        "context_id": context.context_id,
        "hypothesis": "Pairwise ranking may improve within-user ordering.",
        "family": "objective",
        "change_summary": "Configure verified pairwise preference learning.",
        "expected_mechanism": "Improve relative positive-negative ordering.",
        "success_criteria": "Trusted primary score improves beyond uncertainty.",
        "falsification_condition": "No stable gain over the parent.",
        "estimated_cost": CostEstimate(
            llm_tokens_upper_bound=500,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=0,
            cost_tier="medium",
        ),
        "campaign_id": "campaign_1",
        "variant_id": "objective_01",
        "variant_instruction": "Use the complete typed BPR configuration.",
        "variant_parameters": parameters,
        "method_card_ids": ["objective_pairwise_bpr"],
        "evidence_event_ids": list(context.source_event_ids),
    }
    values["duplicate_key"] = compute_duplicate_key(values)

    implementation_path = "solution/research_scaffold.py"
    immutable_bytes = read_blob_at_commit(
        harness.config.repository_root,
        harness.config.baseline_commit_sha,
        implementation_path,
    )
    checkout_path = harness.config.repository_root / implementation_path
    checkout_path.write_text(
        checkout_path.read_text(encoding="utf-8")
        + "\n# Concurrent checkout edit must not affect run identity.\n",
        encoding="utf-8",
    )

    spec = harness.context_builder.bind_implementation(ResearchProposal(**values))

    assert spec.target_files == ["solution/experiment_config.py"]
    assert spec.trial_type.value == "configuration"
    assert spec.implementation_id == "objective_bpr_v2"
    assert spec.implementation_sha256 == hashlib.sha256(immutable_bytes).hexdigest()
    assert spec.implementation_sha256 != hashlib.sha256(
        checkout_path.read_bytes()
    ).hexdigest()
    assert set(spec.active_parameter_names) == set(parameters)


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
    assert context.estimated_tokens <= harness.config.context_token_limit
    assert context.target_interface_excerpts == {
        "solution/experiment_config.py": (
            harness.config.target_interface_excerpts["solution/experiment_config.py"]
        )
    }
    assert "solution/model.py" not in context.content
    assert "unconstrained real-valued ranking" in " ".join(
        context.coding_invariants
    )
    invariants = " ".join(context.coding_invariants)
    assert "selected Git parent's executable behavior" in invariants
    assert "not probabilities or non-baseline parent outputs" in invariants


def test_coder_context_makes_cited_prior_results_non_optional(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    events = harness.events()
    first_spec = next(
        event.payload.spec
        for event in events
        if event.payload.type == "experiment.proposed"
    )
    evidence_ids = [
        event.event_id
        for event in events
        if event.payload.type in {"evaluation.completed", "experiment.decided"}
        and (
            getattr(event.payload, "result", None) is not None
            or getattr(event.payload, "decision", None) is not None
        )
    ]
    second_spec = first_spec.model_copy(
        update={
            "experiment_id": "exp_002",
            "evidence_event_ids": evidence_ids,
            "method_card_ids": ["objective_pairwise_bpr"],
        }
    )

    context = harness.context_builder.build_coder(events, second_spec)

    assert context.prior_result_summaries
    prior = context.prior_result_summaries[0]
    assert prior.experiment_id == first_spec.experiment_id
    assert set(prior.source_event_ids).issubset(evidence_ids)
    assert set(prior.source_event_ids).issubset(context.source_event_ids)
    assert not set(prior.source_event_ids).intersection(context.excluded_source_ids)
    assert "Approved prior-result constraints" in context.content
    assert context.selected_method_cards


def test_coder_context_fails_instead_of_trimming_approved_guidance(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    events = harness.events()
    proposal = asyncio.run(
        harness.planner.propose(harness.context_builder.build_planner(events))
    ).spec
    assert proposal is not None
    values = proposal.model_dump()
    values["method_card_ids"] = ["objective_pairwise_bpr"]
    values["duplicate_key"] = compute_duplicate_key(values)
    spec = harness.context_builder.bind_implementation(
        ResearchProposal(**values)
    )

    with pytest.raises(ContextBuildError, match="mandatory"):
        harness.context_builder.build_coder(harness.events(), spec, max_tokens=1)


def test_coder_context_rejects_unavailable_method_guidance(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    planner_context = harness.context_builder.build_planner(harness.events())
    proposal = asyncio.run(harness.planner.propose(planner_context)).spec
    assert proposal is not None
    spec = harness.context_builder.bind_implementation(proposal).model_copy(
        update={"method_card_ids": ["method_that_does_not_exist"]}
    )

    with pytest.raises(ContextBuildError, match="method_that_does_not_exist"):
        harness.context_builder.build_coder(harness.events(), spec)


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
        "full",
        "full",
    ]
    assert [request.attempt for request in requests] == [1, 2, 3, 4, 5]
