from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from tacorank.artifacts import ArtifactStore
from tacorank.evaluation.types import (
    EvaluationResult as DomainEvaluationResult,
    MetricDelta,
    MetricSet,
    PredictionChange,
    TrustAssessment,
)
from tacorank.orchestrator.live import ProtectedEvaluationBridge
from tacorank.schemas import (
    ArtifactKind,
    EvaluationDecisionContext,
    EvaluationDiagnostics,
    Fidelity,
    Integrity,
    LessonCategory,
    Population,
    Stability,
    TrustVerdict,
)


class StaticEventStore:
    def __init__(self, events):
        self.events = events

    def read_events(self, repair_tail=True):
        del repair_tail
        return list(self.events)


def event(event_id, payload_type, *, causation_event_id=None, **payload_values):
    return SimpleNamespace(
        event_id=event_id,
        causation_event_id=causation_event_id,
        payload=SimpleNamespace(type=payload_type, **payload_values),
    )


def test_protected_decision_attaches_eligible_research_lesson():
    domain = DomainEvaluationResult(
        run_id="run_test",
        experiment_id="exp_001",
        attempt=5,
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        seed=33,
        public_query_index=3,
        evaluator_sha256="a" * 64,
        contract_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        metric_set=MetricSet({"primary": 0.62}, "primary", 0.62),
        baseline_delta=MetricDelta(0.02, {"primary": 0.02}),
        parent_delta=MetricDelta(0.02, {"primary": 0.02}),
        previous_best_delta=MetricDelta(0.02, {"primary": 0.02}),
        prediction_change=PredictionChange(0.8, 0.2, 0.0, 0.9),
        trust=TrustAssessment(
            TrustVerdict.ACCEPTED,
            Stability.CONFIRMED,
            Integrity.CLEAN,
            eta_applied=0.0016,
            seed_mean=0.62,
            seed_stderr=0.0,
            seed_count=3,
        ),
        seed_evidence_event_ids=("evt_seed_1", "evt_seed_2"),
    )
    canonical = domain.to_canonical()
    spec = SimpleNamespace(
        experiment_id="exp_001",
        parent_experiment_id="baseline",
        family="features",
        hypothesis="A bounded feature cross improves within-user ordering.",
        expected_mechanism="The cross exposes a missing interaction.",
        target_stage="feature_engineering",
        falsification_condition="The confirmed full score does not improve.",
    )
    request = SimpleNamespace(patch_commit_sha="d" * 40, seed=11)
    events = [
        event("evt_proposal", "experiment.proposed", spec=spec),
        event("evt_started", "execution.started", request=request),
        event(
            "evt_finished",
            "execution.finished",
            causation_event_id="evt_started",
        ),
        event(
            "evt_output",
            "output.checked",
            causation_event_id="evt_finished",
        ),
        event(
            "evt_evaluation",
            "evaluation.completed",
            causation_event_id="evt_output",
            result=canonical,
        ),
    ]
    bridge = ProtectedEvaluationBridge(
        config=SimpleNamespace(max_confirmation_attempts=2),
        event_store=StaticEventStore(events),
        populations={},
        baseline_predictions={},
        evaluator_adapter=object(),
    )
    bridge._domain_results[("exp_001", 5, "full")] = domain

    decision = asyncio.run(
        bridge.decide(
            canonical,
            EvaluationDecisionContext(
                run_id="run_test",
                experiment_id="exp_001",
                baseline_score=0.6,
                parent_score=0.6,
                previous_best_score=0.6,
            ),
        )
    )

    lesson = decision.lesson_candidate
    assert lesson is not None
    assert lesson.category == LessonCategory.RESEARCH_RESULT
    assert lesson.source_event_ids == [
        "evt_seed_1",
        "evt_seed_2",
        "evt_evaluation",
    ]
    assert lesson.source_commit_shas == ["d" * 40]
    assert lesson.measured_under_frame_experiment_id == "baseline"


def test_full_diagnostics_resolve_proxy_delta_from_the_same_commit():
    proxy_domain = DomainEvaluationResult(
        run_id="run_test",
        experiment_id="exp_001",
        attempt=2,
        population=Population.INTERNAL_PROXY,
        fidelity=Fidelity.PROXY,
        seed=11,
        public_query_index=None,
        evaluator_sha256="a" * 64,
        contract_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        metric_set=MetricSet({"primary": 0.59}, "primary", 0.59),
        baseline_delta=MetricDelta(-0.01, {"primary": -0.01}),
        parent_delta=MetricDelta(-0.02, {"primary": -0.02}),
        previous_best_delta=MetricDelta(-0.01, {"primary": -0.01}),
        prediction_change=PredictionChange(0.8, 0.2, 0.0, 0.9),
        trust=TrustAssessment(
            TrustVerdict.NEGATIVE,
            Stability.NOT_APPLICABLE,
            Integrity.CLEAN,
        ),
    )
    request = SimpleNamespace(patch_commit_sha="d" * 40, seed=11)
    events = [
        event("evt_started", "execution.started", request=request),
        event(
            "evt_finished",
            "execution.finished",
            causation_event_id="evt_started",
        ),
        event(
            "evt_output",
            "output.checked",
            causation_event_id="evt_finished",
        ),
        event(
            "evt_proxy",
            "evaluation.completed",
            causation_event_id="evt_output",
            result=proxy_domain.to_canonical(),
        ),
    ]
    bridge = ProtectedEvaluationBridge(
        config=SimpleNamespace(max_confirmation_attempts=2),
        event_store=StaticEventStore(events),
        populations={},
        baseline_predictions={},
        evaluator_adapter=object(),
    )
    full_request = SimpleNamespace(
        experiment_id="exp_001",
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        seed=11,
    )

    assert bridge._internal_proxy_delta(full_request, "d" * 40) == -0.02
    assert bridge._internal_proxy_delta(full_request, "e" * 40) is None


def test_diagnostics_artifact_contains_aggregates_without_row_evidence(tmp_path):
    diagnostics = {
        "proxy_parent_delta": 0.02,
        "proxy_full_delta_gap": 0.03,
        "validation_arm_deltas": {"val_a": 0.01, "val_b": -0.01},
        "validation_arm_gap": 0.02,
        "temporal_delta_slope": -0.004,
        "gain_concentration_top10pct": 0.75,
        "slice_deltas": {"user_history.cold": -0.02},
        "best_slice": "user_history.cold",
        "worst_slice": "user_history.cold",
        "failure_hypotheses": ["A measured cohort regressed."],
        "limitations": ["A controlled ablation is required."],
    }
    domain = DomainEvaluationResult(
        run_id="run_test",
        experiment_id="exp_001",
        attempt=3,
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        seed=33,
        public_query_index=2,
        evaluator_sha256="a" * 64,
        contract_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        metric_set=MetricSet({"primary": 0.59}, "primary", 0.59),
        baseline_delta=MetricDelta(-0.01, {"primary": -0.01}),
        parent_delta=MetricDelta(-0.02, {"primary": -0.02}),
        previous_best_delta=MetricDelta(-0.01, {"primary": -0.01}),
        prediction_change=PredictionChange(0.8, 0.2, 0.0, 0.9),
        trust=TrustAssessment(
            TrustVerdict.NEGATIVE,
            Stability.NOT_APPLICABLE,
            Integrity.CLEAN,
        ),
        diagnostic_metrics={"validation_arm_gap": 0.02},
        diagnostics=EvaluationDiagnostics(**diagnostics),
    )
    store = ArtifactStore(tmp_path, ("runs",))
    bridge = ProtectedEvaluationBridge(
        config=SimpleNamespace(max_confirmation_attempts=2),
        event_store=StaticEventStore([]),
        populations={},
        baseline_predictions={},
        evaluator_adapter=object(),
        artifact_store=store,
    )

    artifact = bridge._write_diagnostics_artifact(domain)

    store.verify(artifact)
    payload = json.loads((tmp_path / artifact.path).read_text(encoding="utf-8"))
    assert payload["diagnostics"] == domain.diagnostics.model_dump(mode="json")
    assert payload["diagnostic_metrics"] == {"validation_arm_gap": 0.02}
    serialized = json.dumps(payload, sort_keys=True)
    assert "user_ids" not in serialized
    assert "labels" not in serialized
    assert "candidate_scores" not in serialized
    assert "parent_scores" not in serialized


def test_training_diagnostics_resolve_from_canonical_artifact_store(tmp_path):
    store = ArtifactStore(tmp_path, ("runs",))
    artifact = store.write(
        artifact_id="training_diagnostics",
        kind=ArtifactKind.CHECKPOINT,
        relative_path="runs/run_test/artifacts/training-diagnostics.json",
        content=(
            json.dumps(
                {
                    "train_rows": 1200,
                    "loss_start": 0.72,
                    "loss_end": 0.61,
                    "pairwise_accuracy": 0.64,
                    "ignored": "not numeric evidence",
                }
            ).encode("utf-8")
            + b"\n"
        ),
        content_type="application/json",
    )
    events = [
        event(
            "evt_finished",
            "execution.finished",
            result=SimpleNamespace(checkpoint_artifact=artifact),
        ),
        event(
            "evt_output",
            "output.checked",
            causation_event_id="evt_finished",
        ),
    ]
    bridge = ProtectedEvaluationBridge(
        config=SimpleNamespace(max_confirmation_attempts=2),
        event_store=StaticEventStore(events),
        populations={},
        baseline_predictions={},
        evaluator_adapter=object(),
        artifact_store=store,
    )

    assert bridge._training_diagnostics("evt_output") == {
        "train_rows": 1200.0,
        "loss_start": 0.72,
        "loss_end": 0.61,
        "pairwise_accuracy": 0.64,
    }
