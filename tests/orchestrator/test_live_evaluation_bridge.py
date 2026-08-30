from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tacorank.evaluation.types import (
    EvaluationResult as DomainEvaluationResult,
    MetricDelta,
    MetricSet,
    PredictionChange,
    TrustAssessment,
)
from tacorank.orchestrator.live import ProtectedEvaluationBridge
from tacorank.schemas import (
    EvaluationDecisionContext,
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
    request = SimpleNamespace(patch_commit_sha="d" * 40)
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
