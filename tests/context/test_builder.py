from __future__ import annotations

import pytest

from tacorank.context.builder import ContextBuildError
from tacorank.schemas import (
    EvaluationCompletedPayload,
    Event,
    EventType,
    MetricSet,
    Population,
)


def test_mandatory_context_cannot_exceed_hard_budget(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)

    with pytest.raises(ContextBuildError, match="mandatory context exceeds"):
        harness.context_builder.build_planner(harness.events(), max_tokens=1)


def test_hidden_final_evidence_is_excluded_from_planner_context(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    events = harness.events()
    hidden_result = baseline_evaluation.model_copy(
        update={
            "experiment_id": "exp_hidden",
            "population": Population.HIDDEN_FINAL,
            "public_query_index": None,
            "metric_set": MetricSet(
                metrics={"gauc": 0.99, "ndcg@5": 0.99, "primary": 0.99},
                primary_metric_name="primary",
                primary_score=0.99,
            ),
        }
    )
    hidden_event = Event(
        event_id="evt_%06d" % (len(events) + 1),
        seq=len(events) + 1,
        timestamp=events[-1].timestamp,
        run_id=events[-1].run_id,
        event_type=EventType.EVALUATION_COMPLETED,
        idempotency_key="hidden-final-evidence",
        payload=EvaluationCompletedPayload(result=hidden_result),
        artifact_refs=[],
        prev_event_hash=events[-1].event_hash,
        event_hash="f" * 64,
    )

    context = harness.context_builder.build_planner([*events, hidden_event])

    assert context.excluded_source_ids[hidden_event.event_id] == "hidden_final"
    assert hidden_event.event_id not in context.source_event_ids
    assert "exp_hidden" not in context.content
    assert "0.99" not in context.content
