from __future__ import annotations

import asyncio

from tacorank.memory.retrieval import (
    verified_experiment_history,
    visible_development_events,
)
from tacorank.schemas import (
    EvaluationCompletedPayload,
    Fidelity,
    Population,
    TrustAssessment,
    TrustVerdict,
)


def test_hidden_final_and_suspicious_results_are_not_positive_planner_evidence(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    evaluation_event = next(
        event
        for event in reversed(harness.events())
        if event.payload.type == "evaluation.completed"
    )

    hidden_result = evaluation_event.payload.result.__class__.model_validate(
        {
            **evaluation_event.payload.result.model_dump(mode="json"),
            "population": Population.HIDDEN_FINAL.value,
            "fidelity": Fidelity.FINAL.value,
            "public_query_index": None,
        }
    )
    hidden_event = evaluation_event.__class__.model_validate(
        {
            **evaluation_event.model_dump(mode="json"),
            "payload": EvaluationCompletedPayload(result=hidden_result).model_dump(
                mode="json"
            ),
        }
    )
    assert hidden_event not in visible_development_events(
        list(harness.events()) + [hidden_event]
    )

    suspicious_result = evaluation_event.payload.result.__class__.model_validate(
        {
            **evaluation_event.payload.result.model_dump(mode="json"),
            "trust": TrustAssessment(
                **{
                    **evaluation_event.payload.result.trust.model_dump(mode="json"),
                    "verdict": TrustVerdict.SUSPICIOUS,
                }
            ).model_dump(mode="json"),
        }
    )
    suspicious_event = evaluation_event.__class__.model_validate(
        {
            **evaluation_event.model_dump(mode="json"),
            "payload": EvaluationCompletedPayload(result=suspicious_result).model_dump(
                mode="json"
            ),
        }
    )
    selected = verified_experiment_history([suspicious_event])
    assert selected == []
