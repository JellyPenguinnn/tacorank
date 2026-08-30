from __future__ import annotations

import asyncio

from tacorank.memory.retrieval import (
    recent_experiment_feedback,
    verified_experiment_history,
    visible_development_events,
)
from tacorank.schemas import (
    EvaluationCompletedPayload,
    Fidelity,
    Population,
    Stability,
    TrustAssessment,
    TrustVerdict,
)


def test_hidden_final_is_excluded_but_suspicious_result_remains_working_memory(
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
    assert recent_experiment_feedback([hidden_event]) == []

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
    assert recent_experiment_feedback([suspicious_event]) == [suspicious_event]
    assert verified_experiment_history([suspicious_event]) == []

    negative_proxy_result = evaluation_event.payload.result.__class__.model_validate(
        {
            **evaluation_event.payload.result.model_dump(mode="json"),
            "population": Population.INTERNAL_PROXY.value,
            "fidelity": Fidelity.PROXY.value,
            "public_query_index": None,
            "trust": TrustAssessment(
                **{
                    **evaluation_event.payload.result.trust.model_dump(mode="json"),
                    "verdict": TrustVerdict.NEGATIVE,
                    "stability": Stability.NOT_APPLICABLE,
                }
            ).model_dump(mode="json"),
        }
    )
    negative_proxy_event = evaluation_event.__class__.model_validate(
        {
            **evaluation_event.model_dump(mode="json"),
            "payload": EvaluationCompletedPayload(
                result=negative_proxy_result
            ).model_dump(mode="json"),
        }
    )
    assert recent_experiment_feedback([negative_proxy_event]) == [
        negative_proxy_event
    ]
    assert verified_experiment_history([negative_proxy_event]) == []
