from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from tacorank.schemas import (
    CheckResult,
    CheckStatus,
    Event,
    EventType,
    OutputCheckResult,
    PatchCheckResult,
    ResourceDelta,
    SubmissionCheckedPayload,
)


def completed_events(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    return list(harness.events())


def test_envelope_resources_must_match_nested_adapter_resources(
    harness, baseline_evaluation
):
    event = next(
        event
        for event in completed_events(harness, baseline_evaluation)
        if event.event_type == EventType.EXECUTION_FINISHED
        and event.resource_delta.wall_time_ms > 0
    )
    data = event.model_dump(mode="json")
    data["resource_delta"] = ResourceDelta().model_dump(mode="json")

    with pytest.raises(ValidationError, match="resource_delta must match"):
        Event.model_validate(data)


def test_accepted_patch_gate_cannot_contain_failed_check(
    harness, baseline_evaluation
):
    event = next(
        event
        for event in completed_events(harness, baseline_evaluation)
        if event.event_type == EventType.PATCH_CHECKED
    )
    data = event.payload.result.model_dump(mode="json")
    data["checks"][0]["status"] = CheckStatus.FAIL.value

    with pytest.raises(ValidationError, match="accepted patch checks"):
        PatchCheckResult.model_validate(data)


def test_accepted_output_gate_cannot_contain_failed_check(
    harness, baseline_evaluation
):
    event = next(
        event
        for event in completed_events(harness, baseline_evaluation)
        if event.event_type == EventType.OUTPUT_CHECKED
    )
    data = event.payload.result.model_dump(mode="json")
    data["checks"]["schema"] = CheckStatus.FAIL.value

    with pytest.raises(ValidationError, match="accepted output checks"):
        OutputCheckResult.model_validate(data)


def test_accepted_submission_cannot_contain_failed_check(harness, baseline_evaluation):
    artifact = next(
        event.payload.result.receipt_artifact
        for event in completed_events(harness, baseline_evaluation)
        if event.event_type == EventType.PATCH_CHECKED
    )

    with pytest.raises(ValidationError, match="accepted submissions"):
        SubmissionCheckedPayload(
            accepted=True,
            submission_artifact=artifact,
            checks=[CheckResult(name="schema", status=CheckStatus.FAIL)],
        )
