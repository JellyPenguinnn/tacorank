from __future__ import annotations

import asyncio

import pytest

from tacorank.memory.canonical_json import canonical_sha256
from tacorank.orchestrator.convergence import StopDecision
from tacorank.orchestrator.state import RunStatus
from tacorank.orchestrator.state_machine import TransitionError, validate_transition
from tacorank.schemas import (
    EventType,
    ExecutionFinishedPayload,
    ExperimentDecidedPayload,
    ExperimentDecisionKind,
    Fidelity,
    FinalSelectedPayload,
    Population,
    RecoveryAction,
    RecoveryDecidedPayload,
    RecoveryDecision,
    RunOutcome,
)


def completed_events(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    return list(harness.events())


def event_index(events, event_type, predicate=lambda event: True):
    return next(
        index
        for index, event in enumerate(events)
        if event.event_type == event_type and predicate(event)
    )


def test_evaluation_result_must_match_execution_and_frozen_hashes(
    harness, baseline_evaluation
):
    events = completed_events(harness, baseline_evaluation)
    index = event_index(
        events,
        EventType.EVALUATION_COMPLETED,
        lambda event: event.payload.result.fidelity == Fidelity.PROXY,
    )
    original = events[index].payload
    relabeled = original.result.model_copy(
        update={
            "attempt": original.result.attempt + 1,
            "fidelity": Fidelity.FULL,
            "population": Population.PUBLIC_VALIDATION,
            "seed": original.result.seed + 1,
            "public_query_index": 1,
            "evaluator_sha256": "c" * 64,
            "contract_sha256": "d" * 64,
        }
    )

    with pytest.raises(
        TransitionError, match="evaluation attempt|execution request|frozen contract"
    ):
        validate_transition(
            events[:index], original.model_copy(update={"result": relabeled})
        )


def _failed_execution_prefix(events):
    index = event_index(events, EventType.EXECUTION_FINISHED)
    original = events[index]
    failed_result = original.payload.result.model_copy(
        update={
            "outcome": RunOutcome.CODE_ERROR,
            "error_class": "SyntheticError",
            "error_fingerprint": "synthetic-error",
        }
    )
    failed_event = original.model_copy(
        update={"payload": ExecutionFinishedPayload(result=failed_result)}
    )
    return [*events[:index], failed_event], failed_event


def _recovery_payload(run_id, experiment_id, failure_event_id, repair_attempt=1):
    return RecoveryDecidedPayload(
        decision=RecoveryDecision(
            run_id=run_id,
            experiment_id=experiment_id,
            failure_event_id=failure_event_id,
            repair_attempt=repair_attempt,
            action=RecoveryAction.TRAE_REPAIR,
            reason_code="synthetic_repair",
            instructions="Repair only the recorded failure.",
            same_error_count=1,
            remaining_repair_budget=2 - repair_attempt,
        )
    )


def test_recovery_must_reference_latest_failure_and_use_next_slot(
    harness, baseline_evaluation
):
    events = completed_events(harness, baseline_evaluation)
    prefix, failure = _failed_execution_prefix(events)
    experiment_id = failure.payload.result.experiment_id
    invalid_reference = _recovery_payload(
        failure.run_id, experiment_id, "evt_000001"
    )
    with pytest.raises(TransitionError, match="latest failure"):
        validate_transition(prefix, invalid_reference)

    first = _recovery_payload(failure.run_id, experiment_id, failure.event_id)
    validate_transition(prefix, first)
    first_event = failure.model_copy(
        update={
            "event_id": "evt_%06d" % (failure.seq + 1),
            "seq": failure.seq + 1,
            "event_type": EventType.RECOVERY_DECIDED,
            "idempotency_key": "%s:%s:synthetic_recovery:1:%s"
            % (failure.run_id, experiment_id, canonical_sha256(first)),
            "payload": first,
            "artifact_refs": [],
            "resource_delta": first.decision.resource_delta,
            "prev_event_hash": failure.event_hash,
            "event_hash": "e" * 64,
        }
    )
    with pytest.raises(TransitionError, match="next available budget slot"):
        validate_transition([*prefix, first_event], first)


def test_runtime_budget_is_rechecked_after_planner_action(harness, baseline_evaluation):
    harness.config.token_limit = 100
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())
    event_types = [event.event_type for event in harness.events()]

    assert state.status == RunStatus.STOPPED
    assert event_types[-1] == EventType.RUN_STOPPED
    assert EventType.EXPERIMENT_PROPOSED in event_types
    assert EventType.PATCH_CREATED not in event_types


def test_run_stopped_rejects_inflight_development_transitions(
    harness, baseline_evaluation
):
    events = completed_events(harness, baseline_evaluation)
    execution_result = next(
        event.payload
        for event in events
        if event.event_type == EventType.EXECUTION_FINISHED
    )
    harness.stop(StopDecision(True, "test_stop", "Stop for terminality test."))

    with pytest.raises(TransitionError, match="only hidden-final|final selection"):
        validate_transition(list(harness.events()), execution_result)


def test_final_selection_requires_exact_commit_and_existing_trusted_evidence(
    harness, baseline_evaluation
):
    events = completed_events(harness, baseline_evaluation)
    full_evaluation = next(
        event
        for event in events
        if event.event_type == EventType.EVALUATION_COMPLETED
        and event.payload.result.fidelity == Fidelity.FULL
    )
    harness.stop(StopDecision(True, "test_stop", "Stop for final-selection test."))
    stopped = list(harness.events())
    state = harness.state()

    with pytest.raises(TransitionError, match="final commit"):
        validate_transition(
            stopped,
            FinalSelectedPayload(
                experiment_id=state.best_experiment_id,
                commit_sha="c" * 40,
                reproduction_evaluation_event_id=full_evaluation.event_id,
            ),
        )
    with pytest.raises(TransitionError, match="unknown reproduction"):
        validate_transition(
            stopped,
            FinalSelectedPayload(
                experiment_id=state.best_experiment_id,
                commit_sha=state.best_commit_sha,
                reproduction_evaluation_event_id="evt_999999",
            ),
        )

    validate_transition(
        stopped,
        FinalSelectedPayload(
            experiment_id=state.best_experiment_id,
            commit_sha=state.best_commit_sha,
            reproduction_evaluation_event_id=full_evaluation.event_id,
        ),
    )


def test_rejected_decision_cannot_be_best_eligible(harness, baseline_evaluation):
    events = completed_events(harness, baseline_evaluation)
    index = event_index(
        events,
        EventType.EXPERIMENT_DECIDED,
        lambda event: event.payload.decision.fidelity_completed == Fidelity.FULL,
    )
    original = events[index].payload
    rejected = original.decision.model_copy(
        update={
            "decision": ExperimentDecisionKind.REJECT,
            "best_eligible": True,
            "next_fidelity": None,
        }
    )

    with pytest.raises(TransitionError, match="accepted experiment"):
        validate_transition(
            events[:index], original.model_copy(update={"decision": rejected})
        )


def test_best_update_requires_patch_commit_and_strict_improvement(
    harness, baseline_evaluation
):
    events = completed_events(harness, baseline_evaluation)
    index = event_index(events, EventType.BEST_UPDATED)
    original = events[index].payload

    with pytest.raises(TransitionError, match="best commit"):
        validate_transition(
            events[:index], original.model_copy(update={"commit_sha": "c" * 40})
        )
    with pytest.raises(TransitionError, match="strictly improve"):
        validate_transition(events, original)


def test_promotion_cannot_move_backward_in_fidelity(harness, baseline_evaluation):
    events = completed_events(harness, baseline_evaluation)
    index = event_index(
        events,
        EventType.EXPERIMENT_DECIDED,
        lambda event: event.payload.decision.fidelity_completed == Fidelity.PROXY,
    )
    original = events[index].payload
    backward = original.decision.model_copy(update={"next_fidelity": Fidelity.SMOKE})

    with pytest.raises(TransitionError, match="higher fidelity"):
        validate_transition(
            events[:index], ExperimentDecidedPayload(decision=backward)
        )
