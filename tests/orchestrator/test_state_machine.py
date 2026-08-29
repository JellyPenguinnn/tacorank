from __future__ import annotations

import asyncio

import pytest

from tacorank.orchestrator.state_machine import TransitionError, validate_transition
from tacorank.schemas import EventType


def _completed_events(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    return harness.events()


def _proposal(events):
    index = next(
        index
        for index, event in enumerate(events)
        if event.event_type == EventType.EXPERIMENT_PROPOSED
    )
    return index, events[index].payload


def test_proposal_requires_an_existing_planner_context(harness, baseline_evaluation):
    events = _completed_events(harness, baseline_evaluation)
    _, proposal = _proposal(events)
    baseline_prefix = events[:3]

    with pytest.raises(TransitionError, match="proposal requires planner context"):
        validate_transition(baseline_prefix, proposal)


def test_proposal_context_must_match_latest_planner_context(harness, baseline_evaluation):
    events = _completed_events(harness, baseline_evaluation)
    proposal_index, proposal = _proposal(events)
    invalid_spec = proposal.spec.model_copy(update={"context_id": "ctx_wrong"})
    invalid_proposal = proposal.model_copy(update={"spec": invalid_spec})

    with pytest.raises(TransitionError, match="proposal context mismatch"):
        validate_transition(events[:proposal_index], invalid_proposal)


def test_parent_commit_must_match_selected_parent(harness, baseline_evaluation):
    events = _completed_events(harness, baseline_evaluation)
    proposal_index, proposal = _proposal(events)
    invalid_spec = proposal.spec.model_copy(update={"parent_commit_sha": "c" * 40})
    invalid_proposal = proposal.model_copy(update={"spec": invalid_spec})

    with pytest.raises(TransitionError, match="parent_commit_sha"):
        validate_transition(events[:proposal_index], invalid_proposal)
