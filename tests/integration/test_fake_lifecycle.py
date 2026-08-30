from __future__ import annotations

import asyncio

from tacorank.memory.replay import replay
from tacorank.orchestrator.state import ExperimentStatus
from tacorank.schemas import EventType, Fidelity


def test_complete_fake_lifecycle_is_replayable(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    state = asyncio.run(harness.run_one_experiment())

    assert state.best_experiment_id == "exp_001"
    assert state.best_primary_score == 0.62
    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    assert state.full_evaluations_completed == 3
    assert state.public_validation_queries == 3
    assert state.resource_totals.provider_tokens == 160
    assert state.resource_totals.estimated_tokens == 200
    assert list(state.lessons) == ["lesson_001"]

    events = harness.events()
    assert events[-1].event_type == EventType.BEST_UPDATED
    full_seeds = [
        event.payload.result.seed
        for event in events
        if event.event_type == EventType.EVALUATION_COMPLETED
        and event.payload.result.fidelity.value == "full"
    ]
    assert full_seeds == [11, 22, 33]
    assert harness._execution_seed("exp_001", Fidelity.FULL, "d" * 40) == 11
    lesson_index = (
        harness.config.repository_root
        / "runs"
        / harness.config.run_id
        / "lessons"
        / "INDEX.md"
    ).read_text(encoding="utf-8")
    assert "lesson_001" in lesson_index
    replayed = replay(events, artifact_store=harness.event_store.artifact_store)
    assert replayed.best_experiment_id == state.best_experiment_id
    assert replayed.best_primary_score == state.best_primary_score
    assert replayed.last_event_hash == state.last_event_hash


def test_resume_does_not_duplicate_bootstrap(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    first = list(harness.events())
    harness.bootstrap(baseline_evaluation)
    assert harness.events() == first
