from types import SimpleNamespace

import pytest

from tacorank.evaluation.stability import (
    aggregate_metric_sets,
    confirmed_seed_evaluation_events,
    mean_mapping,
    stable_primary_score,
)
from tacorank.schemas import EventType, Fidelity, Integrity, MetricSet, Population


def _event(event_id, score, *, seed_ids=(), seed_mean=None, seed_count=1):
    result = SimpleNamespace(
        run_id="run_local",
        experiment_id="exp_001",
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        metric_set=MetricSet(
            metrics={"primary": score, "gauc": score},
            primary_metric_name="primary",
            primary_score=score,
        ),
        trust=SimpleNamespace(
            seed_mean=seed_mean,
            seed_count=seed_count,
            integrity=Integrity.CLEAN,
        ),
        seed_evidence_event_ids=list(seed_ids),
        parent_metric_deltas={"primary": score - 0.59, "gauc": score - 0.59},
    )
    return SimpleNamespace(
        event_id=event_id,
        event_type=EventType.EVALUATION_COMPLETED,
        payload=SimpleNamespace(result=result),
    )


def test_confirmed_projection_uses_only_declared_current_run_seeds():
    first = _event("evt_1", 0.60)
    second = _event("evt_2", 0.60)
    terminal = _event(
        "evt_3",
        0.6031,
        seed_ids=("evt_1", "evt_2"),
        seed_mean=0.6010333333333333,
        seed_count=3,
    )

    selected = confirmed_seed_evaluation_events(
        (first, second, terminal), terminal
    )
    aggregate = aggregate_metric_sets(
        [event.payload.result.metric_set for event in selected]
    )

    assert [event.event_id for event in selected] == ["evt_1", "evt_2", "evt_3"]
    assert aggregate.primary_score == pytest.approx(0.6010333333333333)
    assert mean_mapping(
        [event.payload.result.parent_metric_deltas for event in selected]
    )["primary"] == pytest.approx(0.0110333333333333)
    assert stable_primary_score(terminal.payload.result) == pytest.approx(
        0.6010333333333333
    )


def test_missing_seed_reference_cannot_be_treated_as_stable():
    terminal = _event(
        "evt_3",
        0.6031,
        seed_ids=("missing", "evt_2"),
        seed_mean=0.601,
        seed_count=3,
    )

    assert confirmed_seed_evaluation_events((terminal,), terminal) == ()
    assert stable_primary_score(terminal.payload.result) == pytest.approx(0.601)


def test_seed_projection_rejects_evidence_from_another_run():
    foreign = _event("evt_1", 0.60)
    foreign.payload.result.run_id = "run_other"
    terminal = _event(
        "evt_3",
        0.6031,
        seed_ids=("evt_1", "evt_2"),
        seed_mean=0.601,
        seed_count=3,
    )

    assert confirmed_seed_evaluation_events((foreign, terminal), terminal) == ()
