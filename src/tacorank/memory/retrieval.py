"""Deterministic exact/graph/lexical retrieval over typed events."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from ..schemas import Event, EventType, LessonStatus, Population, TrustVerdict


def visible_development_events(events: Iterable[Event]) -> List[Event]:
    """Remove hidden-final evidence from every development context."""

    visible: List[Event] = []
    for event in events:
        if event.event_type == EventType.EVALUATION_COMPLETED:
            if event.payload.result.population == Population.HIDDEN_FINAL:
                continue
        visible.append(event)
    return visible


def active_lessons(
    events: Sequence[Event], *, tags: Iterable[str] = (), limit: int = 5
) -> List[Event]:
    tag_set = {tag.lower() for tag in tags}
    statuses = {}
    recorded = {}
    for event in events:
        if event.event_type == EventType.LESSON_RECORDED:
            recorded[event.payload.lesson_id] = event
            statuses[event.payload.lesson_id] = LessonStatus.ACTIVE
        elif event.event_type == EventType.LESSON_STATUS_CHANGED:
            statuses[event.payload.lesson_id] = event.payload.status
    candidates = [
        event
        for lesson_id, event in recorded.items()
        if statuses.get(lesson_id) == LessonStatus.ACTIVE
        and (
            not tag_set
            or tag_set.intersection(
                {tag.lower() for tag in event.payload.candidate.tags}
            )
        )
    ]
    candidates.sort(key=lambda event: (-event.seq, event.event_id))
    return candidates[:limit]


def verified_experiment_history(
    events: Sequence[Event], *, family: Optional[str] = None, limit: int = 10
) -> List[Event]:
    """Return accepted evaluations for consumers that require positive evidence."""

    accepted_evaluations = [
        event
        for event in recent_experiment_feedback(
            events, family=family, limit=len(events)
        )
        if event.payload.result.trust.verdict == TrustVerdict.ACCEPTED
    ]
    return accepted_evaluations[:limit]


def recent_experiment_feedback(
    events: Sequence[Event], *, family: Optional[str] = None, limit: int = 10
) -> List[Event]:
    """Return recent development evaluations regardless of their trust verdict.

    This is planner working memory, not positive evidence or durable lesson
    memory. Hidden-final evaluations are excluded even when callers have not
    already applied the global development-visibility policy.
    """

    family_by_experiment = {}
    for event in events:
        if event.event_type == EventType.EXPERIMENT_PROPOSED:
            family_by_experiment[event.payload.spec.experiment_id] = (
                event.payload.spec.family
            )
    evaluations = []
    for event in events:
        if event.event_type != EventType.EVALUATION_COMPLETED:
            continue
        result = event.payload.result
        if result.population == Population.HIDDEN_FINAL:
            continue
        if (
            family is not None
            and family_by_experiment.get(result.experiment_id) != family
        ):
            continue
        evaluations.append(event)
    evaluations.sort(key=lambda event: (-event.seq, event.event_id))
    return evaluations[:limit]


def experiment_events(events: Sequence[Event], experiment_id: str) -> List[Event]:
    selected = []
    for event in events:
        data = event.payload.model_dump(mode="python")
        if data.get("experiment_id") == experiment_id:
            selected.append(event)
            continue
        if any(
            isinstance(value, dict) and value.get("experiment_id") == experiment_id
            for value in data.values()
        ):
            selected.append(event)
    return selected


def failure_chain(events: Sequence[Event], experiment_id: str, limit: int = 12) -> List[Event]:
    candidates = experiment_events(events, experiment_id)
    failures = [
        event
        for event in candidates
        if event.event_type
        in (
            EventType.PATCH_CHECKED,
            EventType.EXECUTION_FINISHED,
            EventType.ADAPTER_FAILED,
            EventType.OUTPUT_CHECKED,
            EventType.EVALUATION_COMPLETED,
            EventType.RECOVERY_DECIDED,
        )
    ]
    return failures[-limit:]
