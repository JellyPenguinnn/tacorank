"""Ledger replay entry point with optional artifact revalidation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..artifacts import ArtifactStore
from ..orchestrator.state import RunState
from ..schemas import Event
from .event_store import EventStore
from .projections import project


def replay(
    source: Iterable[Event],
    *,
    artifact_store: Optional[ArtifactStore] = None,
    validate_transitions: bool = True,
) -> RunState:
    events = list(source)
    if validate_transitions:
        from ..orchestrator.state_machine import validate_transition

        prefix = []
        for event in events:
            validate_transition(prefix, event.payload)
            prefix.append(event)
    if artifact_store is not None:
        seen = set()
        for event in events:
            for artifact in event.artifact_refs:
                if artifact.artifact_id not in seen:
                    artifact_store.verify(artifact)
                    seen.add(artifact.artifact_id)
    return project(events)


def replay_ledger(
    ledger_path: Path,
    *,
    artifact_store: Optional[ArtifactStore] = None,
    repair_tail: bool = False,
) -> RunState:
    store = EventStore(ledger_path, artifact_store=artifact_store)
    return replay(
        store.read_events(repair_tail=repair_tail), artifact_store=artifact_store
    )
