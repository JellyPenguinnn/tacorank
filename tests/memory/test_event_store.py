from __future__ import annotations

import json

import pytest

from tacorank.artifacts import ArtifactStore
from tacorank.memory.event_store import (
    DuplicateIdempotencyKey,
    EventStore,
    LedgerCorruptionError,
)
from tacorank.orchestrator.state_machine import TransitionError, validator
from tacorank.schemas import RunStartedPayload


def started_payload():
    return RunStartedPayload(
        config_sha256="1" * 64,
        contract_sha256="2" * 64,
        protected_paths_sha256="3" * 64,
        max_experiments=5,
        wall_time_limit_seconds=60,
        seed_schedule=[1],
    )


def test_hash_chain_and_duplicate_idempotency(repository):
    store = EventStore(
        repository / "runs/r/events.jsonl",
        artifact_store=ArtifactStore(repository),
        transition_validator=validator,
    )
    event = store.append(
        run_id="r", payload=started_payload(), idempotency_key="r:run:start:0:hash"
    )
    assert event.event_id == "evt_000001"
    assert store.read_events()[0].event_hash == event.event_hash
    with pytest.raises(DuplicateIdempotencyKey):
        store.append(
            run_id="r", payload=started_payload(), idempotency_key="r:run:start:0:hash"
        )
    with pytest.raises(TransitionError):
        store.append(
            run_id="r", payload=started_payload(), idempotency_key="r:run:start:0:other"
        )


def test_complete_line_tampering_is_detected(repository):
    path = repository / "runs/r/events.jsonl"
    store = EventStore(path, transition_validator=validator)
    store.append(run_id="r", payload=started_payload(), idempotency_key="key")
    data = json.loads(path.read_text().strip())
    data["payload"]["max_experiments"] = 999
    path.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n")
    with pytest.raises(LedgerCorruptionError, match="event_hash mismatch"):
        store.read_events()


def test_only_incomplete_tail_is_truncated(repository):
    path = repository / "runs/r/events.jsonl"
    store = EventStore(path, transition_validator=validator)
    first = store.append(run_id="r", payload=started_payload(), idempotency_key="key")
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    with pytest.raises(LedgerCorruptionError, match="incomplete"):
        store.read_events()
    assert store.repair_incomplete_tail()
    assert store.read_events() == [first]
