from __future__ import annotations

import json

import pytest

import tacorank.memory.event_store as event_store_module
from tacorank.artifacts import ArtifactStore
from tacorank.memory.canonical_json import canonical_dumps, canonical_sha256, event_hash_input
from tacorank.memory.event_store import (
    DuplicateIdempotencyKey,
    EventStore,
    LedgerError,
    LedgerCorruptionError,
)
from tacorank.orchestrator.state_machine import TransitionError, validator
from tacorank.schemas import (
    ArtifactKind,
    ArtifactRef,
    RunStartedPayload,
    SubmissionCheckedPayload,
)


def started_payload():
    return RunStartedPayload(
        config_sha256="1" * 64,
        contract_sha256="2" * 64,
        protected_paths_sha256="3" * 64,
        max_experiments=5,
        wall_time_limit_seconds=60,
        seed_schedule=[1],
    )


def event_key(payload, *, experiment_id="run", stage="start", attempt=0):
    return "r:%s:%s:%d:%s" % (
        experiment_id,
        stage,
        attempt,
        canonical_sha256(payload),
    )


def test_hash_chain_and_duplicate_idempotency(repository):
    store = EventStore(
        repository / "runs/r/events.jsonl",
        artifact_store=ArtifactStore(repository),
        transition_validator=validator,
    )
    payload = started_payload()
    event = store.append(run_id="r", payload=payload, idempotency_key=event_key(payload))
    assert event.event_id == "evt_000001"
    assert store.read_events()[0].event_hash == event.event_hash
    with pytest.raises(DuplicateIdempotencyKey):
        store.append(
            run_id="r", payload=payload, idempotency_key=event_key(payload)
        )
    with pytest.raises(TransitionError):
        store.append(
            run_id="r",
            payload=payload,
            idempotency_key=event_key(payload, stage="start_again"),
        )


def test_backward_compatible_defaults_do_not_rewrite_old_ledger(repository):
    path = repository / "runs/r/events.jsonl"
    store = EventStore(path, transition_validator=validator)
    payload = started_payload()
    store.append(run_id="r", payload=payload, idempotency_key=event_key(payload))

    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["payload"].pop("convergence_epsilon")
    legacy["payload"].pop("convergence_patience")
    payload_hash = canonical_sha256(legacy["payload"])
    legacy["idempotency_key"] = "r:run:start:0:%s" % payload_hash
    legacy["event_hash"] = canonical_sha256(event_hash_input(legacy))
    path.write_text(canonical_dumps(legacy) + "\n", encoding="utf-8")

    restored = store.read_events()[0]
    assert restored.payload.convergence_epsilon == 0.002
    assert restored.payload.convergence_patience == 3


def test_complete_line_tampering_is_detected(repository):
    path = repository / "runs/r/events.jsonl"
    store = EventStore(path, transition_validator=validator)
    payload = started_payload()
    store.append(run_id="r", payload=payload, idempotency_key=event_key(payload))
    data = json.loads(path.read_text().strip())
    data["payload"]["max_experiments"] = 999
    path.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n")
    with pytest.raises(LedgerCorruptionError, match="idempotency key|event_hash mismatch"):
        store.read_events()


def test_only_incomplete_tail_is_truncated(repository):
    path = repository / "runs/r/events.jsonl"
    store = EventStore(path, transition_validator=validator)
    payload = started_payload()
    first = store.append(run_id="r", payload=payload, idempotency_key=event_key(payload))
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    with pytest.raises(LedgerCorruptionError, match="incomplete"):
        store.read_events()
    assert store.repair_incomplete_tail()
    assert store.read_events() == [first]


def test_windows_lock_backend_uses_a_stable_byte(repository, monkeypatch):
    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        def __init__(self):
            self.calls = []

        def locking(self, descriptor, mode, size):
            self.calls.append((descriptor, mode, size))

    backend = FakeMsvcrt()
    monkeypatch.setattr(event_store_module, "_fcntl", None)
    monkeypatch.setattr(event_store_module, "_msvcrt", backend)
    store = EventStore(repository / "runs/r/events.jsonl")

    with store._locked():
        assert store.lock_path.read_bytes() == b"\0"

    assert [(mode, size) for _, mode, size in backend.calls] == [
        (backend.LK_LOCK, 1),
        (backend.LK_UNLCK, 1),
    ]


def test_nested_payload_artifacts_cannot_be_removed_from_envelope(repository):
    path = repository / "runs/r/events.jsonl"
    artifact = ArtifactRef(
        artifact_id="submission",
        kind=ArtifactKind.SUBMISSION,
        path="artifacts/submission.csv",
        sha256="a" * 64,
        size_bytes=10,
    )
    store = EventStore(path)
    payload = SubmissionCheckedPayload(
        accepted=True,
        submission_artifact=artifact,
        checks=[],
    )
    store.append(
        run_id="r",
        payload=payload,
        idempotency_key=event_key(payload, stage="submission"),
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    data["artifact_refs"] = []
    # Re-hash the tampered envelope so the artifact invariant, rather than the
    # hash-chain check, is what rejects it.
    data["event_hash"] = canonical_sha256(event_hash_input(data))
    path.write_text(canonical_dumps(data) + "\n", encoding="utf-8")

    with pytest.raises(LedgerCorruptionError, match="artifact_refs must exactly match"):
        store.read_events()


@pytest.mark.parametrize(
    "key",
    ["arbitrary", "r:run:start:0:%s" % ("0" * 64)],
)
def test_arbitrary_idempotency_keys_are_rejected(repository, key):
    store = EventStore(repository / "runs/r/events.jsonl")
    with pytest.raises(LedgerError, match="idempotency key input hash"):
        store.append(run_id="r", payload=started_payload(), idempotency_key=key)


def test_noncanonical_jsonl_is_rejected(repository):
    path = repository / "runs/r/events.jsonl"
    payload = started_payload()
    store = EventStore(path)
    store.append(run_id="r", payload=payload, idempotency_key=event_key(payload))
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(data, sort_keys=False) + "\n", encoding="utf-8")

    with pytest.raises(LedgerCorruptionError, match="not canonical JSON"):
        store.read_events()
