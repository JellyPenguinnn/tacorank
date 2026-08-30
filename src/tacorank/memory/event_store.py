"""Crash-safe, append-only JSONL event store with a SHA-256 hash chain."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from pydantic import ValidationError

from ..artifacts import ArtifactStore
from ..schemas import (
    Event,
    EventPayload,
    EventType,
    ResourceDelta,
    payload_artifacts,
)
from .canonical_json import canonical_dumps, canonical_sha256, event_hash_input


try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

try:  # Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None


GENESIS_HASH = "0" * 64


class LedgerError(RuntimeError):
    pass


class LedgerCorruptionError(LedgerError):
    pass


class DuplicateIdempotencyKey(LedgerError):
    def __init__(self, key: str, event: Event):
        super().__init__("duplicate idempotency key %r (existing %s)" % (key, event.event_id))
        self.key = key
        self.event = event


def _validate_idempotency_payload_hash(key: str, payload: EventPayload) -> None:
    if key.rsplit(":", 1)[-1] != canonical_sha256(payload):
        raise ValueError("idempotency key input hash does not match payload")


class EventStore:
    def __init__(
        self,
        ledger_path: Path,
        *,
        artifact_store: Optional[ArtifactStore] = None,
        transition_validator: Optional[Callable[[List[Event], EventPayload], None]] = None,
    ):
        self.ledger_path = ledger_path
        self.lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
        self.artifact_store = artifact_store
        self.transition_validator = transition_validator

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(descriptor, "r+b") as lock_handle:
                if _fcntl is not None:
                    _fcntl.flock(lock_handle.fileno(), _fcntl.LOCK_EX)
                elif _msvcrt is not None:
                    # msvcrt locks a byte range rather than the whole file. Keep
                    # one stable byte in the sidecar so every process contends on
                    # the same range.
                    lock_handle.seek(0, os.SEEK_END)
                    if lock_handle.tell() == 0:
                        lock_handle.write(b"\0")
                        lock_handle.flush()
                    lock_handle.seek(0)
                    _msvcrt.locking(lock_handle.fileno(), _msvcrt.LK_LOCK, 1)
                else:  # pragma: no cover - every supported OS has one backend
                    raise LedgerError("no supported file-locking backend is available")
                try:
                    yield
                finally:
                    if _fcntl is not None:
                        _fcntl.flock(lock_handle.fileno(), _fcntl.LOCK_UN)
                    elif _msvcrt is not None:
                        lock_handle.seek(0)
                        _msvcrt.locking(lock_handle.fileno(), _msvcrt.LK_UNLCK, 1)
        finally:
            # fdopen owns and closes the descriptor in the normal path.
            pass

    def _repair_incomplete_tail_locked(self) -> None:
        if not self.ledger_path.exists():
            return
        data = self.ledger_path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        newline = data.rfind(b"\n")
        valid_size = 0 if newline < 0 else newline + 1
        with self.ledger_path.open("r+b") as handle:
            handle.truncate(valid_size)
            handle.flush()
            os.fsync(handle.fileno())

    def repair_incomplete_tail(self) -> bool:
        with self._locked():
            before = self.ledger_path.stat().st_size if self.ledger_path.exists() else 0
            self._repair_incomplete_tail_locked()
            after = self.ledger_path.stat().st_size if self.ledger_path.exists() else 0
            return before != after

    def _read_locked(self) -> List[Event]:
        if not self.ledger_path.exists():
            return []
        data = self.ledger_path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise LedgerCorruptionError("ledger has an incomplete trailing fragment")
        events: List[Event] = []
        previous_hash = GENESIS_HASH
        run_id: Optional[str] = None
        idempotency: Dict[str, Event] = {}
        complete_lines = data[:-1].split(b"\n") if data else []
        for line_number, raw_line in enumerate(complete_lines, start=1):
            try:
                decoded = json.loads(raw_line.decode("utf-8"))
                event = Event.model_validate(decoded)
                _validate_idempotency_payload_hash(
                    event.idempotency_key, decoded["payload"]
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
                raise LedgerCorruptionError(
                    "invalid complete ledger line %d: %s" % (line_number, exc)
                ) from exc
            except ValueError as exc:
                raise LedgerCorruptionError(
                    "invalid complete ledger line %d: %s" % (line_number, exc)
                ) from exc

            # Validate the exact stored representation. Re-serializing the
            # current model would inject defaults added by a later compatible
            # schema version and falsely classify an older hash-valid event as
            # corruption.
            if raw_line != canonical_dumps(decoded).encode("utf-8"):
                raise LedgerCorruptionError(
                    "ledger line %d is not canonical JSON" % line_number
                )

            expected_seq = len(events) + 1
            if event.seq != expected_seq:
                raise LedgerCorruptionError(
                    "non-contiguous sequence at line %d: expected %d" % (line_number, expected_seq)
                )
            if event.prev_event_hash != previous_hash:
                raise LedgerCorruptionError("prev_event_hash mismatch at %s" % event.event_id)
            expected_hash = canonical_sha256(event_hash_input(decoded))
            if event.event_hash != expected_hash:
                raise LedgerCorruptionError("event_hash mismatch at %s" % event.event_id)
            if run_id is None:
                run_id = event.run_id
            elif event.run_id != run_id:
                raise LedgerCorruptionError("a ledger may contain only one run_id")
            if event.idempotency_key in idempotency:
                raise LedgerCorruptionError("duplicate idempotency key in ledger")
            if event.causation_event_id and event.causation_event_id not in {
                item.event_id for item in events
            }:
                raise LedgerCorruptionError("unknown causation event %s" % event.causation_event_id)
            if self.transition_validator is not None:
                try:
                    self.transition_validator(events, event.payload)
                except Exception as exc:
                    raise LedgerCorruptionError(
                        "illegal transition at %s: %s" % (event.event_id, exc)
                    ) from exc
            idempotency[event.idempotency_key] = event
            events.append(event)
            previous_hash = event.event_hash
        return events

    def read_events(self, *, repair_tail: bool = False) -> List[Event]:
        with self._locked():
            if repair_tail:
                self._repair_incomplete_tail_locked()
            return self._read_locked()

    def append(
        self,
        *,
        run_id: str,
        payload: EventPayload,
        idempotency_key: str,
        causation_event_id: Optional[str] = None,
        resource_delta: Optional[ResourceDelta] = None,
        timestamp: Optional[datetime] = None,
    ) -> Event:
        with self._locked():
            try:
                _validate_idempotency_payload_hash(idempotency_key, payload)
            except ValueError as exc:
                raise LedgerError(str(exc)) from exc
            self._repair_incomplete_tail_locked()
            events = self._read_locked()
            for existing in events:
                if existing.idempotency_key == idempotency_key:
                    raise DuplicateIdempotencyKey(idempotency_key, existing)
            if events and events[0].run_id != run_id:
                raise LedgerError("run_id does not match the existing ledger")
            if causation_event_id and causation_event_id not in {
                event.event_id for event in events
            }:
                raise LedgerError("causation_event_id does not exist")
            payload_data = payload.model_dump(mode="python")
            nested_run_ids = []

            def collect_run_ids(value: object) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key == "run_id" and isinstance(item, str):
                            nested_run_ids.append(item)
                        else:
                            collect_run_ids(item)
                elif isinstance(value, list):
                    for item in value:
                        collect_run_ids(item)

            collect_run_ids(payload_data)
            if any(item != run_id for item in nested_run_ids):
                raise LedgerError("payload run_id does not match event envelope")
            if self.transition_validator is not None:
                self.transition_validator(events, payload)

            artifacts = payload_artifacts(payload)
            if self.artifact_store is not None:
                for artifact in artifacts:
                    self.artifact_store.verify(artifact)

            seq = len(events) + 1
            event_type = EventType(payload.type)
            event_timestamp = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
            envelope = {
                "schema_version": "1.0",
                "event_id": "evt_%06d" % seq,
                "seq": seq,
                "timestamp": event_timestamp.isoformat().replace("+00:00", "Z"),
                "run_id": run_id,
                "event_type": event_type,
                "idempotency_key": idempotency_key,
                "causation_event_id": causation_event_id,
                "payload": payload,
                "artifact_refs": artifacts,
                "resource_delta": resource_delta or ResourceDelta(),
                "prev_event_hash": events[-1].event_hash if events else GENESIS_HASH,
            }
            dumped = {
                key: value.model_dump(mode="json", exclude_none=False)
                if hasattr(value, "model_dump")
                else value.value if isinstance(value, EventType) else value
                for key, value in envelope.items()
            }
            dumped["artifact_refs"] = [item.model_dump(mode="json") for item in artifacts]
            dumped["event_hash"] = canonical_sha256(dumped)
            event = Event.model_validate(dumped)

            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("ab") as handle:
                handle.write(canonical_dumps(event).encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def get_by_idempotency_key(self, key: str) -> Optional[Event]:
        for event in self.read_events():
            if event.idempotency_key == key:
                return event
        return None
