from __future__ import annotations

import json

from tacorank.memory.canonical_json import canonical_sha256
from tacorank.memory.event_store import EventStore
from tacorank.reporting import rebuild_views
from tacorank.run_layout import (
    RunLayout,
    experiment_artifact_prefix,
    run_artifact_root,
)
from tacorank.schemas import (
    LessonCandidate,
    LessonCategory,
    LessonOrigin,
    LessonRecordedPayload,
    LessonStatus,
    LessonStatusChangedPayload,
    RunStartedPayload,
)


def _key(run_id: str, stage: str, payload: object) -> str:
    return "%s:run:%s:0:%s" % (run_id, stage, canonical_sha256(payload))


def test_run_layout_uses_run_scoped_artifacts(repository):
    layout = RunLayout(repository, "run_001")
    assert layout.ledger == repository / "runs/run_001/events.jsonl"
    assert layout.state == repository / "runs/run_001/state.json"
    assert run_artifact_root("run_001") == "runs/run_001/artifacts"
    assert (
        experiment_artifact_prefix("run_001", "exp_012", attempt=2)
        == "runs/run_001/artifacts/exp_012/attempt_002"
    )


def test_rebuild_materializes_per_lesson_memory_and_head_bound_state(repository):
    run_id = "run_lessons"
    layout = RunLayout(repository, run_id)
    store = EventStore(layout.ledger)
    started_payload = RunStartedPayload(
        config_sha256="1" * 64,
        contract_sha256="2" * 64,
        protected_paths_sha256="3" * 64,
        max_experiments=4,
        wall_time_limit_seconds=60,
        seed_schedule=[1],
    )
    started = store.append(
        run_id=run_id,
        payload=started_payload,
        idempotency_key=_key(run_id, "started", started_payload),
    )
    recorded_payload = LessonRecordedPayload(
        lesson_id="lesson_0001",
        candidate=LessonCandidate(
            origin=LessonOrigin.OPERATIONAL,
            category=LessonCategory.PROCESS_RULE,
            tags=["recovery", "gpu"],
            summary="Reduce the batch size after a verified GPU-memory failure.",
            applicability="The error fingerprint identifies GPU memory exhaustion.",
            avoid_when="The failure is caused by invalid model code.",
            confidence=0.9,
            source_event_ids=[started.event_id],
            source_commit_shas=[],
        ),
    )
    recorded = store.append(
        run_id=run_id,
        payload=recorded_payload,
        idempotency_key=_key(run_id, "lesson", recorded_payload),
        causation_event_id=started.event_id,
    )
    rebuild_views(layout.run_directory, store.read_events())

    state = json.loads(layout.state.read_text(encoding="utf-8"))
    assert state["derived_from"]["event_id"] == recorded.event_id
    assert not layout.state.with_suffix(".json.tmp").exists()
    index = (layout.lessons / "INDEX.md").read_text(encoding="utf-8")
    lesson = (layout.lessons / "lesson_0001.md").read_text(encoding="utf-8")
    assert "lesson_0001" in index
    assert "`active`" in lesson
    assert "Reduce the batch size" in lesson

    stale_payload = LessonStatusChangedPayload(
        lesson_id="lesson_0001",
        status=LessonStatus.STALE,
        reason="The execution frame changed.",
    )
    stale = store.append(
        run_id=run_id,
        payload=stale_payload,
        idempotency_key=_key(run_id, "lesson_status", stale_payload),
        causation_event_id=recorded.event_id,
    )
    rebuild_views(layout.run_directory, store.read_events())
    lesson = (layout.lessons / "lesson_0001.md").read_text(encoding="utf-8")
    assert "`stale`" in lesson
    assert stale.event_id in lesson
    assert "The execution frame changed" in lesson
