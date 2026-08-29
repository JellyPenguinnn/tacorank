"""Derived human-readable evidence views; events remain the authority."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

from ..memory.projections import project
from ..schemas import Event, LessonStatus


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def render_status(events: Sequence[Event]) -> str:
    state = project(events)
    payload = {
        "run_id": state.run_id,
        "status": state.status.value,
        "phase": state.phase,
        "last_event_id": state.last_event_id,
        "active_experiment_id": state.active_experiment_id,
        "active_attempt": state.active_attempt,
        "active_fidelity": state.active_fidelity.value if state.active_fidelity else None,
        "best_experiment_id": state.best_experiment_id,
        "best_commit_sha": state.best_commit_sha,
        "best_primary_score": state.best_primary_score,
        "experiments_proposed": state.experiments_proposed,
        "remaining_experiments": state.remaining_experiments,
        "full_evaluations_completed": state.full_evaluations_completed,
        "public_validation_queries": state.public_validation_queries,
        "manual_interventions": state.manual_intervention_count,
    }
    return "# TacoRank status\n\n```json\n%s\n```\n" % json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    )


def render_lessons(events: Sequence[Event]) -> str:
    state = project(events)
    lines = ["# Active lessons", ""]
    active = [lesson for lesson in state.lessons.values() if lesson.status == LessonStatus.ACTIVE.value]
    active.sort(key=lambda lesson: lesson.lesson_id)
    if not active:
        lines.append("No active lessons recorded.")
    for lesson in active:
        lines.extend(
            (
                "## %s" % lesson.lesson_id,
                "",
                lesson.summary,
                "",
                "Tags: %s" % (", ".join(lesson.tags) if lesson.tags else "none"),
                "Evidence: `%s`" % lesson.source_event_id,
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_summary(events: Sequence[Event]) -> str:
    state = project(events)
    lines = [
        "# TacoRank run summary",
        "",
        "| Experiment | Family | Status | Fidelity | Primary | Commit |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for node in sorted(state.experiments.values(), key=lambda item: item.experiment_id):
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                node.experiment_id,
                node.family,
                node.status.value,
                node.highest_fidelity.value if node.highest_fidelity else "—",
                (
                    "%.8f" % node.metric_set.primary_score
                    if node.metric_set is not None
                    else "—"
                ),
                node.latest_commit_sha or "—",
            )
        )
    totals = state.resource_totals
    lines.extend(
        (
            "",
            "## Resource totals",
            "",
            "- Provider tokens: %d" % totals.provider_tokens,
            "- Estimated tokens: %d" % totals.estimated_tokens,
            "- Agent elapsed wall-clock: %.3f seconds" % state.elapsed_wall_time_seconds,
            "- Action CPU time: %.3f seconds" % (totals.cpu_time_ms / 1000.0),
            "- GPU-hours: %.6f" % totals.gpu_hours,
            "- Manual interventions: %d" % state.manual_intervention_count,
            "",
            "Ledger head: `%s` / `%s`" % (state.last_event_id, state.last_event_hash),
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def rebuild_views(run_directory: Path, events: Sequence[Event]) -> None:
    _atomic_write(run_directory / "STATUS.md", render_status(events))
    _atomic_write(run_directory / "LESSONS.md", render_lessons(events))
    _atomic_write(run_directory / "SUMMARY.md", render_summary(events))
