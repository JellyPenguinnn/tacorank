"""Judge-facing deterministic Markdown result projections."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from tacorank.evaluation.comparisons import normalized_headroom
from tacorank.evaluation.types import EvaluationResult

from ..memory.projections import project
from ..schemas import Event, LessonStatus
from .resources import ResourceSummary


def render_metric_table(
    named_results: Mapping[str, EvaluationResult],
) -> str:
    if not named_results:
        return ""
    metric_names = sorted(next(iter(named_results.values())).metric_set.metrics)
    header = "| Candidate | " + " | ".join(metric_names) + " | primary |"
    rule = "| --- | " + " | ".join("---:" for _ in metric_names) + " | ---: |"
    rows = [header, rule]
    for name, result in named_results.items():
        values = ["%.6f" % result.metric_set.metrics[metric] for metric in metric_names]
        rows.append(
            "| %s | %s | %.6f |"
            % (name, " | ".join(values), result.metric_set.primary_score)
        )
    return "\n".join(rows)


def render_evaluation_summary(
    run_id: str,
    final_result: EvaluationResult,
    baseline_primary: float,
    oracle_primary: float,
    resources: ResourceSummary,
    verdicts: Sequence[str],
    experiments_used: int,
    experiment_limit: int,
    public_queries: int,
    limitations: Sequence[str],
) -> str:
    score = final_result.metric_set.primary_score
    headroom = 100.0 * normalized_headroom(score, baseline_primary, oracle_primary)
    census = Counter(verdicts)
    census_text = " | ".join(
        "%s %d" % item for item in sorted(census.items())
    ) or "none"
    metric_text = " | ".join(
        "%s %.6f" % item for item in sorted(final_result.metric_set.metrics.items())
    )
    limitation_text = "\n".join("- %s" % value for value in limitations) or "- None recorded."
    return "\n".join(
        [
            "# Run Summary - %s" % run_id,
            "",
            "## Result",
            "",
            "primary %.6f (baseline %.6f, delta %+.6f, headroom captured %.2f%%)"
            % (score, baseline_primary, score - baseline_primary, headroom),
            metric_text,
            "",
            "## Resource",
            "",
            "provider tokens: %d in / %d out; estimated tokens: %d in / %d out"
            % (
                resources.llm_input_tokens_provider,
                resources.llm_output_tokens_provider,
                resources.llm_input_tokens_estimated,
                resources.llm_output_tokens_estimated,
            ),
            "action wall time: %.1fs; GPU-hours: %.4f; manual interventions: %d"
            % (
                resources.action_wall_time_ms / 1000.0,
                resources.gpu_hours,
                resources.manual_interventions,
            ),
            "experiments: %d/%d; public validation queries: %d"
            % (experiments_used, experiment_limit, public_queries),
            "",
            "## Verdict Census",
            "",
            census_text,
            "",
            "## Limitations",
            "",
            limitation_text,
        ]
    )


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
            "- Unmeasured tokens: %d" % totals.unmeasured_tokens,
            "- Total reported tokens: %d" % totals.total_reported_tokens,
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
