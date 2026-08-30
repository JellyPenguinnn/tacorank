"""Deterministic run-memory and report projections."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Dict, Mapping, Sequence

from tacorank.evaluation.comparisons import normalized_headroom
from tacorank.evaluation.types import EvaluationResult

from ..memory.projections import project
from ..memory.retrieval import experiment_events
from ..orchestrator.state import ExperimentStatus
from ..run_layout import RunLayout
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


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _state_payload(events: Sequence[Event]) -> dict:
    state = project(events)
    active_jobs = []
    node = state.current_experiment
    terminal = {
        ExperimentStatus.ACCEPTED,
        ExperimentStatus.REJECTED,
        ExperimentStatus.PRUNED,
        ExperimentStatus.INVALID,
    }
    if (
        node is not None
        and node.status not in terminal
        and state.active_attempt is not None
        and state.status.value == "running"
    ):
        active_jobs.append(
            {
                # Phase A remains sequential.  This deterministic projection is
                # replaced by ledger-owned job IDs when parallel scheduling lands.
                "job_id": "job_%s_attempt_%03d"
                % (node.experiment_id, state.active_attempt),
                "experiment_id": node.experiment_id,
                "attempt": state.active_attempt,
                "phase": state.phase,
                "fidelity": (
                    state.active_fidelity.value if state.active_fidelity else None
                ),
                "worker": None,
                "identity_source": "derived_sequential",
            }
        )
    totals = state.resource_totals
    return {
        "schema_version": "1.0",
        "run_id": state.run_id,
        "execution_mode": "sequential",
        "derived_from": {
            "event_id": state.last_event_id,
            "event_hash": state.last_event_hash,
        },
        "global": {
            "status": state.status.value,
            "phase": state.phase,
            "best_experiment_id": state.best_experiment_id,
            "best_commit_sha": state.best_commit_sha,
            "best_primary_score": state.best_primary_score,
            "baseline_primary_score": state.baseline_primary_score,
            "experiments_proposed": state.experiments_proposed,
            "remaining_iterations": state.remaining_experiments,
            "full_evaluations_completed": state.full_evaluations_completed,
            "public_validation_queries": state.public_validation_queries,
            "manual_interventions": state.manual_intervention_count,
            "stop_reason_code": state.stop_reason_code,
            "final_experiment_id": state.final_experiment_id,
        },
        "active_jobs": active_jobs,
        "resources": {
            "provider_tokens": totals.provider_tokens,
            "estimated_tokens": totals.estimated_tokens,
            "unmeasured_tokens": totals.unmeasured_tokens,
            "total_reported_tokens": totals.total_reported_tokens,
            "cpu_time_ms": totals.cpu_time_ms,
            "gpu_hours": totals.gpu_hours,
            "elapsed_wall_time_seconds": state.elapsed_wall_time_seconds,
        },
    }


def render_status(events: Sequence[Event]) -> str:
    return "# TacoRank status\n\n```json\n%s\n```\n" % json.dumps(
        _state_payload(events), ensure_ascii=False, sort_keys=True, indent=2
    )


def render_lessons(events: Sequence[Event]) -> str:
    lessons = _lesson_records(events)
    lines = [_GENERATED_MARKER, "", "# Lesson memory", ""]
    if not lessons:
        lines.append("No lessons recorded.")
        return "\n".join(lines).rstrip() + "\n"
    lines.extend(
        (
            "| Lesson | Status | Origin | Category | Confidence |",
            "| --- | --- | --- | --- | ---: |",
        )
    )
    for lesson_id in sorted(lessons):
        record = lessons[lesson_id]
        candidate = record["candidate"]
        lines.append(
            "| [%s](%s.md) | %s | %s | %s | %.2f |"
            % (
                lesson_id,
                lesson_id,
                record["status"],
                candidate.origin.value,
                candidate.category.value,
                candidate.confidence,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


_GENERATED_MARKER = "<!-- generated by TacoRank; do not edit -->"


def _lesson_records(events: Sequence[Event]) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    for event in events:
        if event.payload.type == "lesson.recorded":
            records[event.payload.lesson_id] = {
                "candidate": event.payload.candidate,
                "status": LessonStatus.ACTIVE.value,
                "recorded_event_id": event.event_id,
                "status_event_id": None,
                "status_reason": None,
            }
        elif event.payload.type == "lesson.status_changed":
            record = records.get(event.payload.lesson_id)
            if record is not None:
                record["status"] = event.payload.status.value
                record["status_event_id"] = event.event_id
                record["status_reason"] = event.payload.reason
    return records


def _render_lesson(lesson_id: str, record: dict) -> str:
    candidate = record["candidate"]
    sources = ", ".join("`%s`" % item for item in candidate.source_event_ids)
    commits = ", ".join("`%s`" % item for item in candidate.source_commit_shas)
    lines = [
        _GENERATED_MARKER,
        "",
        "# %s" % lesson_id,
        "",
        "- Status: `%s`" % record["status"],
        "- Origin: `%s`" % candidate.origin.value,
        "- Category: `%s`" % candidate.category.value,
        "- Confidence: `%.2f`" % candidate.confidence,
        "- Tags: %s" % (", ".join(candidate.tags) if candidate.tags else "none"),
        "- Recorded by: `%s`" % record["recorded_event_id"],
        "",
        "## Finding",
        "",
        candidate.summary,
        "",
        "## Applies when",
        "",
        candidate.applicability,
    ]
    if candidate.avoid_when:
        lines.extend(("", "## Avoid when", "", candidate.avoid_when))
    lines.extend(
        (
            "",
            "## Evidence",
            "",
            "- Events: %s" % (sources or "none"),
            "- Commits: %s" % (commits or "none"),
        )
    )
    if record["status_event_id"]:
        lines.extend(
            (
                "",
                "## Latest status change",
                "",
                "- Event: `%s`" % record["status_event_id"],
                "- Reason: %s" % record["status_reason"],
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_summary(events: Sequence[Event]) -> str:
    state = project(events)
    adapter_failures = [
        event.payload.result
        for event in events
        if event.payload.type == "adapter.failed"
    ]
    failure_stages = Counter(result.failure_stage for result in adapter_failures)
    failure_text = ", ".join(
        "%s %d" % item for item in sorted(failure_stages.items())
    ) or "none"
    lines = [
        _GENERATED_MARKER,
        "",
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
            "- Adapter failures: %d (%s)" % (len(adapter_failures), failure_text),
            "",
            "Ledger head: `%s` / `%s`" % (state.last_event_id, state.last_event_hash),
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def render_resources(events: Sequence[Event]) -> str:
    state = project(events)
    totals = state.resource_totals
    return "\n".join(
        (
            _GENERATED_MARKER,
            "",
            "# TacoRank resource report",
            "",
            "- Provider tokens: %d" % totals.provider_tokens,
            "- Estimated tokens: %d" % totals.estimated_tokens,
            "- Unmeasured tokens: %d" % totals.unmeasured_tokens,
            "- Total reported tokens: %d" % totals.total_reported_tokens,
            "- Agent elapsed wall-clock: %.3f seconds"
            % state.elapsed_wall_time_seconds,
            "- Action CPU time: %.3f seconds" % (totals.cpu_time_ms / 1000.0),
            "- GPU-hours: %.6f" % totals.gpu_hours,
            "- Manual interventions: %d" % state.manual_intervention_count,
            "",
            "Ledger head: `%s` / `%s`"
            % (state.last_event_id, state.last_event_hash),
        )
    ).rstrip() + "\n"


def _direction_directories(families: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    used: Dict[str, str] = {}
    for family in sorted(set(families)):
        base = re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-") or "other"
        base = base[:64].rstrip("-") or "other"
        directory = base
        previous = used.get(directory)
        if previous is not None and previous != family:
            digest = hashlib.sha256(family.encode("utf-8")).hexdigest()[:8]
            directory = "%s-%s" % (base[:55].rstrip("-"), digest)
        used[directory] = family
        result[family] = directory
    return result


def _graph_payload(events: Sequence[Event]) -> dict:
    state = project(events)
    proposal_events = [
        event for event in events if event.payload.type == "experiment.proposed"
    ]
    specifications = {
        event.payload.spec.experiment_id: event.payload.spec
        for event in proposal_events
    }
    directions = _direction_directories(
        [spec.family for spec in specifications.values()]
    )
    nodes = []
    baseline_event = next(
        (event for event in events if event.payload.type == "baseline.verified"),
        None,
    )
    if baseline_event is not None:
        payload = baseline_event.payload
        nodes.append(
            {
                "experiment_id": payload.experiment_id,
                "node_type": "baseline",
                "parent_experiment_id": None,
                "direction": None,
                "direction_directory": None,
                "family": None,
                "hypothesis": "Frozen verified baseline",
                "method_card_ids": [],
                "base_commit_sha": None,
                "latest_commit_sha": payload.commit_sha,
                "status": "accepted",
                "highest_fidelity": payload.evaluation.fidelity.value,
                "metric_set": payload.metric_set.model_dump(mode="json"),
                "trust": payload.evaluation.trust.model_dump(mode="json"),
                "diagnostic_metrics": dict(payload.evaluation.diagnostic_metrics),
                "adapter_failures": [],
                "recovery_decisions": [],
                "estimated_cost": None,
                "best_eligible": state.best_experiment_id == payload.experiment_id,
                "event_ids": [
                    event.event_id
                    for event in experiment_events(events, payload.experiment_id)
                ],
            }
        )
    for experiment_id in sorted(specifications):
        spec = specifications[experiment_id]
        node = state.experiments[experiment_id]
        experiment_evaluations = [
            event.payload.result
            for event in events
            if event.payload.type == "evaluation.completed"
            and event.payload.result.experiment_id == experiment_id
        ]
        latest_evaluation = (
            experiment_evaluations[-1] if experiment_evaluations else None
        )
        adapter_failures = [
            {
                "event_id": event.event_id,
                "attempt": event.payload.result.attempt,
                "failure_stage": event.payload.result.failure_stage,
                "error_class": event.payload.result.error_class,
                "error_fingerprint": event.payload.result.error_fingerprint,
                "error_summary": event.payload.result.error_summary,
                "diagnostic_artifacts": [
                    artifact.model_dump(mode="json")
                    for artifact in event.payload.result.diagnostic_artifacts
                ],
                "resource_delta": event.resource_delta.model_dump(mode="json"),
            }
            for event in events
            if event.payload.type == "adapter.failed"
            and event.payload.result.experiment_id == experiment_id
        ]
        recovery_decisions = [
            {
                "event_id": event.event_id,
                "failure_event_id": event.payload.decision.failure_event_id,
                "action": event.payload.decision.action.value,
                "reason_code": event.payload.decision.reason_code,
                "instructions": event.payload.decision.instructions,
            }
            for event in events
            if event.payload.type == "recovery.decided"
            and event.payload.decision.experiment_id == experiment_id
        ]
        nodes.append(
            {
                "experiment_id": experiment_id,
                "node_type": "experiment",
                "parent_experiment_id": spec.parent_experiment_id,
                "direction": spec.family,
                "direction_directory": directions[spec.family],
                "family": spec.family,
                "hypothesis": spec.hypothesis,
                "change_summary": spec.change_summary,
                "expected_mechanism": spec.expected_mechanism,
                "success_criteria": spec.success_criteria,
                "falsification_condition": spec.falsification_condition,
                "method_card_ids": list(spec.method_card_ids),
                "base_commit_sha": node.base_commit_sha,
                "latest_commit_sha": node.latest_commit_sha,
                "status": node.status.value,
                "highest_fidelity": (
                    node.highest_fidelity.value if node.highest_fidelity else None
                ),
                "metric_set": (
                    node.metric_set.model_dump(mode="json")
                    if node.metric_set is not None
                    else None
                ),
                "trust": (
                    node.trust.model_dump(mode="json")
                    if node.trust is not None
                    else None
                ),
                "diagnostic_metrics": (
                    dict(latest_evaluation.diagnostic_metrics)
                    if latest_evaluation is not None
                    else {}
                ),
                "adapter_failures": adapter_failures,
                "recovery_decisions": recovery_decisions,
                "estimated_cost": spec.estimated_cost.model_dump(mode="json"),
                "best_eligible": node.best_eligible,
                "event_ids": [
                    event.event_id
                    for event in experiment_events(events, experiment_id)
                ],
            }
        )
    return {
        "schema_version": "1.0",
        "run_id": state.run_id,
        "derived_from": {
            "event_id": state.last_event_id,
            "event_hash": state.last_event_hash,
        },
        "nodes": nodes,
        "edges": [
            {
                "parent": node["parent_experiment_id"],
                "child": node["experiment_id"],
            }
            for node in nodes
            if node["parent_experiment_id"] is not None
        ],
        "directions": [
            {
                "direction": family,
                "directory": directory,
                "experiment_ids": [
                    node["experiment_id"]
                    for node in nodes
                    if node["direction"] == family
                ],
            }
            for family, directory in sorted(directions.items())
        ],
    }


def _render_graph(payload: dict) -> str:
    lines = [
        _GENERATED_MARKER,
        "",
        "# Experiment graph",
        "",
        "| Experiment | Parent | Direction | Status | Fidelity | Primary |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for node in payload["nodes"]:
        metric_set = node["metric_set"]
        score = metric_set["primary_score"] if metric_set is not None else None
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                node["experiment_id"],
                node["parent_experiment_id"] or "—",
                node["direction"] or "—",
                node["status"],
                node["highest_fidelity"] or "—",
                "%.8f" % score if score is not None else "—",
            )
        )
    lines.extend(("", "## Edges", ""))
    if not payload["edges"]:
        lines.append("No experiment edges recorded.")
    else:
        lines.extend(
            "- `%s` → `%s`" % (edge["parent"], edge["child"])
            for edge in payload["edges"]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_experiment(node: dict, events: Sequence[Event]) -> str:
    metric_set = node["metric_set"]
    lines = [
        _GENERATED_MARKER,
        "",
        "# %s" % node["experiment_id"],
        "",
        "- Parent: `%s`" % (node["parent_experiment_id"] or "none"),
        "- Direction: `%s`" % (node["direction"] or "baseline"),
        "- Status: `%s`" % node["status"],
        "- Base commit: `%s`" % (node["base_commit_sha"] or "none"),
        "- Latest commit: `%s`" % (node["latest_commit_sha"] or "none"),
        "- Method cards: %s"
        % (
            ", ".join("`%s`" % item for item in node["method_card_ids"])
            or "none"
        ),
        "",
        "## Hypothesis",
        "",
        node["hypothesis"],
    ]
    if node.get("expected_mechanism"):
        lines.extend(("", "## Expected mechanism", "", node["expected_mechanism"]))
    if metric_set is not None:
        lines.extend(
            (
                "",
                "## Result",
                "",
                "Primary: `%.8f`" % metric_set["primary_score"],
                "",
                "```json",
                json.dumps(metric_set, ensure_ascii=False, sort_keys=True, indent=2),
                "```",
            )
        )
    if node["diagnostic_metrics"]:
        lines.extend(
            (
                "",
                "## Label-free candidate diagnostics",
                "",
                "```json",
                json.dumps(
                    node["diagnostic_metrics"],
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ),
                "```",
            )
        )
    if node["adapter_failures"] or node["recovery_decisions"]:
        lines.extend(
            (
                "",
                "## Adapter failures and recovery",
                "",
                "```json",
                json.dumps(
                    {
                        "adapter_failures": node["adapter_failures"],
                        "recovery_decisions": node["recovery_decisions"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ),
                "```",
            )
        )
    event_by_id = {event.event_id: event for event in events}
    lines.extend(("", "## Lifecycle", ""))
    for event_id in node["event_ids"]:
        event = event_by_id[event_id]
        lines.append(
            "- `%s` — `%s` — %s"
            % (event.event_id, event.event_type.value, event.timestamp.isoformat())
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_direction(direction: dict, node_by_id: Mapping[str, dict]) -> str:
    methods = sorted(
        {
            method
            for experiment_id in direction["experiment_ids"]
            for method in node_by_id[experiment_id]["method_card_ids"]
        }
    )
    lines = [
        _GENERATED_MARKER,
        "",
        "# Direction: %s" % direction["direction"],
        "",
        "Methods: %s" % (", ".join("`%s`" % item for item in methods) or "none"),
        "",
        "| Experiment | Parent | Status | Primary |",
        "| --- | --- | --- | ---: |",
    ]
    for experiment_id in direction["experiment_ids"]:
        node = node_by_id[experiment_id]
        metric_set = node["metric_set"]
        score = metric_set["primary_score"] if metric_set is not None else None
        lines.append(
            "| [%s](experiments/%s.md) | %s | %s | %s |"
            % (
                experiment_id,
                experiment_id,
                node["parent_experiment_id"] or "—",
                node["status"],
                "%.8f" % score if score is not None else "—",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _prepare_generated_directory(path: Path) -> None:
    marker = path / ".tacorank-derived"
    if path.exists():
        if any(path.iterdir()) and not marker.is_file():
            raise RuntimeError(
                "refusing to replace an unmarked derived-view directory: %s" % path
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write(marker, "Generated by TacoRank; safe to rebuild.\n")


def _layout_for_run_directory(run_directory: Path) -> RunLayout:
    resolved = Path(run_directory).resolve()
    layout = RunLayout(resolved.parent.parent, resolved.name)
    if layout.run_directory != resolved or resolved.parent.name != "runs":
        raise ValueError("run_directory must be repository_root/runs/<run_id>")
    return layout


def rebuild_operational_state(run_directory: Path, events: Sequence[Event]) -> None:
    """Atomically refresh the overwriteable current-state projections."""

    layout = _layout_for_run_directory(run_directory)
    state = project(events)
    if state.run_id is not None and layout.run_id != state.run_id:
        raise ValueError("run directory does not match projected run_id")
    _atomic_write_json(layout.state, _state_payload(events))
    _atomic_write(layout.status, render_status(events))


def rebuild_views(run_directory: Path, events: Sequence[Event]) -> None:
    """Rebuild every non-authoritative view from the ledger."""

    layout = _layout_for_run_directory(run_directory)
    rebuild_operational_state(layout.run_directory, events)

    _prepare_generated_directory(layout.lessons)
    lessons = _lesson_records(events)
    _atomic_write(layout.lessons / "INDEX.md", render_lessons(events))
    for lesson_id, record in sorted(lessons.items()):
        _atomic_write(
            layout.lessons / (lesson_id + ".md"),
            _render_lesson(lesson_id, record),
        )

    _prepare_generated_directory(layout.experiment_graph)
    graph = _graph_payload(events)
    _atomic_write_json(layout.experiment_graph / "graph.json", graph)
    _atomic_write(layout.experiment_graph / "GRAPH.md", _render_graph(graph))
    directions_root = layout.experiment_graph / "directions"
    directions_root.mkdir(parents=True, exist_ok=True)
    node_by_id = {node["experiment_id"]: node for node in graph["nodes"]}
    for direction in graph["directions"]:
        directory = directions_root / direction["directory"]
        experiments_directory = directory / "experiments"
        experiments_directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(directory / "README.md", _render_direction(direction, node_by_id))
        for experiment_id in direction["experiment_ids"]:
            _atomic_write(
                experiments_directory / (experiment_id + ".md"),
                _render_experiment(node_by_id[experiment_id], events),
            )

    _prepare_generated_directory(layout.reports)
    _atomic_write(layout.reports / "SUMMARY.md", render_summary(events))
    _atomic_write(layout.reports / "RESOURCES.md", render_resources(events))
