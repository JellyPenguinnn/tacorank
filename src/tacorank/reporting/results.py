"""Deterministic run-memory and report projections."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Dict, Mapping, Optional, Sequence

from tacorank.evaluation.comparisons import normalized_headroom
from tacorank.evaluation.types import EvaluationResult

from ..memory.projections import project
from ..memory.retrieval import experiment_events
from ..orchestrator.state import ExperimentStatus
from ..run_layout import RunLayout
from ..schemas import (
    Event,
    ExperimentDecisionKind,
    LessonStatus,
    RecoveryAction,
    RunOutcome,
    TrustVerdict,
)
from .resources import ResourceSummary


def _reported_primary_score(node: object) -> Optional[float]:
    trust = getattr(node, "trust", None)
    seed_mean = getattr(trust, "seed_mean", None)
    if seed_mean is not None:
        return float(seed_mean)
    metric_set = getattr(node, "metric_set", None)
    return None if metric_set is None else float(metric_set.primary_score)


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


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _phase_after_event(event: Event, previous: str) -> str:
    """Return the controller phase after one durable event."""

    payload = event.payload
    event_type = payload.type
    if event_type == "run.started":
        return "contract_verification"
    if event_type == "contract.verified":
        return "baseline"
    if event_type == "baseline.verified":
        return "planning"
    if event_type == "context.created":
        return "%s_context" % payload.context.role
    if event_type == "experiment.proposed":
        return "coding"
    if event_type == "patch.created":
        return "patch_gate"
    if event_type == "patch.checked":
        return "execution" if payload.result.accepted else "recovery"
    if event_type == "execution.started":
        return "running"
    if event_type == "execution.finished":
        return (
            "output_gate"
            if payload.result.outcome == RunOutcome.SUCCESS
            else "recovery"
        )
    if event_type == "adapter.failed":
        return "recovery"
    if event_type == "recovery.decided":
        if payload.decision.action in (
            RecoveryAction.ABANDON,
            RecoveryAction.ROLLBACK,
        ):
            return "planning"
        if payload.decision.action in (
            RecoveryAction.RETRY_SAME_COMMIT,
            RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING,
        ):
            return "execution"
        return "recovery"
    if event_type == "output.checked":
        return "evaluation" if payload.result.accepted else "recovery"
    if event_type == "evaluation.completed":
        return (
            "recovery"
            if payload.result.trust.verdict == TrustVerdict.NO_OP
            else "decision"
        )
    if event_type == "experiment.decided":
        return (
            "execution"
            if payload.decision.decision == ExperimentDecisionKind.PROMOTE
            else "planning"
        )
    if event_type == "run.stopped":
        return "stopped"
    if event_type == "final.selected":
        return "submission"
    if event_type == "submission.checked":
        return "finalized" if payload.accepted else "failed"
    return previous


def _configured_stage_timeout(
    events: Sequence[Event], phase: str, experiment_id: Optional[str]
) -> Optional[int]:
    if phase == "coder_context":
        for event in reversed(events):
            if event.payload.type != "context.created":
                continue
            context = event.payload.context
            if context.role == "coder" and context.experiment_id == experiment_id:
                return context.wall_time_limit_seconds
    if phase == "running":
        for event in reversed(events):
            if event.payload.type != "execution.started":
                continue
            if event.payload.request.experiment_id == experiment_id:
                return event.payload.request.timeout_seconds
    return None


def runtime_status(events: Sequence[Event]) -> dict:
    """Project live-monitoring anchors without consulting wall-clock time."""

    state = project(events)
    phase = "not_started"
    stage_started_at = None
    for event in events:
        next_phase = _phase_after_event(event, phase)
        if next_phase != phase:
            stage_started_at = event.timestamp
        phase = next_phase
    if phase != state.phase:
        raise ValueError("runtime phase projection disagrees with run state")
    timeout_seconds = _configured_stage_timeout(
        events, state.phase, state.active_experiment_id
    )
    deadline = (
        stage_started_at + timedelta(seconds=timeout_seconds)
        if stage_started_at is not None and timeout_seconds is not None
        else None
    )
    last_event = events[-1] if events else None
    elapsed_at_head = (
        max(0.0, (last_event.timestamp - stage_started_at).total_seconds())
        if last_event is not None and stage_started_at is not None
        else 0.0
    )
    return {
        "experiment_id": state.active_experiment_id,
        "phase": state.phase,
        "attempt": state.active_attempt,
        "fidelity": (
            state.active_fidelity.value if state.active_fidelity is not None else None
        ),
        "stage_started_at": (
            _timestamp(stage_started_at) if stage_started_at is not None else None
        ),
        "stage_elapsed_seconds_at_ledger_head": elapsed_at_head,
        "configured_timeout_seconds": timeout_seconds,
        "estimated_deadline": _timestamp(deadline) if deadline is not None else None,
        "last_event_id": last_event.event_id if last_event is not None else None,
        "last_event_type": (
            last_event.event_type.value if last_event is not None else None
        ),
        "last_event_at": (
            _timestamp(last_event.timestamp) if last_event is not None else None
        ),
    }


def _duration_seconds(start: Event, finish: Event) -> float:
    return max(0.0, (finish.timestamp - start.timestamp).total_seconds())


def experiment_timing(events: Sequence[Event], experiment_id: str) -> dict:
    """Return exact ledger-bound lifecycle durations for one experiment."""

    selected = experiment_events(events, experiment_id)
    proposal = next(
        (event for event in selected if event.payload.type == "experiment.proposed"),
        None,
    )
    terminal = None
    for event in selected:
        if event.payload.type == "experiment.decided" and (
            event.payload.decision.decision != ExperimentDecisionKind.PROMOTE
        ):
            terminal = event
            break
        if event.payload.type == "recovery.decided" and (
            event.payload.decision.action
            in (RecoveryAction.ABANDON, RecoveryAction.ROLLBACK)
        ):
            terminal = event
            break

    coding_seconds = 0.0
    coding_start = None
    execution_seconds = 0.0
    execution_start = None
    event_by_id = {event.event_id: event for event in selected}
    recovery_seconds = 0.0
    for event in selected:
        payload = event.payload
        if payload.type == "context.created" and payload.context.role == "coder":
            coding_start = event
        elif payload.type == "recovery.decided" and (
            payload.decision.action == RecoveryAction.TRAE_REPAIR
        ):
            coding_start = event
        elif payload.type == "patch.created" and coding_start is not None:
            coding_seconds += _duration_seconds(coding_start, event)
            coding_start = None
        elif payload.type == "adapter.failed" and (
            payload.result.failure_stage == "coding" and coding_start is not None
        ):
            coding_seconds += _duration_seconds(coding_start, event)
            coding_start = None

        if payload.type == "execution.started":
            execution_start = event
        elif payload.type == "execution.finished" and execution_start is not None:
            execution_seconds += _duration_seconds(execution_start, event)
            execution_start = None
        elif payload.type == "adapter.failed" and (
            payload.result.failure_stage == "execution"
            and execution_start is not None
        ):
            execution_seconds += _duration_seconds(execution_start, event)
            execution_start = None

        if payload.type == "recovery.decided":
            failure = event_by_id.get(payload.decision.failure_event_id)
            if failure is not None:
                recovery_seconds += _duration_seconds(failure, event)

    return {
        "proposed_at": _timestamp(proposal.timestamp) if proposal else None,
        "terminal_at": _timestamp(terminal.timestamp) if terminal else None,
        "terminal_event_id": terminal.event_id if terminal else None,
        "loop_time_seconds": (
            _duration_seconds(proposal, terminal)
            if proposal is not None and terminal is not None
            else None
        ),
        "trae_coding_time_seconds": coding_seconds,
        "execution_time_seconds": execution_seconds,
        "recovery_time_seconds": recovery_seconds,
    }


def _state_payload(events: Sequence[Event]) -> dict:
    state = project(events)
    runtime = runtime_status(events)
    active_jobs = []
    node = state.current_experiment
    terminal = {
        ExperimentStatus.ACCEPTED,
        ExperimentStatus.REJECTED,
        ExperimentStatus.PRUNED,
        ExperimentStatus.RETAINED,
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
                "stage_started_at": runtime["stage_started_at"],
                "configured_timeout_seconds": runtime[
                    "configured_timeout_seconds"
                ],
                "estimated_deadline": runtime["estimated_deadline"],
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
        "current": runtime,
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
        "experiment_timings": {
            experiment_id: experiment_timing(events, experiment_id)
            for experiment_id in sorted(state.experiments)
        },
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
    ]
    if candidate.measured_under_frame_experiment_id:
        lines.append(
            "- Evaluation frame: `%s`"
            % candidate.measured_under_frame_experiment_id
        )
    lines.extend(
        [
            "",
            "## Finding",
            "",
            candidate.summary,
            "",
            "## Applies when",
            "",
            candidate.applicability,
        ]
    )
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
    specifications = {
        event.payload.spec.experiment_id: event.payload.spec
        for event in events
        if event.payload.type == "experiment.proposed"
    }
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
        "| Experiment | Family | Campaign variant | Status | Fidelity | Primary | Loop time | Trae coding | Execution | Recovery | Commit |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for node in sorted(state.experiments.values(), key=lambda item: item.experiment_id):
        timing = experiment_timing(events, node.experiment_id)
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %.3fs | %.3fs | %.3fs | %s |"
            % (
                node.experiment_id,
                node.family,
                (
                    specifications[node.experiment_id].variant_id
                    if node.experiment_id in specifications
                    and specifications[node.experiment_id].variant_id
                    else "—"
                ),
                node.status.value,
                node.highest_fidelity.value if node.highest_fidelity else "—",
                (
                    "%.8f" % _reported_primary_score(node)
                    if _reported_primary_score(node) is not None
                    else "—"
                ),
                (
                    "%.3fs" % timing["loop_time_seconds"]
                    if timing["loop_time_seconds"] is not None
                    else "—"
                ),
                timing["trae_coding_time_seconds"],
                timing["execution_time_seconds"],
                timing["recovery_time_seconds"],
                node.latest_commit_sha or "—",
            )
        )
    campaign_specs = [
        spec for spec in specifications.values() if spec.campaign_id is not None
    ]
    if campaign_specs:
        family_order = list(dict.fromkeys(spec.family for spec in campaign_specs))
        lines.extend(
            (
                "",
                "## Campaign comparison",
                "",
                "| Family | Attempted | Full evaluations | Best experiment | Best primary | Baseline delta |",
                "| --- | ---: | ---: | --- | ---: | ---: |",
            )
        )
        for family in family_order:
            family_nodes = [
                state.experiments[spec.experiment_id]
                for spec in campaign_specs
                if spec.family == family and spec.experiment_id in state.experiments
            ]
            full_nodes = [
                node
                for node in family_nodes
                if node.highest_fidelity is not None
                and node.highest_fidelity.value == "full"
                and node.metric_set is not None
            ]
            best = max(
                full_nodes,
                key=lambda node: _reported_primary_score(node),
                default=None,
            )
            best_score = _reported_primary_score(best) if best is not None else None
            baseline_delta = (
                best_score - state.baseline_primary_score
                if best_score is not None and state.baseline_primary_score is not None
                else None
            )
            lines.append(
                "| %s | %d | %d | %s | %s | %s |"
                % (
                    family,
                    len(family_nodes),
                    len(full_nodes),
                    best.experiment_id if best is not None else "—",
                    "%.8f" % best_score if best_score is not None else "—",
                    "%+.8f" % baseline_delta if baseline_delta is not None else "—",
                )
            )
    latest_evaluations = {}
    for event in events:
        if event.payload.type == "evaluation.completed":
            latest_evaluations[event.payload.result.experiment_id] = event.payload.result
    diagnostic_rows = []
    for experiment_id, result in sorted(latest_evaluations.items()):
        hypotheses = result.diagnostics.failure_hypotheses
        if hypotheses:
            diagnostic_rows.append(
                "- `%s`: %s" % (experiment_id, " ".join(hypotheses))
            )
    if diagnostic_rows:
        lines.extend(("", "## Diagnostic findings", "", *diagnostic_rows))
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
    latest_evaluations = {}
    for event in events:
        if event.payload.type == "evaluation.completed":
            latest_evaluations[event.payload.result.experiment_id] = event.payload.result
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
                "primary_score": payload.metric_set.primary_score,
                "trust": payload.evaluation.trust.model_dump(mode="json"),
                "diagnostics": payload.evaluation.diagnostics.model_dump(mode="json"),
                "metrics_artifact": (
                    payload.evaluation.metrics_artifact.model_dump(mode="json")
                    if payload.evaluation.metrics_artifact is not None
                    else None
                ),
                "diagnostic_metrics": dict(payload.evaluation.diagnostic_metrics),
                "adapter_failures": [],
                "recovery_decisions": [],
                "estimated_cost": None,
                "timing": None,
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
        latest_evaluation = latest_evaluations.get(experiment_id)
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
                "campaign_id": spec.campaign_id,
                "variant_id": spec.variant_id,
                "variant_instruction": spec.variant_instruction,
                "variant_parameters": dict(spec.variant_parameters),
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
                "primary_score": _reported_primary_score(node),
                "trust": (
                    node.trust.model_dump(mode="json")
                    if node.trust is not None
                    else None
                ),
                "diagnostics": (
                    latest_evaluation.diagnostics.model_dump(mode="json")
                    if latest_evaluation is not None
                    else None
                ),
                "metrics_artifact": (
                    latest_evaluation.metrics_artifact.model_dump(mode="json")
                    if latest_evaluation is not None
                    and latest_evaluation.metrics_artifact is not None
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
                "timing": experiment_timing(events, experiment_id),
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
        score = node.get("primary_score")
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
        "- Campaign: `%s`" % (node.get("campaign_id") or "none"),
        "- Variant: `%s`" % (node.get("variant_id") or "none"),
        "",
        "## Hypothesis",
        "",
        node["hypothesis"],
    ]
    if node.get("expected_mechanism"):
        lines.extend(("", "## Expected mechanism", "", node["expected_mechanism"]))
    if node.get("variant_instruction"):
        lines.extend(
            ("", "## Campaign configuration", "", node["variant_instruction"])
        )
        lines.extend(
            (
                "",
                "```json",
                json.dumps(node["variant_parameters"], sort_keys=True),
                "```",
            )
        )
    if node.get("timing") is not None:
        lines.extend(
            (
                "",
                "## Timing",
                "",
                "```json",
                json.dumps(
                    node["timing"], ensure_ascii=False, sort_keys=True, indent=2
                ),
                "```",
            )
        )
    if metric_set is not None:
        lines.extend(
            (
                "",
                "## Result",
                "",
                "Primary: `%.8f`" % node["primary_score"],
                "",
                "```json",
                json.dumps(metric_set, ensure_ascii=False, sort_keys=True, indent=2),
                "```",
            )
        )
    diagnostics = node.get("diagnostics")
    if diagnostics is not None and any(
        value not in (None, [], {}) for value in diagnostics.values()
    ):
        lines.extend(
            (
                "",
                "## Diagnostics",
                "",
                "These aggregate signals support hypotheses, not causal claims.",
                "",
                "```json",
                json.dumps(diagnostics, ensure_ascii=False, sort_keys=True, indent=2),
                "```",
            )
        )
        if node.get("metrics_artifact") is not None:
            lines.extend(
                (
                    "",
                    "Metrics artifact: `%s`"
                    % node["metrics_artifact"]["path"],
                )
            )
    if node["diagnostic_metrics"]:
        lines.extend(
            (
                "",
                "## Compact diagnostic metrics",
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
        score = node.get("primary_score")
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
