"""Pure event reducers for run state, experiment graph, and lessons."""

from __future__ import annotations

from typing import Iterable

from ..accounting import aggregate_resources
from ..orchestrator.state import (
    ExperimentNode,
    ExperimentStatus,
    LessonView,
    RunState,
    RunStatus,
)
from ..orchestrator.convergence import is_finalizable_stop_reason
from ..schemas import (
    Event,
    EventType,
    ExperimentDecisionKind,
    Fidelity,
    LessonStatus,
    RecoveryAction,
    RunOutcome,
    TrustVerdict,
)


def _node(state: RunState, experiment_id: str) -> ExperimentNode:
    try:
        return state.experiments[experiment_id]
    except KeyError as exc:
        raise ValueError("unknown experiment %s" % experiment_id) from exc


def project(events: Iterable[Event]) -> RunState:
    state = RunState()
    materialized = list(events)
    convergence_incumbent = None

    for event in materialized:
        payload = event.payload
        state.run_id = event.run_id
        state.last_event_id = event.event_id
        state.last_event_hash = event.event_hash
        state.last_event_at = event.timestamp

        if event.event_type == EventType.RUN_STARTED:
            state.status = RunStatus.INITIALIZING
            state.phase = "contract_verification"
            state.started_at = event.timestamp
            state.max_experiments = payload.max_experiments
            state.parallel_directions = payload.parallel_directions
            state.synthesize_parallel_improvements = (
                payload.synthesize_parallel_improvements
            )
            state.wall_time_limit_seconds = payload.wall_time_limit_seconds
            state.token_limit = payload.token_limit
            state.gpu_seconds_limit = payload.gpu_seconds_limit
            state.max_repairs_per_experiment = payload.max_repairs_per_experiment
            state.max_confirmation_attempts = payload.max_confirmation_attempts
            state.seed_schedule = list(payload.seed_schedule)
            state.convergence_epsilon = payload.convergence_epsilon
            state.convergence_patience = payload.convergence_patience
        elif event.event_type == EventType.CONTRACT_VERIFIED:
            state.phase = "baseline"
        elif event.event_type == EventType.BASELINE_VERIFIED:
            state.status = RunStatus.READY
            state.phase = "planning"
            state.baseline_primary_score = payload.metric_set.primary_score
            convergence_incumbent = payload.metric_set.primary_score
            if state.best_primary_score is None:
                state.best_experiment_id = payload.experiment_id
                state.best_commit_sha = payload.commit_sha
                state.best_primary_score = payload.metric_set.primary_score
        elif event.event_type == EventType.CONTEXT_CREATED:
            state.phase = "%s_context" % payload.context.role
        elif event.event_type == EventType.EXPERIMENT_PROPOSED:
            spec = payload.spec
            state.experiments[spec.experiment_id] = ExperimentNode(
                experiment_id=spec.experiment_id,
                parent_experiment_id=spec.parent_experiment_id,
                hypothesis=spec.hypothesis,
                family=spec.family,
                base_commit_sha=spec.parent_commit_sha,
                duplicate_key=spec.duplicate_key,
                evidence_event_ids=list(spec.evidence_event_ids),
            )
            state.experiments_proposed += 1
            state.active_experiment_id = spec.experiment_id
            state.active_attempt = 1
            state.status = RunStatus.RUNNING
            state.phase = "coding"
        elif event.event_type == EventType.PATCH_CREATED:
            candidate = payload.candidate
            node = _node(state, candidate.experiment_id)
            node.latest_commit_sha = candidate.patch_commit_sha
            node.same_commit_retry_count = 0
            node.attempt_count = max(node.attempt_count, candidate.attempt)
            node.status = ExperimentStatus.PATCH_READY
            state.active_attempt = candidate.attempt
            state.phase = "patch_gate"
        elif event.event_type == EventType.PATCH_CHECKED:
            result = payload.result
            node = _node(state, result.experiment_id)
            node.status = (
                ExperimentStatus.READY_TO_RUN if result.accepted else ExperimentStatus.RECOVERING
            )
            state.phase = "execution" if result.accepted else "recovery"
        elif event.event_type == EventType.EXECUTION_STARTED:
            request = payload.request
            node = _node(state, request.experiment_id)
            node.status = ExperimentStatus.RUNNING
            node.attempt_count = max(node.attempt_count, request.attempt)
            state.active_attempt = request.attempt
            state.active_fidelity = request.fidelity
            state.phase = "running"
        elif event.event_type == EventType.EXECUTION_FINISHED:
            result = payload.result
            node = _node(state, result.experiment_id)
            if result.outcome == RunOutcome.SUCCESS:
                node.status = ExperimentStatus.OUTPUT_READY
                state.phase = "output_gate"
            else:
                node.status = ExperimentStatus.RECOVERING
                node.last_error_fingerprint = result.error_fingerprint
                state.phase = "recovery"
        elif event.event_type == EventType.ADAPTER_FAILED:
            result = payload.result
            node = _node(state, result.experiment_id)
            node.status = ExperimentStatus.RECOVERING
            node.last_error_fingerprint = result.error_fingerprint
            state.phase = "recovery"
        elif event.event_type == EventType.PLANNING_FAILED:
            state.active_experiment_id = None
            state.active_attempt = None
            state.active_fidelity = None
            state.phase = "planning_failure"
        elif event.event_type == EventType.RECOVERY_DECIDED:
            decision = payload.decision
            node = _node(state, decision.experiment_id)
            if decision.action == RecoveryAction.TRAE_REPAIR:
                node.repair_count = max(node.repair_count, decision.repair_attempt)
            if decision.action == RecoveryAction.RETRY_SAME_COMMIT:
                node.same_commit_retry_count += 1
            if decision.action in (
                RecoveryAction.ABANDON,
                RecoveryAction.ROLLBACK,
            ):
                node.status = ExperimentStatus.INVALID
                node.terminal_event_id = event.event_id
                state.active_experiment_id = None
                state.active_attempt = None
                state.active_fidelity = None
                state.phase = "planning"
            elif decision.action in (
                RecoveryAction.RETRY_SAME_COMMIT,
                RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING,
            ):
                node.status = ExperimentStatus.READY_TO_RUN
                state.phase = "execution"
            else:
                node.status = ExperimentStatus.RECOVERING
                state.phase = "recovery"
        elif event.event_type == EventType.OUTPUT_CHECKED:
            result = payload.result
            node = _node(state, result.experiment_id)
            node.status = (
                ExperimentStatus.OUTPUT_VERIFIED if result.accepted else ExperimentStatus.RECOVERING
            )
            state.phase = "evaluation" if result.accepted else "recovery"
        elif event.event_type == EventType.EVALUATION_COMPLETED:
            result = payload.result
            node = _node(state, result.experiment_id)
            node.status = (
                ExperimentStatus.RECOVERING
                if result.trust.verdict == TrustVerdict.NO_OP
                else ExperimentStatus.EVALUATED
            )
            node.metric_set = result.metric_set
            node.trust = result.trust
            node.highest_fidelity = result.fidelity
            state.phase = (
                "recovery"
                if result.trust.verdict == TrustVerdict.NO_OP
                else "decision"
            )
            if result.population.value == "public_validation":
                state.public_validation_queries += 1
            if result.fidelity == Fidelity.FULL:
                state.full_evaluations_completed += 1
        elif event.event_type == EventType.EXPERIMENT_DECIDED:
            decision = payload.decision
            node = _node(state, decision.experiment_id)
            status_by_decision = {
                ExperimentDecisionKind.ACCEPT: ExperimentStatus.ACCEPTED,
                ExperimentDecisionKind.REJECT: ExperimentStatus.REJECTED,
                ExperimentDecisionKind.PRUNE: ExperimentStatus.PRUNED,
                ExperimentDecisionKind.INVALID: ExperimentStatus.INVALID,
                ExperimentDecisionKind.PROMOTE: ExperimentStatus.READY_TO_RUN,
            }
            node.status = status_by_decision[decision.decision]
            node.best_eligible = decision.best_eligible
            if decision.decision == ExperimentDecisionKind.PROMOTE:
                state.active_fidelity = decision.next_fidelity
                if decision.next_fidelity == Fidelity.FULL and decision.fidelity_completed == Fidelity.FULL:
                    node.confirmation_count += 1
                state.phase = "execution"
            else:
                node.terminal_event_id = event.event_id
                state.active_experiment_id = None
                state.active_attempt = None
                state.active_fidelity = None
                state.phase = "planning"
                if (
                    decision.fidelity_completed == Fidelity.FULL
                    and node.metric_set is not None
                    and node.trust is not None
                    and node.trust.verdict == TrustVerdict.ACCEPTED
                ):
                    score = node.metric_set.primary_score
                    if (
                        convergence_incumbent is None
                        or score > convergence_incumbent + state.convergence_epsilon
                    ):
                        convergence_incumbent = score
                        state.consecutive_non_improving_full_evaluations = 0
                    else:
                        state.consecutive_non_improving_full_evaluations += 1
        elif event.event_type == EventType.BEST_UPDATED:
            state.best_experiment_id = payload.experiment_id
            state.best_commit_sha = payload.commit_sha
            state.best_primary_score = payload.primary_score
        elif event.event_type == EventType.LESSON_RECORDED:
            state.lessons[payload.lesson_id] = LessonView(
                lesson_id=payload.lesson_id,
                status=LessonStatus.ACTIVE.value,
                summary=payload.candidate.summary,
                tags=list(payload.candidate.tags),
                source_event_id=event.event_id,
            )
        elif event.event_type == EventType.LESSON_STATUS_CHANGED:
            lesson = state.lessons[payload.lesson_id]
            lesson.status = payload.status.value
        elif event.event_type == EventType.MANUAL_INTERVENTION:
            state.manual_intervention_count += 1
        elif event.event_type == EventType.RUN_STOPPED:
            finalizable = is_finalizable_stop_reason(payload.reason_code)
            state.status = RunStatus.STOPPED if finalizable else RunStatus.FAILED
            state.phase = "stopped" if finalizable else "failed"
            state.stop_reason_code = payload.reason_code
            state.active_experiment_id = None
            state.active_attempt = None
            state.active_fidelity = None
        elif event.event_type == EventType.FINAL_SELECTED:
            state.status = RunStatus.FINALIZING
            state.phase = "submission"
            state.final_experiment_id = payload.experiment_id
        elif event.event_type == EventType.SUBMISSION_CHECKED:
            state.status = RunStatus.FINALIZED if payload.accepted else RunStatus.FAILED
            state.phase = "finalized" if payload.accepted else "failed"

    state.resource_totals = aggregate_resources(event.resource_delta for event in materialized)
    if state.started_at and state.last_event_at:
        state.elapsed_wall_time_seconds = max(
            0.0, (state.last_event_at - state.started_at).total_seconds()
        )
    return state
