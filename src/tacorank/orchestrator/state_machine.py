"""Legal transition validation for the event-sourced orchestration lifecycle."""

from __future__ import annotations

from typing import Iterable, List, Optional

from ..memory.projections import project
from ..schemas import (
    Event,
    EventPayload,
    EventType,
    ExperimentDecisionKind,
    Fidelity,
    Integrity,
    Population,
    RecoveryAction,
    RunOutcome,
    TrustVerdict,
)
from .state import ExperimentStatus, RunStatus


class TransitionError(RuntimeError):
    pass


def _events_of(events: List[Event], event_type: EventType) -> List[Event]:
    return [event for event in events if event.event_type == event_type]


def _last_for_experiment(
    events: List[Event], event_type: EventType, experiment_id: str
) -> Optional[Event]:
    for event in reversed(events):
        if event.event_type != event_type:
            continue
        data = event.payload.model_dump(mode="python")
        nested = next(
            (
                value
                for value in data.values()
                if isinstance(value, dict) and value.get("experiment_id") == experiment_id
            ),
            None,
        )
        if nested is not None or data.get("experiment_id") == experiment_id:
            return event
    return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransitionError(message)


def _latest_recoverable_failure(events: List[Event], experiment_id: str) -> Optional[Event]:
    for event in reversed(events):
        if event.event_type == EventType.PATCH_CHECKED:
            result = event.payload.result
            if result.experiment_id == experiment_id and not result.accepted:
                return event
        elif event.event_type == EventType.EXECUTION_FINISHED:
            result = event.payload.result
            if result.experiment_id == experiment_id and result.outcome != RunOutcome.SUCCESS:
                return event
        elif event.event_type == EventType.OUTPUT_CHECKED:
            result = event.payload.result
            if result.experiment_id == experiment_id and not result.accepted:
                return event
        elif event.event_type == EventType.EVALUATION_COMPLETED:
            result = event.payload.result
            if (
                result.experiment_id == experiment_id
                and result.trust.verdict == TrustVerdict.NO_OP
            ):
                return event
    return None


def _evaluation_request(events: List[Event], result) -> object:
    output = _last_for_experiment(events, EventType.OUTPUT_CHECKED, result.experiment_id)
    _require(output is not None and output.payload.result.accepted, "evaluation requires accepted output")
    _require(output.payload.result.attempt == result.attempt, "evaluation attempt does not match output")
    finished = _last_for_experiment(events, EventType.EXECUTION_FINISHED, result.experiment_id)
    _require(
        finished is not None and finished.payload.result.outcome == RunOutcome.SUCCESS,
        "evaluation requires a successful execution",
    )
    started = _last_for_experiment(events, EventType.EXECUTION_STARTED, result.experiment_id)
    _require(started is not None, "evaluation has no execution request")
    request = started.payload.request
    _require(
        finished.payload.result.attempt == request.attempt == result.attempt
        and finished.payload.result.fidelity == request.fidelity
        and finished.payload.result.patch_commit_sha == request.patch_commit_sha,
        "evaluation source execution identity mismatch",
    )
    return request


def validate_transition(events: List[Event], payload: EventPayload) -> None:
    """Reject a payload that cannot legally follow the existing event prefix."""

    event_type = EventType(payload.type)
    if not events:
        _require(event_type == EventType.RUN_STARTED, "the first event must be run.started")
        return

    _require(event_type != EventType.RUN_STARTED, "run.started may occur only once")
    state = project(events)
    if state.status in (RunStatus.FINALIZED, RunStatus.FAILED):
        raise TransitionError("no events may follow a finalized run")
    if state.status == RunStatus.STOPPED:
        _require(
            event_type
            in (
                EventType.EXECUTION_STARTED,
                EventType.EXECUTION_FINISHED,
                EventType.OUTPUT_CHECKED,
                EventType.EVALUATION_COMPLETED,
                EventType.FINAL_SELECTED,
            ),
            "only sealed finalization evidence may follow run.stopped",
        )
    if state.status == RunStatus.FINALIZING:
        _require(
            event_type == EventType.SUBMISSION_CHECKED,
            "only submission checking may follow final selection",
        )

    if event_type == EventType.CONTRACT_VERIFIED:
        _require(
            len(events) == 1 and events[0].event_type == EventType.RUN_STARTED,
            "contract.verified must immediately follow run.started",
        )
    elif event_type == EventType.BASELINE_VERIFIED:
        _require(
            bool(_events_of(events, EventType.CONTRACT_VERIFIED)),
            "baseline requires a verified contract",
        )
        _require(
            not _events_of(events, EventType.BASELINE_VERIFIED),
            "baseline may be verified only once",
        )
        contract = _events_of(events, EventType.CONTRACT_VERIFIED)[-1].payload
        _require(
            payload.evaluation.contract_sha256 == contract.contract_sha256
            and payload.evaluation.evaluator_sha256 == contract.evaluator_sha256,
            "baseline evaluation hashes do not match the frozen contract",
        )
    elif event_type == EventType.CONTEXT_CREATED:
        _require(
            bool(_events_of(events, EventType.BASELINE_VERIFIED)),
            "context cannot be created before baseline parity",
        )
        _require(state.status not in (RunStatus.STOPPED, RunStatus.FINALIZING), "run is stopped")
    elif event_type == EventType.PLANNER_RECOMMENDED:
        last_context = _events_of(events, EventType.CONTEXT_CREATED)
        _require(bool(last_context), "planner recommendation requires context")
        _require(last_context[-1].payload.context.role == "planner", "latest context must be planner")
    elif event_type == EventType.EXPERIMENT_PROPOSED:
        spec = payload.spec
        _require(state.status in (RunStatus.READY, RunStatus.RUNNING), "run is not accepting proposals")
        _require(spec.experiment_id not in state.experiments, "duplicate experiment_id")
        _require(
            all(node.duplicate_key != spec.duplicate_key for node in state.experiments.values()),
            "duplicate experiment proposal",
        )
        if spec.parent_experiment_id:
            _require(
                spec.parent_experiment_id in state.experiments
                or spec.parent_experiment_id == "baseline",
                "unknown parent experiment",
            )
        if spec.parent_experiment_id in (None, "baseline"):
            baseline = _events_of(events, EventType.BASELINE_VERIFIED)[-1]
            expected_parent_commit = baseline.payload.commit_sha
        else:
            parent = state.experiments[spec.parent_experiment_id]
            expected_parent_commit = parent.latest_commit_sha
            _require(
                expected_parent_commit is not None,
                "selected parent experiment has no patch commit",
            )
        _require(
            spec.parent_commit_sha == expected_parent_commit,
            "parent_commit_sha does not match the selected parent",
        )
        last_context = _events_of(events, EventType.CONTEXT_CREATED)
        _require(bool(last_context), "proposal requires planner context")
        context = last_context[-1].payload.context
        _require(context.role == "planner" and context.context_id == spec.context_id, "proposal context mismatch")
    elif event_type == EventType.PATCH_CREATED:
        candidate = payload.candidate
        _require(candidate.experiment_id in state.experiments, "patch has unknown experiment")
        node = state.experiments[candidate.experiment_id]
        _require(
            node.status in (ExperimentStatus.PROPOSED, ExperimentStatus.RECOVERING),
            "patch is not legal in the current experiment state",
        )
        proposed = _last_for_experiment(
            events, EventType.EXPERIMENT_PROPOSED, candidate.experiment_id
        )
        _require(
            proposed is not None and proposed.event_id == candidate.experiment_spec_event_id,
            "patch references the wrong experiment proposal",
        )
        contexts = _events_of(events, EventType.CONTEXT_CREATED)
        _require(
            any(
                event.payload.context.context_id == candidate.context_id
                and event.payload.context.role in ("coder", "recovery")
                for event in contexts
            ),
            "patch context does not exist",
        )
    elif event_type == EventType.PATCH_CHECKED:
        result = payload.result
        node = state.experiments.get(result.experiment_id)
        _require(node is not None and node.status == ExperimentStatus.PATCH_READY, "no patch is ready")
        patch = _last_for_experiment(events, EventType.PATCH_CREATED, result.experiment_id)
        _require(patch is not None, "patch check requires patch.created")
        _require(
            patch.payload.candidate.patch_commit_sha == result.patch_commit_sha
            and patch.payload.candidate.diff_sha256 == result.diff_sha256,
            "patch check identity mismatch",
        )
    elif event_type == EventType.EXECUTION_STARTED:
        request = payload.request
        node = state.experiments.get(request.experiment_id)
        if state.status == RunStatus.STOPPED:
            _require(
                node is not None
                and request.experiment_id == state.best_experiment_id
                and request.patch_commit_sha == state.best_commit_sha
                and request.fidelity == Fidelity.FULL
                and request.command_id
                in {"clean_reproduce", "candidate_final_infer", "submission_check"},
                "post-stop execution must target the selected sealed commit",
            )
            if request.command_id == "candidate_final_infer":
                _require(
                    any(
                        event.event_type == EventType.EVALUATION_COMPLETED
                        and event.payload.result.experiment_id == request.experiment_id
                        and event.payload.result.population
                        == Population.PUBLIC_VALIDATION
                        and event.payload.result.fidelity == Fidelity.FULL
                        and event.payload.result.trust.verdict == TrustVerdict.ACCEPTED
                        and event.payload.result.trust.integrity == Integrity.CLEAN
                        and _evaluation_request(
                            events[: events.index(event)], event.payload.result
                        ).command_id
                        == "clean_reproduce"
                        for event in events
                    ),
                    "final inference requires trusted clean reproduction",
                )
            if request.command_id == "submission_check":
                final_outputs = [
                    event
                    for event in events
                    if event.event_type == EventType.OUTPUT_CHECKED
                    and event.payload.result.experiment_id == request.experiment_id
                    and event.payload.result.accepted
                ]
                _require(
                    bool(final_outputs),
                    "submission check requires accepted final output",
                )
                final_request = next(
                    (
                        event.payload.request
                        for event in reversed(events)
                        if event.event_type == EventType.EXECUTION_STARTED
                        and event.payload.request.experiment_id == request.experiment_id
                        and event.payload.request.attempt
                        == final_outputs[-1].payload.result.attempt
                    ),
                    None,
                )
                _require(
                    final_request is not None
                    and final_request.command_id == "candidate_final_infer",
                    "submission check requires accepted final-inference output",
                )
        else:
            _require(
                node is not None and node.status == ExperimentStatus.READY_TO_RUN,
                "patch is not runnable",
            )
        check = _last_for_experiment(events, EventType.PATCH_CHECKED, request.experiment_id)
        _require(check is not None and check.payload.result.accepted, "execution requires Gate A receipt")
        _require(
            check.payload.result.receipt_id == request.patch_receipt_id
            and check.payload.result.patch_commit_sha == request.patch_commit_sha,
            "execution request does not match receipt",
        )
    elif event_type == EventType.EXECUTION_FINISHED:
        result = payload.result
        node = state.experiments.get(result.experiment_id)
        _require(node is not None and node.status == ExperimentStatus.RUNNING, "no execution is running")
        started = _last_for_experiment(events, EventType.EXECUTION_STARTED, result.experiment_id)
        _require(started is not None, "execution result has no request")
        request = started.payload.request
        _require(
            (request.attempt, request.fidelity, request.patch_commit_sha)
            == (result.attempt, result.fidelity, result.patch_commit_sha),
            "execution result identity mismatch",
        )
    elif event_type == EventType.RECOVERY_DECIDED:
        decision = payload.decision
        node = state.experiments.get(decision.experiment_id)
        _require(node is not None and node.status == ExperimentStatus.RECOVERING, "nothing is recoverable")
        consumes_repair = int(decision.action == RecoveryAction.TRAE_REPAIR)
        expected_attempt = min(
            max(1, node.repair_count + 1),
            max(1, state.max_repairs_per_experiment),
        )
        _require(
            decision.repair_attempt == expected_attempt,
            "repair attempt is inconsistent; no next available budget slot",
        )
        failure = _latest_recoverable_failure(events, decision.experiment_id)
        _require(
            failure is not None and decision.failure_event_id == failure.event_id,
            "recovery decision must reference the latest failure",
        )
        _require(
            decision.remaining_repair_budget
            == state.max_repairs_per_experiment - node.repair_count - consumes_repair,
            "remaining repair budget is inconsistent",
        )
        if decision.action == RecoveryAction.RETRY_SAME_COMMIT:
            _require(node.same_commit_retry_count == 0, "same-commit retry has already been used")
    elif event_type == EventType.OUTPUT_CHECKED:
        result = payload.result
        node = state.experiments.get(result.experiment_id)
        _require(node is not None and node.status == ExperimentStatus.OUTPUT_READY, "no output is ready")
        finished = _last_for_experiment(events, EventType.EXECUTION_FINISHED, result.experiment_id)
        _require(finished is not None and finished.payload.result.outcome == RunOutcome.SUCCESS, "Gate B requires a successful execution")
        _require(
            finished.payload.result.prediction_artifact.artifact_id
            == result.prediction_artifact.artifact_id,
            "Gate B checked a different prediction",
        )
        _require(
            finished.payload.result.attempt == result.attempt,
            "Gate B attempt does not match execution",
        )
    elif event_type == EventType.EVALUATION_COMPLETED:
        result = payload.result
        if result.experiment_id == "baseline":
            raise TransitionError("baseline evaluation is recorded only inside baseline.verified")
        node = state.experiments.get(result.experiment_id)
        if result.population == Population.HIDDEN_FINAL:
            _require(
                node is not None
                and result.experiment_id == state.best_experiment_id
                and node.status in (ExperimentStatus.ACCEPTED, ExperimentStatus.OUTPUT_VERIFIED),
                "hidden-final evaluation requires the selected development best",
            )
        else:
            _require(
                node is not None and node.status == ExperimentStatus.OUTPUT_VERIFIED,
                "evaluation requires Gate B",
            )
        request = _evaluation_request(events, result)
        _require(
            (result.attempt, result.fidelity, result.seed)
            == (request.attempt, request.fidelity, request.seed),
            "evaluation result does not match its execution request",
        )
        contract = _events_of(events, EventType.CONTRACT_VERIFIED)[-1].payload
        _require(
            result.contract_sha256 == contract.contract_sha256
            and result.evaluator_sha256 == contract.evaluator_sha256,
            "evaluation hashes do not match the frozen contract",
        )
        if result.population == Population.HIDDEN_FINAL:
            _require(state.status == RunStatus.STOPPED, "hidden-final is available only after stop")
            _require(result.fidelity == Fidelity.FULL, "hidden-final requires full fidelity")
        else:
            if state.status == RunStatus.STOPPED:
                _require(
                    request.command_id == "clean_reproduce"
                    and result.population == Population.PUBLIC_VALIDATION
                    and result.fidelity == Fidelity.FULL,
                    "post-stop evaluation is restricted to clean public reproduction",
                )
            expected_population = (
                Population.INTERNAL_PROXY
                if request.fidelity == Fidelity.PROXY
                else Population.PUBLIC_VALIDATION
            )
            _require(request.fidelity != Fidelity.SMOKE, "smoke output cannot be evaluated")
            _require(
                result.population == expected_population,
                "evaluation population does not match requested fidelity",
            )
        expected_query_index = (
            state.public_validation_queries + 1
            if result.population == Population.PUBLIC_VALIDATION
            else None
        )
        _require(
            result.public_query_index == expected_query_index,
            "evaluation public query index is not the next frozen index",
        )
    elif event_type == EventType.EXPERIMENT_DECIDED:
        decision = payload.decision
        node = state.experiments.get(decision.experiment_id)
        _require(node is not None, "decision has unknown experiment")
        if decision.fidelity_completed == Fidelity.SMOKE:
            _require(node.status == ExperimentStatus.OUTPUT_VERIFIED, "smoke decision requires Gate B")
            _require(decision.evaluation_event_id is None, "smoke decision cannot cite evaluation")
        else:
            _require(node.status == ExperimentStatus.EVALUATED, "decision requires evaluation")
            evaluation = _last_for_experiment(events, EventType.EVALUATION_COMPLETED, decision.experiment_id)
            _require(
                evaluation is not None and evaluation.event_id == decision.evaluation_event_id,
                "decision cites the wrong evaluation",
            )
            _require(
                decision.fidelity_completed == evaluation.payload.result.fidelity,
                "decision fidelity does not match evaluation",
            )
            if decision.best_eligible:
                result = evaluation.payload.result
                _require(
                    result.fidelity == Fidelity.FULL
                    and result.population == Population.PUBLIC_VALIDATION
                    and result.trust.verdict == TrustVerdict.ACCEPTED,
                    "best eligibility requires trusted public full evaluation",
                )
                _require(
                    decision.decision == ExperimentDecisionKind.ACCEPT,
                    "only an accepted experiment can be best eligible",
                )
        if decision.decision == ExperimentDecisionKind.PROMOTE:
            _require(decision.next_fidelity is not None, "promotion requires next fidelity")
            if decision.fidelity_completed == Fidelity.FULL:
                _require(decision.next_fidelity == Fidelity.FULL, "full can only promote to confirmation")
                _require(
                    node.confirmation_count < state.max_confirmation_attempts,
                    "confirmation budget exceeded",
                )
            else:
                fidelity_order = {Fidelity.SMOKE: 0, Fidelity.PROXY: 1, Fidelity.FULL: 2}
                _require(
                    fidelity_order[decision.next_fidelity]
                    > fidelity_order[decision.fidelity_completed],
                    "promotion must move to a higher fidelity",
                )
    elif event_type == EventType.BEST_UPDATED:
        node = state.experiments.get(payload.experiment_id)
        _require(
            node is not None
            and node.status == ExperimentStatus.ACCEPTED
            and node.best_eligible,
            "experiment is not an accepted best-eligible candidate",
        )
        decision = next(
            (
                event
                for event in events
                if event.event_id == payload.decision_event_id
                and event.event_type == EventType.EXPERIMENT_DECIDED
            ),
            None,
        )
        _require(
            decision is not None
            and decision.event_id == node.terminal_event_id
            and decision.payload.decision.experiment_id == payload.experiment_id
            and decision.payload.decision.decision == ExperimentDecisionKind.ACCEPT
            and decision.payload.decision.best_eligible,
            "invalid decision reference",
        )
        _require(
            payload.commit_sha == node.latest_commit_sha,
            "best commit does not match the accepted patch",
        )
        _require(
            node.metric_set is not None
            and node.metric_set.primary_metric_name == payload.primary_metric_name
            and node.metric_set.primary_score == payload.primary_score,
            "best score does not match evaluation",
        )
        _require(
            state.best_primary_score is None or payload.primary_score > state.best_primary_score,
            "best update must strictly improve the incumbent",
        )
    elif event_type == EventType.LESSON_RECORDED:
        _require(payload.lesson_id not in state.lessons, "duplicate lesson_id")
        known_ids = {event.event_id for event in events}
        _require(
            set(payload.candidate.source_event_ids).issubset(known_ids),
            "lesson cites unknown evidence",
        )
    elif event_type == EventType.LESSON_STATUS_CHANGED:
        _require(payload.lesson_id in state.lessons, "unknown lesson_id")
    elif event_type == EventType.RUN_STOPPED:
        _require(bool(_events_of(events, EventType.BASELINE_VERIFIED)), "cannot stop before baseline")
        _require(state.status not in (RunStatus.STOPPED, RunStatus.FINALIZING), "run already stopped")
    elif event_type == EventType.FINAL_SELECTED:
        _require(state.status == RunStatus.STOPPED, "final selection requires stopped run")
        _require(payload.experiment_id == state.best_experiment_id, "must select latest verified best")
        _require(payload.commit_sha == state.best_commit_sha, "final commit must match verified best")
        reproduction = next(
            (
                event
                for event in events
                if event.event_id == payload.reproduction_evaluation_event_id
                and event.event_type
                in (EventType.EVALUATION_COMPLETED, EventType.BASELINE_VERIFIED)
            ),
            None,
        )
        _require(reproduction is not None, "final selection cites unknown reproduction evidence")
        result = (
            reproduction.payload.evaluation
            if reproduction.event_type == EventType.BASELINE_VERIFIED
            else reproduction.payload.result
        )
        _require(
            result.experiment_id == payload.experiment_id
            and result.fidelity == Fidelity.FULL
            and result.population == Population.PUBLIC_VALIDATION
            and result.trust.verdict == TrustVerdict.ACCEPTED
            and result.trust.integrity == Integrity.CLEAN
            and result.metric_set.primary_score == state.best_primary_score,
            "final reproduction is not trusted evidence for the verified best",
        )
        if reproduction.event_type == EventType.BASELINE_VERIFIED:
            _require(
                payload.experiment_id == "baseline"
                and reproduction.payload.commit_sha == payload.commit_sha,
                "baseline finalization does not match frozen parity evidence",
            )
        else:
            reproduction_index = events.index(reproduction)
            request = _evaluation_request(events[:reproduction_index], result)
            _require(
                request.patch_commit_sha == payload.commit_sha
                and request.command_id == "clean_reproduce",
                "final reproduction did not cleanly evaluate the selected commit",
            )
    elif event_type == EventType.SUBMISSION_CHECKED:
        _require(state.status == RunStatus.FINALIZING, "submission check requires final selection")


def validator(events: List[Event], payload: EventPayload) -> None:
    validate_transition(events, payload)
