"""Deterministic outer-loop routing across independently replaceable adapters."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Deque, Optional, Sequence, Tuple

from ..config import RunConfig, VerifiedContract
from ..coding.redaction import SecretRedactor
from ..context.builder import ContextBuilder
from ..evaluation.final_selection import rank_finalists, select_final
from ..memory.canonical_json import canonical_sha256
from ..memory.event_store import DuplicateIdempotencyKey, EventStore, LedgerError
from ..memory.projections import project
from ..recovery.fingerprints import fingerprint_failure, fingerprint_result, normalize_text
from ..reporting import rebuild_views
from ..run_layout import RunLayout
from ..schemas import (
    AdapterFailedPayload,
    AdapterFailureResult,
    ArtifactKind,
    BaselineVerifiedPayload,
    BestUpdatedPayload,
    CheckResult,
    CheckStatus,
    ContextCreatedPayload,
    ContractVerifiedPayload,
    EvaluationCompletedPayload,
    EvaluationDecisionContext,
    EvaluationRequest,
    Event,
    EventType,
    ExperimentDecidedPayload,
    ExperimentDecision,
    ExperimentDecisionKind,
    ExperimentProposedPayload,
    ExperimentSpec,
    ExecutionFinishedPayload,
    ExecutionStartedPayload,
    Fidelity,
    FinalSelectedPayload,
    Integrity,
    LessonRecordedPayload,
    OutputCheckedPayload,
    PatchCheckedPayload,
    PatchCreatedPayload,
    PlannerAction,
    PlannerRecommendedPayload,
    Population,
    RecoveryAction,
    RecoveryDecidedPayload,
    RecoveryPolicyContext,
    ResourceDelta,
    RunOutcome,
    RunRequest,
    RunStartedPayload,
    RunStoppedPayload,
    SubmissionCheckedPayload,
    TrustVerdict,
)
from .convergence import StopDecision, runtime_budget_decision, stop_decision
from .finalize import (
    FinalizationError,
    baseline_reproduction_event_id,
    candidate_finalization_plan,
    finalization_candidates,
)
from .state_machine import TransitionError
from .ports import (
    CodingWorker,
    Evaluator,
    ExecutionRunner,
    FinalSubmissionProvider,
    HealthObserver,
    OutputGate,
    PatchGate,
    RecoveryManager,
    ResearchPlanner,
)


class OrchestrationError(RuntimeError):
    pass


class ResumablePlanningError(OrchestrationError):
    """Provider output is invalid, but the durable planning checkpoint is safe.

    This is deliberately distinct from control-plane failures.  Callers can
    replace the planner and resume from the persisted ``planner_context``
    without marking the run stopped or fabricating a recovery event.
    """


def _is_search_space_exhaustion(reason_code: str) -> bool:
    """Return true only for planner outcomes that prove no legal choice remains."""

    normalized = reason_code.strip().upper()
    return (
        "EXHAUSTED" in normalized
        or normalized.startswith("NO_LEGAL_")
        or normalized in {"NO_METHOD_CARDS", "NO_ELIGIBLE_METHOD"}
    )


class Harness:
    def __init__(
        self,
        *,
        config: RunConfig,
        verified_contract: VerifiedContract,
        event_store: EventStore,
        context_builder: ContextBuilder,
        planner: ResearchPlanner,
        coding_worker: CodingWorker,
        patch_gate: PatchGate,
        runner: ExecutionRunner,
        health_observer: HealthObserver,
        recovery_manager: RecoveryManager,
        output_gate: OutputGate,
        evaluator: Evaluator,
        final_submission_provider: Optional[FinalSubmissionProvider] = None,
    ):
        self.config = config
        self.verified_contract = verified_contract
        self.event_store = event_store
        self.context_builder = context_builder
        self.planner = planner
        self.coding_worker = coding_worker
        self.patch_gate = patch_gate
        self.runner = runner
        self.health_observer = health_observer
        self.recovery_manager = recovery_manager
        self.output_gate = output_gate
        self.evaluator = evaluator
        self.final_submission_provider = final_submission_provider

    def _key(self, experiment_id: str, stage: str, attempt: int, value: object) -> str:
        return "%s:%s:%s:%d:%s" % (
            self.config.run_id,
            experiment_id,
            stage,
            attempt,
            canonical_sha256(value),
        )

    def _append(
        self,
        payload,
        *,
        stage: str,
        experiment_id: str = "run",
        attempt: int = 0,
        causation_event_id: Optional[str] = None,
        resource_delta: Optional[ResourceDelta] = None,
    ) -> Event:
        key = self._key(experiment_id, stage, attempt, payload)
        try:
            event = self.event_store.append(
                run_id=self.config.run_id,
                payload=payload,
                idempotency_key=key,
                causation_event_id=causation_event_id,
                resource_delta=resource_delta,
            )
        except DuplicateIdempotencyKey as duplicate:
            # The immutable input hash proves this is the same acknowledged action.
            event = duplicate.event
        events = self.event_store.read_events()
        rebuild_views(
            RunLayout(
                self.config.repository_root,
                self.config.run_id,
            ).run_directory,
            events,
        )
        return event

    def events(self) -> Sequence[Event]:
        return self.event_store.read_events(repair_tail=True)

    def state(self):
        return project(self.events())

    def _execution_seed(
        self, experiment_id: str, fidelity: Fidelity, patch_commit_sha: str
    ) -> int:
        if fidelity != Fidelity.FULL:
            return self.config.seed_schedule[0]
        events = self.events()
        event_by_id = {event.event_id: event for event in events}
        completed = 0
        for event in events:
            if not (
                event.event_type == EventType.EVALUATION_COMPLETED
                and event.payload.result.experiment_id == experiment_id
                and event.payload.result.fidelity == Fidelity.FULL
                and event.payload.result.population == Population.PUBLIC_VALIDATION
            ):
                continue
            output = event_by_id.get(event.causation_event_id)
            finished = (
                event_by_id.get(output.causation_event_id)
                if output is not None
                else None
            )
            started = (
                event_by_id.get(finished.causation_event_id)
                if finished is not None
                else None
            )
            if (
                started is None
                or started.event_type != EventType.EXECUTION_STARTED
            ):
                raise OrchestrationError(
                    "full evaluation cannot resolve its execution identity"
                )
            if started.payload.request.patch_commit_sha == patch_commit_sha:
                completed += 1
        if completed >= len(self.config.seed_schedule):
            raise OrchestrationError(
                "distinct full-fidelity seed schedule exhausted for %s"
                % experiment_id
            )
        return self.config.seed_schedule[completed]

    def _reference_metrics(
        self, experiment_id: Optional[str], fidelity: Fidelity
    ) -> dict:
        if not experiment_id or experiment_id == "baseline":
            return self._baseline_metrics()
        for event in reversed(self.events()):
            if (
                event.event_type == EventType.EVALUATION_COMPLETED
                and event.payload.result.experiment_id == experiment_id
                and event.payload.result.fidelity == fidelity
            ):
                metric_set = event.payload.result.metric_set
                values = dict(metric_set.metrics)
                values.setdefault(
                    metric_set.primary_metric_name, metric_set.primary_score
                )
                return values
        raise OrchestrationError(
            "reference experiment %s has no %s evaluation"
            % (experiment_id, fidelity.value)
        )

    def bootstrap(self, baseline_evaluation) -> None:
        """Freeze run identity and record independently supplied baseline parity."""

        self.config.validate_metric_set(baseline_evaluation.metric_set)
        started = self._append(
            RunStartedPayload(
                config_sha256=self.verified_contract.config_sha256,
                contract_sha256=self.verified_contract.contract_sha256,
                protected_paths_sha256=self.verified_contract.protected_paths_sha256,
                max_experiments=self.config.max_experiments,
                wall_time_limit_seconds=self.config.wall_time_limit_seconds,
                token_limit=self.config.token_limit,
                gpu_seconds_limit=self.config.gpu_seconds_limit,
                max_repairs_per_experiment=self.config.max_repairs_per_experiment,
                max_confirmation_attempts=self.config.max_confirmation_attempts,
                seed_schedule=self.config.seed_schedule,
                convergence_epsilon=self.config.convergence_epsilon,
                convergence_patience=self.config.convergence_patience,
            ),
            stage="started",
        )
        contract = self._append(
            ContractVerifiedPayload(
                contract_sha256=self.verified_contract.contract_sha256,
                protected_paths_sha256=self.verified_contract.protected_paths_sha256,
                evaluator_sha256=self.config.evaluator_sha256,
                metric_names=self.config.metric_names,
                primary_metric_name=self.config.primary_metric_name,
                command_ids=self.config.command_ids,
                artifact_roots=self.config.artifact_roots,
            ),
            stage="contract_verified",
            causation_event_id=started.event_id,
        )
        self._append(
            BaselineVerifiedPayload(
                commit_sha=self.config.baseline_commit_sha,
                metric_set=baseline_evaluation.metric_set,
                evaluation=baseline_evaluation,
            ),
            stage="baseline_verified",
            experiment_id="baseline",
            causation_event_id=contract.event_id,
            resource_delta=baseline_evaluation.resource_delta,
        )

    def deterministic_stop(self, **kwargs) -> StopDecision:
        events = self.events()
        return stop_decision(project(events), events, self.config, **kwargs)

    def stop(self, decision: StopDecision) -> Event:
        if not decision.stop:
            raise OrchestrationError("refusing to stop without a matched deterministic rule")
        return self._append(
            RunStoppedPayload(reason_code=decision.reason_code, reason=decision.reason),
            stage="stopped",
            causation_event_id=self.events()[-1].event_id,
        )

    def _stop_if_runtime_budget_exhausted(self) -> bool:
        decision = runtime_budget_decision(self.state(), self.config)
        if not decision.stop:
            return False
        self.stop(decision)
        return True

    def _command_for(self, fidelity: Fidelity) -> str:
        for preferred in (
            "candidate_%s" % fidelity.value,
            "run_%s" % fidelity.value,
        ):
            if preferred in self.config.command_ids:
                return preferred
        return self.config.command_ids[0]

    def _baseline_metrics(self) -> dict:
        baseline = next(
            event for event in self.events() if event.payload.type == "baseline.verified"
        )
        metric_set = baseline.payload.metric_set
        values = dict(metric_set.metrics)
        values.setdefault(metric_set.primary_metric_name, metric_set.primary_score)
        return values

    def _previous_failure_fingerprints(
        self, experiment_id: str, before_event_id: str
    ) -> list[str]:
        """Recompute authoritative fingerprints from prior failure evidence."""
        fingerprints = []
        for event in self.events():
            if event.event_id == before_event_id:
                break
            result = getattr(event.payload, "result", None)
            if result is None or getattr(result, "experiment_id", None) != experiment_id:
                continue
            raw_outcome = getattr(result, "outcome", None)
            outcome = getattr(raw_outcome, "value", raw_outcome)
            trust = getattr(result, "trust", None)
            raw_verdict = getattr(trust, "verdict", None)
            verdict = getattr(raw_verdict, "value", raw_verdict)
            accepted = getattr(result, "accepted", None)
            if outcome in {"success", "cancelled"}:
                continue
            if outcome is None and accepted is not False and verdict != "no_op":
                continue
            fingerprints.append(fingerprint_result(result))
        return fingerprints

    def _attempt_history(
        self, experiment_id: str, before_event_id: str
    ) -> list[dict]:
        history = []
        for event in self.events():
            if event.event_id == before_event_id:
                break
            if event.payload.type != "recovery.decided":
                continue
            decision = event.payload.decision
            if decision.experiment_id != experiment_id:
                continue
            history.append(
                {
                    "failure_event_id": decision.failure_event_id,
                    "action": decision.action.value,
                    "reason_code": decision.reason_code,
                    "repair_attempt": decision.repair_attempt,
                    "remaining_repair_budget": decision.remaining_repair_budget,
                }
            )
        return history

    def _experiment_spec(self, experiment_id: str):
        for event in self.events():
            if event.payload.type == "experiment.proposed":
                if event.payload.spec.experiment_id == experiment_id:
                    return event.payload.spec
        raise OrchestrationError("recovery experiment specification is missing")

    def _contract_summary(self) -> str:
        return (
            "the frozen benchmark split and evaluator, protected labels and data, "
            "protected paths, command configuration, baseline identity, and "
            "submission contract"
        )

    def _remaining_run_budget(self, state) -> dict:
        totals = state.resource_totals
        remaining = {
            "experiments": state.remaining_experiments,
            "wall_time_seconds": max(
                0,
                state.wall_time_limit_seconds
                - int(state.elapsed_wall_time_seconds),
            ),
        }
        if self.config.token_limit is not None:
            remaining["token"] = max(
                0,
                self.config.token_limit
                - totals.provider_tokens
                - totals.estimated_tokens,
            )
        if self.config.gpu_seconds_limit is not None:
            remaining["gpu_seconds"] = max(
                0,
                self.config.gpu_seconds_limit
                - int(totals.gpu_weighted_time_ms / 1000),
            )
        return remaining

    def _current_runtime_settings(
        self, experiment_id: str, before_event_id: str
    ) -> dict:
        current = {}
        for event in self.events():
            if event.event_id == before_event_id:
                break
            if event.payload.type != "execution.started":
                continue
            request = event.payload.request
            if request.experiment_id == experiment_id:
                current = dict(request.runtime_settings)
        return current

    def _validate_runtime_adjustments(self, adjustments: dict) -> None:
        configured = self.config.allowed_runtime_adjustments
        for name, value in adjustments.items():
            if name not in configured:
                raise OrchestrationError("runtime adjustment is not frozen")
            approved = configured[name]
            if isinstance(approved, dict):
                approved = approved.get("next_value", approved.get("value"))
            elif isinstance(approved, (list, tuple)):
                approved = approved[0] if approved else None
            if value != approved:
                raise OrchestrationError("runtime adjustment value is not frozen")
            if name == "timeout_profile" and value not in self.config.timeout_profiles:
                raise OrchestrationError("timeout profile is not frozen")

    def _record_adapter_failure(
        self,
        *,
        experiment_id: str,
        attempt: int,
        stage: str,
        error: Exception,
        causation_event_id: Optional[str] = None,
    ) -> Event:
        """Persist a safe, typed adapter exception for policy-driven recovery."""

        if stage not in {
            "coding", "patch_gate", "execution", "output_gate", "evaluation", "recovery"
        }:
            raise OrchestrationError("unknown adapter failure stage")
        outcome = {
            "coding": RunOutcome.CODE_ERROR,
            "patch_gate": RunOutcome.CODE_ERROR,
            "execution": RunOutcome.INFRASTRUCTURE_ERROR,
            "output_gate": RunOutcome.INTERFACE_ERROR,
            "evaluation": RunOutcome.INFRASTRUCTURE_ERROR,
            "recovery": RunOutcome.INFRASTRUCTURE_ERROR,
        }[stage]
        error_class = str(getattr(error, "code", None) or type(error).__name__).strip()
        summary = str(getattr(error, "summary", None) or str(error)).strip()
        output_tail = str(getattr(error, "output_tail", None) or "").strip()
        combined = summary + (("\n" + output_tail) if output_tail else "")
        safe_summary = (
            normalize_text(SecretRedactor().redact(combined))[:800] or error_class
        )
        diagnostic_artifacts = list(
            getattr(error, "diagnostic_artifacts", ()) or ()
        )
        artifact_store = getattr(self.event_store, "artifact_store", None)
        if output_tail and artifact_store is not None and not diagnostic_artifacts:
            try:
                safe_tail = SecretRedactor().redact(output_tail)[-64 * 1024:]
                digest = canonical_sha256(safe_tail)
                diagnostic_artifacts.append(
                    artifact_store.write(
                        artifact_id="trae-failure-%s" % digest[:24],
                        kind=ArtifactKind.LOG,
                        relative_path=(
                            "artifacts/%s/%s/attempt_%d/trae-failure-%s.log"
                            % (
                                self.config.run_id,
                                experiment_id,
                                max(1, int(attempt)),
                                digest[:24],
                            )
                        ),
                        content=(safe_tail.rstrip() + "\n").encode("utf-8"),
                        content_type="text/plain; charset=utf-8",
                    )
                )
            except Exception:
                # Failure evidence must not mask the original adapter error.
                diagnostic_artifacts = []
        raw_delta = getattr(error, "resource_delta", None)
        resource_delta = (
            ResourceDelta.model_validate(raw_delta)
            if raw_delta is not None
            else ResourceDelta()
        )
        result = AdapterFailureResult(
            run_id=self.config.run_id,
            experiment_id=experiment_id,
            attempt=max(1, int(attempt)),
            failure_stage=stage,
            outcome=outcome,
            error_class=error_class,
            error_fingerprint=fingerprint_failure(error_class, safe_summary),
            error_summary=safe_summary,
            diagnostic_artifacts=diagnostic_artifacts,
            resource_delta=resource_delta,
        )
        return self._append(
            AdapterFailedPayload(result=result),
            stage="adapter_failed_%s" % stage,
            experiment_id=experiment_id,
            attempt=result.attempt,
            causation_event_id=causation_event_id,
            resource_delta=result.resource_delta,
        )

    async def _recover(
        self,
        failure_event: Event,
        failed_value,
        experiment_id: str,
    ) -> Tuple[RecoveryAction, object]:
        state = self.state()
        node = state.experiments[experiment_id]
        remaining = max(0, state.max_repairs_per_experiment - node.repair_count)
        context = self.context_builder.build_recovery(
            self.events(), experiment_id, remaining_repair_budget=remaining
        )
        context_event = self._append(
            ContextCreatedPayload(context=context),
            stage="recovery_context",
            experiment_id=experiment_id,
            attempt=node.repair_count + 1,
            causation_event_id=failure_event.event_id,
        )
        decision = await self.recovery_manager.decide(
            failure_event.event_id,
            failed_value,
            RecoveryPolicyContext(
                run_id=self.config.run_id,
                experiment_id=experiment_id,
                original_experiment_spec=self._experiment_spec(experiment_id),
                current_patch_commit_sha=node.latest_commit_sha or node.base_commit_sha,
                failure_event_id=failure_event.event_id,
                failure_stage=getattr(
                    getattr(failed_value, "failure_stage", None), "value", None
                )
                or getattr(failed_value, "failure_stage", None)
                or "execution",
                attempt_history=self._attempt_history(
                    experiment_id, failure_event.event_id
                ),
                repair_attempts_used=node.repair_count,
                max_repair_attempts=state.max_repairs_per_experiment,
                same_commit_retries_used=node.same_commit_retry_count,
                remaining_repair_budget=remaining,
                previous_error_fingerprints=self._previous_failure_fingerprints(
                    experiment_id, failure_event.event_id
                ),
                remaining_run_budget=self._remaining_run_budget(state),
                allowed_runtime_adjustments=self.config.allowed_runtime_adjustments,
                current_runtime_settings=self._current_runtime_settings(
                    experiment_id, failure_event.event_id
                ),
                contract_summary=self._contract_summary(),
            ),
        )
        decision_event = self._append(
            RecoveryDecidedPayload(decision=decision),
            stage="recovery_decided",
            experiment_id=experiment_id,
            attempt=decision.repair_attempt,
            causation_event_id=context_event.event_id,
            resource_delta=decision.resource_delta,
        )
        if decision.lesson_candidate is not None:
            lesson_number = 1 + sum(
                event.payload.type == "lesson.recorded" for event in self.events()
            )
            self._append(
                LessonRecordedPayload(
                    lesson_id="lesson_%03d" % lesson_number,
                    candidate=decision.lesson_candidate,
                ),
                stage="recovery_lesson_recorded",
                experiment_id=experiment_id,
                attempt=decision.repair_attempt,
                causation_event_id=decision_event.event_id,
            )
        if decision.reason_code == "INTEGRITY_VIOLATION":
            self.stop(self.deterministic_stop(fatal_integrity=True))
        return decision.action, (decision, context, decision_event)

    async def _execute_code_repair(
        self,
        recovery: object,
        proposal_event_id: str,
    ):
        decision, context, decision_event = recovery
        if decision.action != RecoveryAction.TRAE_REPAIR:
            raise OrchestrationError("only Trae repair can create a replacement patch")
        context = context.model_copy(
            update={"recovery_instructions": decision.instructions}
        )
        candidate = await self.coding_worker.repair_patch(context, decision)
        candidate = candidate.__class__.model_validate(
            {
                **candidate.model_dump(mode="json"),
                "attempt": decision.repair_attempt + 1,
                "experiment_spec_event_id": proposal_event_id,
                "context_id": context.context_id,
            }
        )
        patch_event = self._append(
            PatchCreatedPayload(candidate=candidate),
            stage="repair_patch_created",
            experiment_id=candidate.experiment_id,
            attempt=candidate.attempt,
            causation_event_id=decision_event.event_id,
            resource_delta=candidate.resource_delta,
        )
        check = await self.patch_gate.check(candidate)
        check_event = self._append(
            PatchCheckedPayload(result=check),
            stage="repair_patch_checked",
            experiment_id=candidate.experiment_id,
            attempt=candidate.attempt,
            causation_event_id=patch_event.event_id,
            resource_delta=check.resource_delta,
        )
        return candidate, check, check_event

    def _adapter_failure_stage(self, experiment_id: str) -> str:
        """Infer the boundary whose adapter was running when an exception escaped."""

        for event in reversed(self.events()):
            if event.event_type == EventType.RECOVERY_DECIDED:
                if event.payload.decision.experiment_id == experiment_id:
                    return "coding"
            elif event.event_type == EventType.PATCH_CREATED:
                if event.payload.candidate.experiment_id == experiment_id:
                    return "patch_gate"
            elif event.event_type == EventType.EXECUTION_STARTED:
                if event.payload.request.experiment_id == experiment_id:
                    return "execution"
            elif event.event_type == EventType.EXECUTION_FINISHED:
                if event.payload.result.experiment_id == experiment_id:
                    return "output_gate"
            elif event.event_type == EventType.OUTPUT_CHECKED:
                if event.payload.result.experiment_id == experiment_id:
                    return "evaluation"
            elif event.event_type == EventType.EVALUATION_COMPLETED:
                if event.payload.result.experiment_id == experiment_id:
                    return "evaluation"
            elif event.event_type == EventType.CONTEXT_CREATED:
                if (
                    event.payload.context.role in ("coder", "recovery")
                    and event.payload.context.experiment_id == experiment_id
                ):
                    return (
                        "recovery"
                        if event.payload.context.role == "recovery"
                        else "coding"
                    )
            elif event.event_type == EventType.EXPERIMENT_PROPOSED:
                if event.payload.spec.experiment_id == experiment_id:
                    return "coding"
        return "recovery"

    async def _handle_unexpected_adapter_failure(self, error: Exception) -> object:
        """Turn an escaped adapter exception into evidence and a bounded decision.

        Ledger/schema failures are not repair inputs. They are stopped fail-closed
        so a broken control plane cannot be handed to a coding worker.
        """

        if isinstance(error, (LedgerError, TransitionError, OrchestrationError, AssertionError)):
            self.stop(
                StopDecision(
                    True,
                    "RECOVERY_CONTROL_PLANE_FAILURE",
                    "The recovery control plane failed; the run was stopped fail-closed.",
                )
            )
            return self.state()
        state = self.state()
        experiment_id = state.active_experiment_id
        if not experiment_id or experiment_id not in state.experiments:
            self.stop(
                StopDecision(
                    True,
                    "ORCHESTRATOR_ADAPTER_FAILURE",
                    "The orchestration adapter failed before an experiment could be recovered.",
                )
            )
            return self.state()
        stage = self._adapter_failure_stage(experiment_id)
        last_event = self.events()[-1]
        failure_event = self._record_adapter_failure(
            experiment_id=experiment_id,
            attempt=max(1, state.active_attempt),
            stage=stage,
            error=error,
            causation_event_id=last_event.event_id,
        )
        action, _ = await self._recover(
            failure_event,
            failure_event.payload.result,
            experiment_id,
        )
        # An exception means the normal continuation point is unknown. Even
        # when policy records a repair/retry action, stop rather than guessing
        # which side effects completed. Resume can inspect the durable evidence.
        self.stop(
            StopDecision(
                True,
                "ADAPTER_FAILURE_%s" % action.value.upper(),
                "The %s adapter failed; recovery recorded %s and the run was stopped safely."
                % (stage, action.value),
            )
        )
        return self.state()

    async def run_one_experiment(self) -> object:
        try:
            return await self._run_one_experiment()
        except ResumablePlanningError:
            # Invalid provider output is an expected, operator-resumable
            # boundary.  Preserve the planning checkpoint and let the caller
            # inspect/replace the provider instead of converting it into a
            # stopped run.
            raise
        except Exception as error:
            try:
                return await self._handle_unexpected_adapter_failure(error)
            except Exception:
                # A second failure while recording/recovering is a control
                # plane failure. Do not recurse or hand it to a coding worker.
                self.stop(
                    StopDecision(
                        True,
                        "RECOVERY_CONTROL_PLANE_FAILURE",
                        "Recovery could not safely record an adapter failure; the run was stopped.",
                    )
                )
                return self.state()

    async def _run_one_experiment(self) -> object:
        events = self.events()
        stop = stop_decision(project(events), events, self.config)
        if stop.stop:
            self.stop(stop)
            return self.state()

        planner_context = self.context_builder.build_planner(events)
        planner_context_event = self._append(
            ContextCreatedPayload(context=planner_context),
            stage="planner_context",
            causation_event_id=events[-1].event_id,
        )
        planner_output = await self.planner.propose(planner_context)
        if planner_output.action != PlannerAction.PROPOSE:
            self._append(
                PlannerRecommendedPayload(output=planner_output),
                stage="planner_recommended",
                causation_event_id=planner_context_event.event_id,
                resource_delta=planner_output.resource_delta,
            )
            if planner_output.action == PlannerAction.BLOCKED:
                if planner_output.reason_code == "INVALID_PROVIDER_PLAN":
                    raise ResumablePlanningError(
                        "research provider failed bounded plan validation; "
                        "resume from the persisted planner checkpoint"
                    )
                if not _is_search_space_exhaustion(planner_output.reason_code):
                    raise ResumablePlanningError(
                        "planner blocked on %s without proving search-space exhaustion; "
                        "resume from the persisted planner checkpoint"
                        % planner_output.reason_code
                    )
                decision = self.deterministic_stop(no_legal_proposal=True)
            else:
                decision = self.deterministic_stop()
                if not decision.stop:
                    raise ResumablePlanningError(
                        "planner stop recommendation is invalid and no frozen stop rule matched; "
                        "resume from the persisted planner checkpoint"
                    )
            self.stop(decision)
            return self.state()

        proposal = planner_output.spec
        assert proposal is not None
        if proposal.context_id != planner_context.context_id:
            raise ResumablePlanningError(
                "planner proposal cites a different context; "
                "resume from the persisted planner checkpoint"
            )
        spec = self.context_builder.bind_implementation(proposal)
        proposal_event = self._append(
            ExperimentProposedPayload(spec=spec),
            stage="proposed",
            experiment_id=spec.experiment_id,
            causation_event_id=planner_context_event.event_id,
            resource_delta=planner_output.resource_delta,
        )
        if self._stop_if_runtime_budget_exhausted():
            return self.state()
        coder_context = self.context_builder.build_coder(self.events(), spec)
        coder_context_event = self._append(
            ContextCreatedPayload(context=coder_context),
            stage="coder_context",
            experiment_id=spec.experiment_id,
            attempt=1,
            causation_event_id=proposal_event.event_id,
        )
        patch_causation_event_id = coder_context_event.event_id
        try:
            patch = await self.coding_worker.create_patch(coder_context, spec)
        except Exception as error:
            failure_event = self._record_adapter_failure(
                experiment_id=spec.experiment_id,
                attempt=1,
                stage="coding",
                error=error,
                causation_event_id=coder_context_event.event_id,
            )
            action, recovery = await self._recover(
                failure_event, failure_event.payload.result, spec.experiment_id
            )
            if action == RecoveryAction.RETRY_SAME_COMMIT:
                # The coding adapter restored its disposable worktree while
                # retaining immutable failure evidence. Waihong's policy permits
                # exactly one retry of the same frozen assignment.
                _, _, recovery_event = recovery
                try:
                    patch = await self.coding_worker.create_patch(coder_context, spec)
                except Exception as retry_error:
                    retry_failure = self._record_adapter_failure(
                        experiment_id=spec.experiment_id,
                        attempt=2,
                        stage="coding",
                        error=retry_error,
                        causation_event_id=recovery_event.event_id,
                    )
                    await self._recover(
                        retry_failure,
                        retry_failure.payload.result,
                        spec.experiment_id,
                    )
                    return self.state()
                patch = patch.__class__.model_validate(
                    {
                        **patch.model_dump(mode="json"),
                        "experiment_spec_event_id": proposal_event.event_id,
                    }
                )
                patch_causation_event_id = recovery_event.event_id
            else:
                # Recovery owns the outcome. ABANDON terminates only this
                # experiment and returns the run to planning; a deliberate
                # integrity violation has already stopped the run in _recover.
                return self.state()
        patch = patch.__class__.model_validate(
            {
                **patch.model_dump(mode="json"),
                "experiment_spec_event_id": proposal_event.event_id,
            }
        )
        patch_event = self._append(
            PatchCreatedPayload(candidate=patch),
            stage="patch_created",
            experiment_id=spec.experiment_id,
            attempt=patch.attempt,
            causation_event_id=patch_causation_event_id,
            resource_delta=patch.resource_delta,
        )
        if self._stop_if_runtime_budget_exhausted():
            return self.state()

        patch_check_event = None
        try:
            patch_check = await self.patch_gate.check(patch)
        except Exception as error:
            failure_event = self._record_adapter_failure(
                experiment_id=spec.experiment_id,
                attempt=patch.attempt,
                stage="patch_gate",
                error=error,
                causation_event_id=patch_event.event_id,
            )
            action, recovery = await self._recover(
                failure_event, failure_event.payload.result, spec.experiment_id
            )
            if action == RecoveryAction.TRAE_REPAIR:
                patch, patch_check, patch_check_event = await self._execute_code_repair(
                    recovery, proposal_event.event_id
                )
            else:
                self.stop(
                    StopDecision(
                        True,
                        "PATCH_GATE_FAILURE",
                        "The patch gate adapter failed; recovery recorded %s and the run was stopped safely."
                        % action.value,
                    )
                )
                return self.state()
        if patch_check_event is None:
            patch_check_event = self._append(
                PatchCheckedPayload(result=patch_check),
                stage="patch_checked",
                experiment_id=spec.experiment_id,
                attempt=patch.attempt,
                causation_event_id=patch_event.event_id,
                resource_delta=patch_check.resource_delta,
            )
        while not patch_check.accepted:
            action, recovery = await self._recover(
                patch_check_event, patch_check, spec.experiment_id
            )
            if action != RecoveryAction.TRAE_REPAIR:
                return self.state()
            patch, patch_check, patch_check_event = await self._execute_code_repair(
                recovery, proposal_event.event_id
            )
            if self._stop_if_runtime_budget_exhausted():
                return self.state()

        stage_queue: Deque[Fidelity] = deque(spec.fidelity_plan)
        execution_attempt = 0
        runtime_settings = {}
        next_request_template = None
        next_execution_cause = None
        while stage_queue:
            fidelity = stage_queue.popleft()
            # Execution artifacts are keyed by experiment + attempt.  Attempts
            # therefore increase across fidelities and retries; resetting them
            # at each fidelity would make proxy/full outputs collide.
            execution_attempt += 1
            attempt = execution_attempt
            if next_request_template is not None:
                request = RunRequest.model_validate(
                    {
                        **next_request_template.model_dump(mode="json"),
                        "attempt": attempt,
                        "fidelity": fidelity,
                        "patch_commit_sha": patch_check.patch_commit_sha,
                        "patch_receipt_id": patch_check.receipt_id,
                        "runtime_settings": runtime_settings,
                    }
                )
                next_request_template = None
            else:
                request = RunRequest(
                    run_id=self.config.run_id,
                    experiment_id=spec.experiment_id,
                    attempt=attempt,
                    fidelity=fidelity,
                    command_id=self._command_for(fidelity),
                    patch_commit_sha=patch_check.patch_commit_sha,
                    patch_receipt_id=patch_check.receipt_id,
                    seed=self._execution_seed(
                        spec.experiment_id,
                        fidelity,
                        patch_check.patch_commit_sha,
                    ),
                    data_manifest_sha256=self.config.data_manifest_sha256,
                    timeout_seconds=self.config.timeout_profiles.get("standard", 600),
                    memory_limit_mb=4096,
                    gpu_memory_limit_mb=0,
                    network_enabled=False,
                    runtime_settings=runtime_settings,
                )
            started = self._append(
                ExecutionStartedPayload(request=request),
                stage="execution_started_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=(next_execution_cause or patch_check_event.event_id),
            )
            next_execution_cause = None
            try:
                result = await self.runner.run(request, self.health_observer)
            except Exception as error:
                failure_event = self._record_adapter_failure(
                    experiment_id=spec.experiment_id,
                    attempt=attempt,
                    stage="execution",
                    error=error,
                    causation_event_id=started.event_id,
                )
                action, recovery = await self._recover(
                    failure_event,
                    failure_event.payload.result,
                    spec.experiment_id,
                )
                decision, _, decision_event = recovery
                if action in (
                    RecoveryAction.RETRY_SAME_COMMIT,
                    RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING,
                ):
                    self._validate_runtime_adjustments(decision.runtime_adjustments)
                    runtime_settings.update(decision.runtime_adjustments)
                    next_request_template = request
                    if "timeout_profile" in decision.runtime_adjustments:
                        next_request_template = request.model_copy(
                            update={
                                "timeout_seconds": self.config.timeout_profiles[
                                    decision.runtime_adjustments["timeout_profile"]
                                ]
                            }
                        )
                    next_execution_cause = decision_event.event_id
                    stage_queue.appendleft(fidelity)
                    continue
                if action == RecoveryAction.TRAE_REPAIR:
                    while action == RecoveryAction.TRAE_REPAIR:
                        patch, patch_check, patch_check_event = (
                            await self._execute_code_repair(
                                recovery, proposal_event.event_id
                            )
                        )
                        if patch_check.accepted:
                            next_request_template = request
                            next_execution_cause = patch_check_event.event_id
                            stage_queue.appendleft(fidelity)
                            break
                        action, recovery = await self._recover(
                            patch_check_event,
                            patch_check,
                            spec.experiment_id,
                        )
                    if patch_check.accepted:
                        continue
                return self.state()
            finished = self._append(
                ExecutionFinishedPayload(result=result),
                stage="execution_finished_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=started.event_id,
                resource_delta=result.resource_delta,
            )
            if self._stop_if_runtime_budget_exhausted():
                return self.state()
            if result.outcome != RunOutcome.SUCCESS:
                action, recovery = await self._recover(finished, result, spec.experiment_id)
                decision, _, decision_event = recovery
                if action in (
                    RecoveryAction.RETRY_SAME_COMMIT,
                    RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING,
                ):
                    self._validate_runtime_adjustments(
                        decision.runtime_adjustments
                    )
                    runtime_settings.update(decision.runtime_adjustments)
                    next_request_template = request
                    if "timeout_profile" in decision.runtime_adjustments:
                        next_request_template = request.model_copy(
                            update={
                                "timeout_seconds": self.config.timeout_profiles[
                                    decision.runtime_adjustments["timeout_profile"]
                                ]
                            }
                        )
                    next_execution_cause = decision_event.event_id
                    stage_queue.appendleft(fidelity)
                    continue
                if action == RecoveryAction.TRAE_REPAIR:
                    while action == RecoveryAction.TRAE_REPAIR:
                        patch, patch_check, patch_check_event = (
                            await self._execute_code_repair(
                                recovery, proposal_event.event_id
                            )
                        )
                        if self._stop_if_runtime_budget_exhausted():
                            return self.state()
                        if patch_check.accepted:
                            next_request_template = request
                            next_execution_cause = patch_check_event.event_id
                            stage_queue.appendleft(fidelity)
                            break
                        action, recovery = await self._recover(
                            patch_check_event,
                            patch_check,
                            spec.experiment_id,
                        )
                    if patch_check.accepted:
                        continue
                return self.state()

            try:
                output = await self.output_gate.check(result)
            except Exception as error:
                failure_event = self._record_adapter_failure(
                    experiment_id=spec.experiment_id,
                    attempt=attempt,
                    stage="output_gate",
                    error=error,
                    causation_event_id=finished.event_id,
                )
                action, recovery = await self._recover(
                    failure_event,
                    failure_event.payload.result,
                    spec.experiment_id,
                )
                if action == RecoveryAction.TRAE_REPAIR:
                    patch, patch_check, patch_check_event = (
                        await self._execute_code_repair(
                            recovery, proposal_event.event_id
                        )
                    )
                    if patch_check.accepted:
                        next_request_template = request
                        next_execution_cause = patch_check_event.event_id
                        stage_queue.appendleft(fidelity)
                        continue
                return self.state()
            output_event = self._append(
                OutputCheckedPayload(result=output),
                stage="output_checked_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=finished.event_id,
                resource_delta=output.resource_delta,
            )
            if not output.accepted:
                action, recovery = await self._recover(
                    output_event, output, spec.experiment_id
                )
                repair_accepted = False
                while action == RecoveryAction.TRAE_REPAIR:
                    patch, patch_check, patch_check_event = (
                        await self._execute_code_repair(
                            recovery, proposal_event.event_id
                        )
                    )
                    if self._stop_if_runtime_budget_exhausted():
                        return self.state()
                    if patch_check.accepted:
                        repair_accepted = True
                        next_request_template = request
                        next_execution_cause = patch_check_event.event_id
                        stage_queue.appendleft(fidelity)
                        break
                    action, recovery = await self._recover(
                        patch_check_event, patch_check, spec.experiment_id
                    )
                if repair_accepted:
                    continue
                return self.state()

            if fidelity == Fidelity.SMOKE:
                next_fidelity = stage_queue[0] if stage_queue else Fidelity.PROXY
                decision = ExperimentDecision(
                    run_id=self.config.run_id,
                    experiment_id=spec.experiment_id,
                    evaluation_event_id=None,
                    decision=ExperimentDecisionKind.PROMOTE,
                    reason_code="smoke_output_verified",
                    fidelity_completed=Fidelity.SMOKE,
                    parent_eligible=True,
                    best_eligible=False,
                    next_fidelity=next_fidelity,
                )
                decision_event = self._append(
                    ExperimentDecidedPayload(decision=decision),
                    stage="decision_smoke",
                    experiment_id=spec.experiment_id,
                    attempt=attempt,
                    causation_event_id=output_event.event_id,
                )
                continue

            population = (
                Population.INTERNAL_PROXY
                if fidelity == Fidelity.PROXY
                else Population.PUBLIC_VALIDATION
            )
            state = self.state()
            baseline = self._baseline_metrics()
            parent = self._reference_metrics(spec.parent_experiment_id, fidelity)
            best = self._reference_metrics(state.best_experiment_id, fidelity)
            evaluation_request = EvaluationRequest(
                run_id=self.config.run_id,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                output_checked_event_id=output_event.event_id,
                prediction_artifact=output.prediction_artifact,
                population=population,
                fidelity=fidelity,
                seed=request.seed,
                contract_sha256=self.verified_contract.contract_sha256,
                evaluator_sha256=self.config.evaluator_sha256,
                baseline_summary=baseline,
                parent_summary=parent,
                previous_best_summary=best,
                public_query_index=(
                    state.public_validation_queries + 1
                    if population == Population.PUBLIC_VALIDATION
                    else None
                ),
            )
            try:
                evaluation = await self.evaluator.evaluate(evaluation_request)
            except Exception as error:
                failure_event = self._record_adapter_failure(
                    experiment_id=spec.experiment_id,
                    attempt=attempt,
                    stage="evaluation",
                    error=error,
                    causation_event_id=output_event.event_id,
                )
                action, recovery = await self._recover(
                    failure_event,
                    failure_event.payload.result,
                    spec.experiment_id,
                )
                decision, _, decision_event = recovery
                if action in (
                    RecoveryAction.RETRY_SAME_COMMIT,
                    RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING,
                ):
                    self._validate_runtime_adjustments(decision.runtime_adjustments)
                    runtime_settings.update(decision.runtime_adjustments)
                    next_request_template = request
                    if "timeout_profile" in decision.runtime_adjustments:
                        next_request_template = request.model_copy(
                            update={
                                "timeout_seconds": self.config.timeout_profiles[
                                    decision.runtime_adjustments["timeout_profile"]
                                ]
                            }
                        )
                    next_execution_cause = decision_event.event_id
                    stage_queue.appendleft(fidelity)
                    continue
                if action == RecoveryAction.TRAE_REPAIR:
                    patch, patch_check, patch_check_event = (
                        await self._execute_code_repair(
                            recovery, proposal_event.event_id
                        )
                    )
                    if patch_check.accepted:
                        next_request_template = request
                        next_execution_cause = patch_check_event.event_id
                        stage_queue.appendleft(fidelity)
                        continue
                return self.state()
            self.config.validate_metric_set(evaluation.metric_set)
            evaluation_event = self._append(
                EvaluationCompletedPayload(result=evaluation),
                stage="evaluation_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=output_event.event_id,
                resource_delta=evaluation.resource_delta,
            )
            if self._stop_if_runtime_budget_exhausted():
                return self.state()
            if evaluation.trust.verdict == TrustVerdict.NO_OP:
                action, recovery = await self._recover(
                    evaluation_event, evaluation, spec.experiment_id
                )
                repair_accepted = False
                while action == RecoveryAction.TRAE_REPAIR:
                    patch, patch_check, patch_check_event = (
                        await self._execute_code_repair(
                            recovery, proposal_event.event_id
                        )
                    )
                    if self._stop_if_runtime_budget_exhausted():
                        return self.state()
                    if patch_check.accepted:
                        repair_accepted = True
                        next_request_template = request
                        next_execution_cause = patch_check_event.event_id
                        stage_queue.appendleft(fidelity)
                        break
                    action, recovery = await self._recover(
                        patch_check_event, patch_check, spec.experiment_id
                    )
                if repair_accepted:
                    continue
                return self.state()
            decision_context = EvaluationDecisionContext(
                run_id=self.config.run_id,
                experiment_id=spec.experiment_id,
                baseline_score=baseline[self.config.primary_metric_name],
                parent_score=parent[self.config.primary_metric_name],
                previous_best_score=best[self.config.primary_metric_name],
                # Clean uncertain proxy results are evidence-bearing candidates.
                # Promotion is based on trust, never the experiment sequence number.
                promote_inconclusive_proxy=True,
            )
            try:
                decision = await self.evaluator.decide(evaluation, decision_context)
            except Exception as error:
                failure_event = self._record_adapter_failure(
                    experiment_id=spec.experiment_id,
                    attempt=attempt,
                    stage="evaluation",
                    error=error,
                    causation_event_id=evaluation_event.event_id,
                )
                action, recovery = await self._recover(
                    failure_event,
                    failure_event.payload.result,
                    spec.experiment_id,
                )
                decision, _, decision_event = recovery
                if action in (
                    RecoveryAction.RETRY_SAME_COMMIT,
                    RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING,
                ):
                    self._validate_runtime_adjustments(decision.runtime_adjustments)
                    runtime_settings.update(decision.runtime_adjustments)
                    next_request_template = request
                    if "timeout_profile" in decision.runtime_adjustments:
                        next_request_template = request.model_copy(
                            update={
                                "timeout_seconds": self.config.timeout_profiles[
                                    decision.runtime_adjustments["timeout_profile"]
                                ]
                            }
                        )
                    next_execution_cause = decision_event.event_id
                    stage_queue.appendleft(fidelity)
                    continue
                if action == RecoveryAction.TRAE_REPAIR:
                    patch, patch_check, patch_check_event = (
                        await self._execute_code_repair(
                            recovery, proposal_event.event_id
                        )
                    )
                    if patch_check.accepted:
                        next_request_template = request
                        next_execution_cause = patch_check_event.event_id
                        stage_queue.appendleft(fidelity)
                        continue
                return self.state()
            decision = decision.__class__.model_validate(
                {
                    **decision.model_dump(mode="json"),
                    "evaluation_event_id": evaluation_event.event_id,
                }
            )
            decision_event = self._append(
                ExperimentDecidedPayload(decision=decision),
                stage="decision_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=evaluation_event.event_id,
                resource_delta=decision.resource_delta,
            )
            if decision.lesson_candidate is not None:
                lesson_number = 1 + sum(
                    event.payload.type == "lesson.recorded" for event in self.events()
                )
                self._append(
                    LessonRecordedPayload(
                        lesson_id="lesson_%03d" % lesson_number,
                        candidate=decision.lesson_candidate,
                    ),
                    stage="lesson_recorded",
                    experiment_id=spec.experiment_id,
                    attempt=attempt,
                    causation_event_id=decision_event.event_id,
                )
            if decision.decision == ExperimentDecisionKind.PROMOTE:
                assert decision.next_fidelity is not None
                if not stage_queue or stage_queue[0] != decision.next_fidelity:
                    stage_queue.appendleft(decision.next_fidelity)
                continue
            if decision.best_eligible:
                current = self.state()
                aggregate_score = (
                    evaluation.trust.seed_mean
                    if evaluation.trust.seed_mean is not None
                    else evaluation.metric_set.primary_score
                )
                if (
                    current.best_primary_score is None
                    or aggregate_score > current.best_primary_score
                ):
                    self._append(
                        BestUpdatedPayload(
                            experiment_id=spec.experiment_id,
                            commit_sha=patch.patch_commit_sha,
                            primary_metric_name=evaluation.metric_set.primary_metric_name,
                            primary_score=aggregate_score,
                            decision_event_id=decision_event.event_id,
                        ),
                        stage="best_updated",
                        experiment_id=spec.experiment_id,
                        attempt=attempt,
                        causation_event_id=decision_event.event_id,
                    )
            return self.state()

        return self.state()

    async def run_until_stopped(self) -> object:
        """Run sequential research iterations until a frozen stop rule matches.

        Every iteration starts from a durable planning checkpoint and consumes
        the ledger-derived context produced by the preceding iteration.  This
        keeps the outer loop deterministic while the planner and coding worker
        remain replaceable adapters.
        """

        if not self.events():
            raise OrchestrationError("run must be bootstrapped before execution")
        while True:
            state = self.state()
            if state.status.value in {"stopped", "finalizing", "finalized", "failed"}:
                return state
            if state.phase not in {"planning", "planner_context"}:
                raise OrchestrationError(
                    "run can resume only from a durable planning checkpoint; "
                    "current phase is %s" % state.phase
                )
            previous_head = state.last_event_id
            state = await self.run_one_experiment()
            if state.status.value == "stopped":
                return state
            if state.last_event_id == previous_head:
                raise OrchestrationError("outer loop made no durable ledger progress")

    async def run_to_completion(self) -> object:
        """Drive research to a deterministic stop and finalize its selected best."""

        await self.run_until_stopped()
        return await self.finalize()

    async def _run_final_execution(
        self,
        *,
        experiment_id: str,
        commit_sha: str,
        receipt_id: str,
        seed: int,
        attempt: int,
        command_id: str,
        causation_event_id: str,
    ):
        request = RunRequest(
            run_id=self.config.run_id,
            experiment_id=experiment_id,
            attempt=attempt,
            fidelity=Fidelity.FULL,
            command_id=command_id,
            patch_commit_sha=commit_sha,
            patch_receipt_id=receipt_id,
            seed=seed,
            data_manifest_sha256=self.config.data_manifest_sha256,
            timeout_seconds=self.config.timeout_profiles.get("standard", 600),
            memory_limit_mb=4096,
            gpu_memory_limit_mb=0,
            network_enabled=False,
        )
        started = self._append(
            ExecutionStartedPayload(request=request),
            stage="final_%s_started" % command_id,
            experiment_id=experiment_id,
            attempt=attempt,
            causation_event_id=causation_event_id,
        )
        result = await self.runner.run(request, self.health_observer)
        finished = self._append(
            ExecutionFinishedPayload(result=result),
            stage="final_%s_finished" % command_id,
            experiment_id=experiment_id,
            attempt=attempt,
            causation_event_id=started.event_id,
            resource_delta=result.resource_delta,
        )
        if result.outcome != RunOutcome.SUCCESS:
            raise FinalizationError(
                "%s failed: %s"
                % (
                    command_id,
                    result.error_summary or result.error_class or result.outcome.value,
                )
            )
        return request, result, finished

    async def _check_final_output(self, result, finished, command_id: str):
        output = await self.output_gate.check(result)
        checked = self._append(
            OutputCheckedPayload(result=output),
            stage="final_%s_output_checked" % command_id,
            experiment_id=result.experiment_id,
            attempt=result.attempt,
            causation_event_id=finished.event_id,
            resource_delta=output.resource_delta,
        )
        if not output.accepted:
            raise FinalizationError("%s output failed Gate B" % command_id)
        return output, checked

    async def finalize(self) -> object:
        """Select by protected evidence, reproduce, and check the submission."""

        events = self.events()
        state = project(events)
        if state.status.value == "finalized":
            return state
        if state.status.value != "stopped":
            raise FinalizationError("finalization requires a stopped run")

        candidates = finalization_candidates(events, state)
        finalists = rank_finalists(candidates)
        if not finalists or finalists[0].experiment_id == "baseline":
            return await self._finalize_baseline(
                events, state, causation_event_id=events[-1].event_id
            )

        preliminary = finalists[0]
        plan = candidate_finalization_plan(
            events, state, experiment_id=preliminary.experiment_id
        )
        stopped_event_id = events[-1].event_id
        _, reproduction_run, reproduction_finished = await self._run_final_execution(
            experiment_id=plan.experiment_id,
            commit_sha=plan.commit_sha,
            receipt_id=plan.patch_receipt_id,
            seed=plan.seed,
            attempt=plan.next_attempt,
            command_id="clean_reproduce",
            causation_event_id=stopped_event_id,
        )
        reproduction_output, reproduction_output_event = await self._check_final_output(
            reproduction_run, reproduction_finished, "clean_reproduce"
        )
        baseline = self._baseline_metrics()
        best_node = state.experiments[plan.experiment_id]
        parent = self._reference_metrics(
            best_node.parent_experiment_id, Fidelity.FULL
        )
        best = self._reference_metrics(plan.experiment_id, Fidelity.FULL)
        evaluation = await self.evaluator.evaluate(
            EvaluationRequest(
                run_id=self.config.run_id,
                experiment_id=plan.experiment_id,
                attempt=plan.next_attempt,
                output_checked_event_id=reproduction_output_event.event_id,
                prediction_artifact=reproduction_output.prediction_artifact,
                population=Population.PUBLIC_VALIDATION,
                fidelity=Fidelity.FULL,
                seed=plan.seed,
                contract_sha256=self.verified_contract.contract_sha256,
                evaluator_sha256=self.config.evaluator_sha256,
                baseline_summary=baseline,
                parent_summary=parent,
                previous_best_summary=best,
                public_query_index=self.state().public_validation_queries + 1,
            )
        )
        self.config.validate_metric_set(evaluation.metric_set)
        reproduction_event = self._append(
            EvaluationCompletedPayload(result=evaluation),
            stage="final_reproduction_evaluated",
            experiment_id=plan.experiment_id,
            attempt=plan.next_attempt,
            causation_event_id=reproduction_output_event.event_id,
            resource_delta=evaluation.resource_delta,
        )
        if not (
            evaluation.trust.verdict == TrustVerdict.ACCEPTED
            and evaluation.trust.integrity == Integrity.CLEAN
            and evaluation.metric_set.primary_score == plan.best_primary_score
        ):
            raise FinalizationError(
                "clean reproduction did not exactly reproduce the trusted best score"
            )

        reproduced = replace(preliminary, clean_reproduction_passed=True)
        baseline_candidate = next(
            candidate
            for candidate in candidates
            if candidate.experiment_id == "baseline"
        )
        strict_selection = select_final((baseline_candidate, reproduced))
        if strict_selection.experiment_id == "baseline":
            return await self._finalize_baseline(
                self.events(),
                self.state(),
                causation_event_id=reproduction_event.event_id,
            )

        final_attempt = plan.next_attempt + 1
        _, final_run, final_finished = await self._run_final_execution(
            experiment_id=plan.experiment_id,
            commit_sha=plan.commit_sha,
            receipt_id=plan.patch_receipt_id,
            seed=plan.seed,
            attempt=final_attempt,
            command_id="candidate_final_infer",
            causation_event_id=reproduction_event.event_id,
        )
        final_output, final_output_event = await self._check_final_output(
            final_run, final_finished, "candidate_final_infer"
        )

        submission_attempt = final_attempt + 1
        _, submission_run, submission_finished = await self._run_final_execution(
            experiment_id=plan.experiment_id,
            commit_sha=plan.commit_sha,
            receipt_id=plan.patch_receipt_id,
            seed=plan.seed,
            attempt=submission_attempt,
            command_id="submission_check",
            causation_event_id=final_output_event.event_id,
        )
        if (
            submission_run.prediction_artifact is None
            or submission_run.prediction_artifact.sha256
            != final_output.prediction_artifact.sha256
        ):
            raise FinalizationError(
                "submission checker did not consume the accepted final prediction"
            )
        submission_artifact = final_output.prediction_artifact.model_copy(
            update={
                "artifact_id": "submission_%s" % plan.experiment_id,
                "kind": ArtifactKind.SUBMISSION,
            }
        )
        checks = [
            CheckResult(
                name="gate_b_%s" % name,
                status=status,
            )
            for name, status in sorted(final_output.checks.items())
        ]
        checks.append(
            CheckResult(
                name="official_submission_check",
                status=CheckStatus.PASS,
            )
        )
        submission = SubmissionCheckedPayload(
            accepted=True,
            submission_artifact=submission_artifact,
            checks=checks,
        )
        selected = self._append(
            FinalSelectedPayload(
                experiment_id=plan.experiment_id,
                commit_sha=plan.commit_sha,
                reproduction_evaluation_event_id=reproduction_event.event_id,
            ),
            stage="final_selected",
            experiment_id=plan.experiment_id,
            causation_event_id=submission_finished.event_id,
        )
        self._append(
            submission,
            stage="submission_checked",
            experiment_id=plan.experiment_id,
            causation_event_id=selected.event_id,
        )
        return self.state()

    async def _finalize_baseline(
        self,
        events: Sequence[Event],
        state: object,
        *,
        causation_event_id: str,
    ) -> object:
        if self.final_submission_provider is None:
            raise FinalizationError(
                "selected baseline requires a protected final submission provider"
            )
        reproduction_event_id = baseline_reproduction_event_id(events, state)
        baseline = next(
            event
            for event in events
            if event.event_type == EventType.BASELINE_VERIFIED
        )
        submission = await self.final_submission_provider.prepare_baseline()
        selected = self._append(
            FinalSelectedPayload(
                experiment_id="baseline",
                commit_sha=baseline.payload.commit_sha,
                reproduction_evaluation_event_id=reproduction_event_id,
            ),
            stage="final_selected",
            experiment_id="baseline",
            causation_event_id=causation_event_id,
        )
        self._append(
            submission,
            stage="submission_checked",
            experiment_id="baseline",
            causation_event_id=selected.event_id,
        )
        return self.state()
