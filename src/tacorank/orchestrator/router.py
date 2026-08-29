"""Deterministic outer-loop routing across independently replaceable adapters."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Sequence, Tuple

from ..config import RunConfig, VerifiedContract
from ..context.builder import ContextBuilder
from ..memory.canonical_json import canonical_sha256
from ..memory.event_store import DuplicateIdempotencyKey, EventStore
from ..memory.projections import project
from ..recovery.fingerprints import fingerprint_result
from ..schemas import (
    BaselineVerifiedPayload,
    BestUpdatedPayload,
    ContextCreatedPayload,
    ContractVerifiedPayload,
    EvaluationCompletedPayload,
    EvaluationDecisionContext,
    EvaluationRequest,
    Event,
    ExperimentDecidedPayload,
    ExperimentDecision,
    ExperimentDecisionKind,
    ExperimentProposedPayload,
    ExecutionFinishedPayload,
    ExecutionStartedPayload,
    Fidelity,
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
    TrustVerdict,
)
from .convergence import StopDecision, stop_decision
from .ports import (
    CodingWorker,
    Evaluator,
    ExecutionRunner,
    HealthObserver,
    OutputGate,
    PatchGate,
    RecoveryManager,
    ResearchPlanner,
)


class OrchestrationError(RuntimeError):
    pass


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
            return self.event_store.append(
                run_id=self.config.run_id,
                payload=payload,
                idempotency_key=key,
                causation_event_id=causation_event_id,
                resource_delta=resource_delta,
            )
        except DuplicateIdempotencyKey as duplicate:
            # The immutable input hash proves this is the same acknowledged action.
            return duplicate.event

    def events(self) -> Sequence[Event]:
        return self.event_store.read_events(repair_tail=True)

    def state(self):
        return project(self.events())

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
            ),
            stage="started",
        )
        contract = self._append(
            ContractVerifiedPayload(
                contract_sha256=self.verified_contract.contract_sha256,
                protected_paths_sha256=self.verified_contract.protected_paths_sha256,
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

    def _command_for(self, fidelity: Fidelity) -> str:
        preferred = "run_%s" % fidelity.value
        if preferred in self.config.command_ids:
            return preferred
        return self.config.command_ids[0]

    def _baseline_metrics(self) -> dict:
        baseline = next(
            event for event in self.events() if event.payload.type == "baseline.verified"
        )
        return dict(baseline.payload.metric_set.metrics)

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
                remaining_repair_budget=remaining,
                previous_error_fingerprints=self._previous_failure_fingerprints(
                    experiment_id, failure_event.event_id
                ),
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
        return decision.action, (decision, context, decision_event)

    async def run_one_experiment(self) -> object:
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
            return self.state()

        spec = planner_output.spec
        assert spec is not None
        if spec.context_id != planner_context.context_id:
            raise OrchestrationError("planner proposal cites a different context")
        proposal_event = self._append(
            ExperimentProposedPayload(spec=spec),
            stage="proposed",
            experiment_id=spec.experiment_id,
            causation_event_id=planner_context_event.event_id,
            resource_delta=planner_output.resource_delta,
        )
        coder_context = self.context_builder.build_coder(self.events(), spec)
        coder_context_event = self._append(
            ContextCreatedPayload(context=coder_context),
            stage="coder_context",
            experiment_id=spec.experiment_id,
            attempt=1,
            causation_event_id=proposal_event.event_id,
        )
        patch = await self.coding_worker.create_patch(coder_context, spec)
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
            causation_event_id=coder_context_event.event_id,
            resource_delta=patch.resource_delta,
        )

        patch_check = await self.patch_gate.check(patch)
        patch_check_event = self._append(
            PatchCheckedPayload(result=patch_check),
            stage="patch_checked",
            experiment_id=spec.experiment_id,
            attempt=patch.attempt,
            causation_event_id=patch_event.event_id,
            resource_delta=patch_check.resource_delta,
        )
        if not patch_check.accepted:
            action, recovery = await self._recover(
                patch_check_event, patch_check, spec.experiment_id
            )
            if action != RecoveryAction.ABANDON:
                raise OrchestrationError(
                    "P0 supports successful patches and deterministic abandon; "
                    "repair adapters are integrated in P2"
                )
            return self.state()

        stage_queue: Deque[Fidelity] = deque(spec.fidelity_plan)
        attempts = {fidelity: 0 for fidelity in Fidelity}
        while stage_queue:
            fidelity = stage_queue.popleft()
            attempts[fidelity] += 1
            attempt = attempts[fidelity]
            seed_index = min(sum(attempts.values()) - 1, len(self.config.seed_schedule) - 1)
            request = RunRequest(
                run_id=self.config.run_id,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                fidelity=fidelity,
                command_id=self._command_for(fidelity),
                patch_commit_sha=patch_check.patch_commit_sha,
                patch_receipt_id=patch_check.receipt_id,
                seed=self.config.seed_schedule[seed_index],
                data_manifest_sha256=self.config.data_manifest_sha256,
                timeout_seconds=600,
                memory_limit_mb=4096,
                gpu_memory_limit_mb=0,
                network_enabled=False,
            )
            started = self._append(
                ExecutionStartedPayload(request=request),
                stage="execution_started_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=patch_check_event.event_id,
            )
            result = await self.runner.run(request, self.health_observer)
            finished = self._append(
                ExecutionFinishedPayload(result=result),
                stage="execution_finished_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=started.event_id,
                resource_delta=result.resource_delta,
            )
            if result.outcome != RunOutcome.SUCCESS:
                action, recovery = await self._recover(finished, result, spec.experiment_id)
                if action == RecoveryAction.RETRY_SAME_COMMIT:
                    stage_queue.appendleft(fidelity)
                    continue
                return self.state()

            output = await self.output_gate.check(result)
            output_event = self._append(
                OutputCheckedPayload(result=output),
                stage="output_checked_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=finished.event_id,
                resource_delta=output.resource_delta,
            )
            if not output.accepted:
                await self._recover(output_event, output, spec.experiment_id)
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
            best = dict(baseline)
            if state.best_experiment_id in state.experiments:
                best_node = state.experiments[state.best_experiment_id]
                if best_node.metric_set:
                    best = dict(best_node.metric_set.metrics)
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
                parent_summary=best,
                previous_best_summary=best,
                public_query_index=(
                    state.public_validation_queries + 1
                    if population == Population.PUBLIC_VALIDATION
                    else None
                ),
            )
            evaluation = await self.evaluator.evaluate(evaluation_request)
            self.config.validate_metric_set(evaluation.metric_set)
            evaluation_event = self._append(
                EvaluationCompletedPayload(result=evaluation),
                stage="evaluation_%s" % fidelity.value,
                experiment_id=spec.experiment_id,
                attempt=attempt,
                causation_event_id=output_event.event_id,
                resource_delta=evaluation.resource_delta,
            )
            decision_context = EvaluationDecisionContext(
                run_id=self.config.run_id,
                experiment_id=spec.experiment_id,
                baseline_score=baseline[self.config.primary_metric_name],
                parent_score=best[self.config.primary_metric_name],
                previous_best_score=best[self.config.primary_metric_name],
            )
            decision = await self.evaluator.decide(evaluation, decision_context)
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
                if (
                    current.best_primary_score is None
                    or evaluation.metric_set.primary_score > current.best_primary_score
                ):
                    self._append(
                        BestUpdatedPayload(
                            experiment_id=spec.experiment_id,
                            commit_sha=patch.patch_commit_sha,
                            primary_metric_name=evaluation.metric_set.primary_metric_name,
                            primary_score=evaluation.metric_set.primary_score,
                            decision_event_id=decision_event.event_id,
                        ),
                        stage="best_updated",
                        experiment_id=spec.experiment_id,
                        attempt=attempt,
                        causation_event_id=decision_event.event_id,
                    )
            return self.state()

        return self.state()
