"""Typed adapter protocols at the deterministic harness boundary."""

from __future__ import annotations

from typing import Protocol, Union

from ..schemas import (
    CoderContext,
    EvaluationDecisionContext,
    EvaluationRequest,
    EvaluationResult,
    ExperimentDecision,
    ExperimentSpec,
    MonitorDirective,
    OutputCheckResult,
    PatchCandidate,
    PatchCheckResult,
    PlannerContext,
    PlannerOutput,
    RecoveryContext,
    RecoveryDecision,
    RecoveryPolicyContext,
    RunRequest,
    RunResult,
    SubmissionCheckedPayload,
    TelemetrySample,
)


class ResearchPlanner(Protocol):
    async def propose(self, context: PlannerContext) -> PlannerOutput:
        ...

    def parallel_direction_capacity(self, context: PlannerContext) -> int:
        ...

    async def propose_parallel_direction(
        self, context: PlannerContext, direction_index: int, direction_count: int
    ) -> PlannerOutput:
        ...

    async def propose_synthesis(
        self, context: PlannerContext, component_experiment_ids: list[str]
    ) -> PlannerOutput:
        ...


class CodingWorker(Protocol):
    async def create_patch(
        self, context: CoderContext, spec: ExperimentSpec
    ) -> PatchCandidate:
        ...

    async def repair_patch(
        self, context: RecoveryContext, decision: RecoveryDecision
    ) -> PatchCandidate:
        ...


class PatchGate(Protocol):
    async def check(self, candidate: PatchCandidate) -> PatchCheckResult:
        ...


class HealthObserver(Protocol):
    def observe(self, sample: TelemetrySample) -> MonitorDirective:
        ...


class ExecutionRunner(Protocol):
    async def run(self, request: RunRequest, observer: HealthObserver) -> RunResult:
        ...


RecoverableResult = Union[PatchCheckResult, RunResult, OutputCheckResult, EvaluationResult]


class RecoveryManager(Protocol):
    async def decide(
        self,
        failure_event_id: str,
        result: RecoverableResult,
        context: RecoveryPolicyContext,
    ) -> RecoveryDecision:
        ...


class OutputGate(Protocol):
    async def check(self, result: RunResult) -> OutputCheckResult:
        ...


class Evaluator(Protocol):
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        ...

    async def decide(
        self, result: EvaluationResult, context: EvaluationDecisionContext
    ) -> ExperimentDecision:
        ...


class FinalSubmissionProvider(Protocol):
    """Prepare a protected baseline submission without appending ledger state."""

    async def prepare_baseline(self) -> SubmissionCheckedPayload:
        ...
