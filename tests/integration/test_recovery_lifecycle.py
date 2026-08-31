from __future__ import annotations

import asyncio

from tacorank.memory.replay import replay
from tacorank.coding.trae_adapter import CodingWorkerError
from tacorank.context.builder import ContextBuildError
from tacorank.orchestrator.fakes import FakeCodingWorker, FakeExecutionRunner
from tacorank.orchestrator.state import ExperimentStatus
from tacorank.recovery.policy import RecoveryManager
from tacorank.schemas import (
    EventType,
    PlannerAction,
    PlannerOutput,
    RecoveryAction,
    RunOutcome,
)


class RepairingCodingWorker(FakeCodingWorker):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.initial_patch = None
        self.repair_calls = []

    async def create_patch(self, context, spec):
        self.initial_patch = await super().create_patch(context, spec)
        return self.initial_patch

    async def repair_patch(self, context, decision):
        self.repair_calls.append((context, decision))
        values = self.initial_patch.model_dump(mode="json")
        values.update(
            {
                "context_id": context.context_id,
                "base_commit_sha": self.initial_patch.patch_commit_sha,
                "patch_commit_sha": "c" * 40,
            }
        )
        return self.initial_patch.__class__.model_validate(values)


class FailingRepairWorker(RepairingCodingWorker):
    async def repair_patch(self, context, decision):
        self.repair_calls.append((context, decision))
        raise RuntimeError("TRAE_REPORTED_FAILURE: trajectory was unsuccessful")


class IntegrityRestartingCodingWorker(RepairingCodingWorker):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.restart_calls = []

    async def restart_from_trusted_parent(self, context, decision):
        self.restart_calls.append((context, decision))
        values = self.initial_patch.model_dump(mode="json")
        values.update(
            {
                "context_id": context.context_id,
                "base_commit_sha": context.original_experiment_spec.parent_commit_sha,
                "patch_commit_sha": "d" * 40,
            }
        )
        return self.initial_patch.__class__.model_validate(values)


class IntegrityRejectOncePatchGate(FakePatchGate):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.calls = 0

    async def check(self, candidate):
        self.calls += 1
        if self.calls == 1:
            return PatchCheckResult(
                run_id=candidate.run_id,
                experiment_id=candidate.experiment_id,
                attempt=candidate.attempt,
                patch_commit_sha=candidate.patch_commit_sha,
                diff_sha256=candidate.diff_sha256,
                accepted=False,
                checks=[
                    CheckResult(
                        name="protected_path",
                        status=CheckStatus.FAIL,
                        summary="candidate patch touches a protected path",
                    )
                ],
                violations=[
                    Violation(
                        code="PROTECTED_PATH_MODIFIED",
                        message="candidate patch touches a protected path",
                    )
                ],
            )
        return await super().check(candidate)


class TransientInitialCodingWorker(FakeCodingWorker):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.calls = 0
        self.contexts = []

    async def create_patch(self, context, spec):
        self.calls += 1
        self.contexts.append(context)
        if self.calls == 1:
            raise CodingWorkerError(
                "TRAE_LAUNCH_FAILED",
                "failed to launch Trae",
                output_tail="authorization=sk-test-secret\nlaunch failed",
            )
        return await super().create_patch(context, spec)


class RepeatedTransientInitialCodingWorker(FakeCodingWorker):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.calls = 0

    async def create_patch(self, context, spec):
        self.calls += 1
        raise CodingWorkerError(
            "TRAE_PROVIDER_UNAVAILABLE",
            "coding provider is temporarily unavailable",
        )


class PermanentInitialCodingWorker(FakeCodingWorker):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.calls = 0

    async def create_patch(self, context, spec):
        self.calls += 1
        raise CodingWorkerError(
            "SOLUTION_VERIFICATION_FAILED",
            "candidate did not implement the approved plan after bounded reviews",
        )


class ProposeThenBlockPlanner:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0

    async def propose(self, context):
        self.calls += 1
        if self.calls == 1:
            return await self.delegate.propose(context)
        return PlannerOutput(
            action=PlannerAction.BLOCKED,
            reason_code="portfolio_exhausted",
            reason="No reviewed non-duplicate method remains.",
            supporting_event_ids=context.source_event_ids,
        )


def test_coder_context_failure_is_recorded_at_the_coding_boundary(
    harness, baseline_evaluation, monkeypatch
):
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    def fail_coder_context(*args, **kwargs):
        del args, kwargs
        raise ContextBuildError("mandatory coder context exceeds configured budget")

    monkeypatch.setattr(harness.context_builder, "build_coder", fail_coder_context)

    state = asyncio.run(harness.run_one_experiment())

    failure = next(
        event
        for event in harness.events()
        if event.event_type == EventType.ADAPTER_FAILED
    )
    assert failure.payload.result.failure_stage == "coding"
    assert state.stop_reason_code is None
    assert state.phase == "planning"
    assert state.current_experiment.status.value == "invalid"


class FailOnceRunner(FakeExecutionRunner):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.requests = []

    async def run(self, request, observer):
        self.requests.append(request)
        result = await super().run(request, observer)
        if len(self.requests) == 1:
            return result.__class__.model_validate(
                {
                    **result.model_dump(mode="json"),
                    "outcome": "code_error",
                    "exit_code": 1,
                    "error_class": "NameError",
                    "error_fingerprint": "a" * 64,
                    "error_summary": "NameError in solution/model.py",
                    "prediction_artifact": None,
                }
            )
        return result


class RaisingOnceRunner(FakeExecutionRunner):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.requests = []

    async def run(self, request, observer):
        self.requests.append(request)
        if len(self.requests) == 1:
            raise RuntimeError("temporary execution backend loss")
        return await super().run(request, observer)


class TwoTransientFailuresRunner(FakeExecutionRunner):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.requests = []

    async def run(self, request, observer):
        self.requests.append(request)
        result = await super().run(request, observer)
        if len(self.requests) <= 2:
            outcome = (
                RunOutcome.INFRASTRUCTURE_ERROR
                if len(self.requests) == 1
                else RunOutcome.HANG
            )
            return result.__class__.model_validate(
                {
                    **result.model_dump(mode="json"),
                    "outcome": outcome.value,
                    "exit_code": 1,
                    "error_class": outcome.value,
                    "error_fingerprint": ("%x" % len(self.requests)) * 64,
                    "error_summary": "%s with distinct evidence" % outcome.value,
                    "prediction_artifact": None,
                }
            )
        return result


class OomOnceRunner(FakeExecutionRunner):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.requests = []

    async def run(self, request, observer):
        self.requests.append(request)
        result = await super().run(request, observer)
        if len(self.requests) == 1:
            return result.__class__.model_validate(
                {
                    **result.model_dump(mode="json"),
                    "outcome": "oom",
                    "exit_code": 1,
                    "error_class": "oom",
                    "error_summary": "CUDA out of memory",
                    "prediction_artifact": None,
                }
            )
        return result


class RaisingOnceEvaluator:
    def __init__(self, delegate):
        self.delegate = delegate
        self.evaluate_requests = []

    async def evaluate(self, request):
        self.evaluate_requests.append(request)
        if len(self.evaluate_requests) == 1:
            raise CodingWorkerError(
                "EVALUATOR_PROVIDER_UNAVAILABLE",
                "evaluation provider is temporarily unavailable",
            )
        return await self.delegate.evaluate(request)

    async def decide(self, result, context):
        return await self.delegate.decide(result, context)


class RaisingOnceDecisionEvaluator:
    def __init__(self, delegate):
        self.delegate = delegate
        self.decide_inputs = []

    async def evaluate(self, request):
        return await self.delegate.evaluate(request)

    async def decide(self, result, context):
        self.decide_inputs.append((result, context))
        if len(self.decide_inputs) == 1:
            raise CodingWorkerError(
                "EVALUATOR_PROVIDER_UNAVAILABLE",
                "evaluation decision provider is temporarily unavailable",
            )
        return await self.delegate.decide(result, context)


class RaisingOncePatchGate:
    def __init__(self, delegate):
        self.delegate = delegate
        self.candidates = []

    async def check(self, candidate):
        self.candidates.append(candidate)
        if len(self.candidates) == 1:
            raise CodingWorkerError(
                "PROVIDER_UNAVAILABLE",
                "patch gate provider is temporarily unavailable",
            )
        return await self.delegate.check(candidate)


class RaisingOnceOutputGate:
    def __init__(self, delegate):
        self.delegate = delegate
        self.results = []

    async def check(self, result):
        self.results.append(result)
        if len(self.results) == 1:
            raise CodingWorkerError(
                "PROVIDER_UNAVAILABLE",
                "output gate provider is temporarily unavailable",
            )
        return await self.delegate.check(result)


def test_first_attempt_stage_results_keep_direct_stage_causation(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    events = harness.events()
    by_id = {event.event_id: event for event in events}
    expected_parent_types = {
        EventType.PATCH_CHECKED: {EventType.PATCH_CREATED},
        EventType.OUTPUT_CHECKED: {EventType.EXECUTION_FINISHED},
        EventType.EVALUATION_COMPLETED: {EventType.OUTPUT_CHECKED},
        EventType.EXPERIMENT_DECIDED: {
            EventType.OUTPUT_CHECKED,
            EventType.EVALUATION_COMPLETED,
        },
    }
    for event in events:
        if event.event_type not in expected_parent_types:
            continue
        cause = by_id[event.causation_event_id]
        assert cause.event_type in expected_parent_types[event.event_type]


def test_real_recovery_repairs_gates_reruns_and_replays(
    harness, baseline_evaluation
):
    artifacts = harness.event_store.artifact_store
    worker = RepairingCodingWorker(artifacts)
    runner = FailOnceRunner(artifacts)
    harness.coding_worker = worker
    harness.runner = runner
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.experiments["exp_001"].repair_count == 1
    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    assert len(worker.repair_calls) == 1
    context, recovery_decision = worker.repair_calls[0]
    assert context.repair_attempt == 1
    assert context.original_experiment_spec.experiment_id == "exp_001"
    assert context.current_patch_commit_sha == "a" * 40
    assert context.accepted_patch_receipt_id == "receipt_exp_001_1"
    assert context.failure_class == "code_error"
    assert context.recovery_instructions == recovery_decision.instructions
    assert context.failed_checks == []
    assert "Adding a deterministic user-item cross" in recovery_decision.instructions
    assert "solution/model.py" in recovery_decision.instructions
    assert recovery_decision.action == RecoveryAction.TRAE_REPAIR
    assert runner.requests[0].patch_commit_sha == "a" * 40
    assert runner.requests[1].patch_commit_sha == "c" * 40
    assert runner.requests[0].seed == runner.requests[1].seed

    events = harness.events()
    recovery_event = next(
        event for event in events if event.event_type == EventType.RECOVERY_DECIDED
    )
    repaired_patch_event = next(
        event
        for event in events
        if event.event_type == EventType.PATCH_CREATED
        and event.payload.candidate.patch_commit_sha == "c" * 40
    )
    assert repaired_patch_event.causation_event_id == recovery_event.event_id
    assert replay(events).experiments["exp_001"].repair_count == 1


def test_trae_exception_quarantines_candidate_and_returns_to_planning(
    harness, baseline_evaluation
):
    artifacts = harness.event_store.artifact_store
    worker = FailingRepairWorker(artifacts)
    harness.coding_worker = worker
    harness.runner = FailOnceRunner(artifacts)
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    events = harness.events()
    adapter_failures = [
        event for event in events if event.event_type == EventType.ADAPTER_FAILED
    ]
    decisions = [
        event.payload.decision
        for event in events
        if event.event_type == EventType.RECOVERY_DECIDED
    ]
    assert len(adapter_failures) == 1
    assert adapter_failures[0].payload.result.failure_stage == "coding"
    assert len(worker.repair_calls) == 1
    assert decisions[-1].action == RecoveryAction.ABANDON
    assert state.status.value == "running"
    assert state.phase == "planning"
    assert state.current_experiment.status == ExperimentStatus.INVALID
    assert replay(events).status.value == "running"


def test_transient_initial_coding_failure_retries_and_persists_redacted_tail(
    harness, baseline_evaluation
):
    worker = TransientInitialCodingWorker(harness.event_store.artifact_store)
    harness.coding_worker = worker
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert worker.calls == 2
    assert worker.contexts[0].owner_retry_error_summary is None
    assert "failed to launch Trae" in worker.contexts[1].owner_retry_error_summary
    assert "sk-test-secret" not in worker.contexts[1].owner_retry_error_summary
    assert "[REDACTED]" in worker.contexts[1].owner_retry_error_summary
    assert "Retry the same coding assignment once" in (
        worker.contexts[1].owner_retry_instructions
    )
    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    events = harness.events()
    failure = next(
        event for event in events if event.event_type == EventType.ADAPTER_FAILED
    )
    assert len(failure.payload.result.diagnostic_artifacts) == 1
    diagnostic = failure.payload.result.diagnostic_artifacts[0]
    content = (harness.config.repository_root / diagnostic.path).read_text(
        encoding="utf-8"
    )
    assert "sk-test-secret" not in content
    assert "[REDACTED]" in content
    recovery_event = next(
        event
        for event in events
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    decision = recovery_event.payload.decision
    assert decision.action == RecoveryAction.RETRY_SAME_COMMIT
    patch_event = next(
        event for event in events if event.event_type == EventType.PATCH_CREATED
    )
    assert patch_event.causation_event_id == recovery_event.event_id


def test_second_transient_initial_coding_failure_is_recorded_then_abandoned(
    harness, baseline_evaluation
) -> None:
    worker = RepeatedTransientInitialCodingWorker(
        harness.event_store.artifact_store
    )
    harness.coding_worker = worker
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert worker.calls == 2
    failures = [
        event
        for event in harness.events()
        if event.event_type == EventType.ADAPTER_FAILED
    ]
    decisions = [
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    ]
    assert len(failures) == 2
    assert [decision.action for decision in decisions] == [
        RecoveryAction.RETRY_SAME_COMMIT,
        RecoveryAction.ABANDON,
    ]
    assert state.experiments["exp_001"].status == ExperimentStatus.INVALID
    assert state.phase == "planning"
    assert state.status.value == "running"
    assert state.active_experiment_id is None


def test_permanent_initial_coding_failure_abandons_only_the_experiment(
    harness, baseline_evaluation
) -> None:
    worker = PermanentInitialCodingWorker(harness.event_store.artifact_store)
    harness.coding_worker = worker
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert worker.calls == 1
    decision = next(
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    assert decision.action == RecoveryAction.ABANDON
    assert state.experiments["exp_001"].status == ExperimentStatus.INVALID
    assert state.phase == "planning"
    assert state.status.value == "running"
    assert state.active_experiment_id is None


def test_outer_loop_replans_after_abandoned_candidate(
    harness, baseline_evaluation
) -> None:
    planner = ProposeThenBlockPlanner(harness.planner)
    harness.planner = planner
    harness.coding_worker = PermanentInitialCodingWorker(
        harness.event_store.artifact_store
    )
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_until_stopped())

    assert planner.calls == 2
    assert state.experiments["exp_001"].status == ExperimentStatus.INVALID
    assert state.stop_reason_code == "no_legal_proposal"
    assert [
        event.payload.reason_code
        for event in harness.events()
        if event.event_type == EventType.RUN_STOPPED
    ] == ["no_legal_proposal"]


def test_real_recovery_allows_only_one_same_commit_retry_across_fingerprints(
    harness, baseline_evaluation
):
    artifacts = harness.event_store.artifact_store
    runner = TwoTransientFailuresRunner(artifacts)
    harness.runner = runner
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    decisions = [
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    ]
    assert [decision.action for decision in decisions] == [
        RecoveryAction.RETRY_SAME_COMMIT,
        RecoveryAction.ABANDON,
    ]
    assert len(runner.requests) == 2
    assert runner.requests[0].seed == runner.requests[1].seed
    assert runner.requests[0].patch_commit_sha == runner.requests[1].patch_commit_sha
    assert state.experiments["exp_001"].same_commit_retry_count == 1
    assert state.experiments["exp_001"].status == ExperimentStatus.INVALID


def test_runner_exception_is_retried_as_infrastructure_failure(
    harness, baseline_evaluation
):
    runner = RaisingOnceRunner(harness.event_store.artifact_store)
    harness.runner = runner
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert len(runner.requests) == 6
    assert runner.requests[0].seed == runner.requests[1].seed
    assert runner.requests[0].patch_commit_sha == runner.requests[1].patch_commit_sha
    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    failure = next(
        event for event in harness.events() if event.event_type == EventType.ADAPTER_FAILED
    )
    assert failure.payload.result.failure_stage == "execution"
    decision = next(
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    assert decision.action == RecoveryAction.RETRY_SAME_COMMIT


def test_evaluator_exception_retries_evaluator_without_rerunning_candidate(
    harness, baseline_evaluation
):
    runner = FakeExecutionRunner(harness.event_store.artifact_store)
    evaluator = RaisingOnceEvaluator(harness.evaluator)
    harness.runner = runner
    harness.evaluator = evaluator
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())
    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    assert len(evaluator.evaluate_requests) >= 2
    assert evaluator.evaluate_requests[0] == evaluator.evaluate_requests[1]
    execution_starts = [
        event
        for event in harness.events()
        if event.event_type == EventType.EXECUTION_STARTED
    ]
    # One smoke, one proxy, and three full-seed executions: the transient
    # evaluator retry must not create an extra execution attempt.
    assert len(execution_starts) == 5
    failure = next(
        event
        for event in harness.events()
        if event.event_type == EventType.ADAPTER_FAILED
    )
    assert failure.payload.result.failure_stage == "evaluation"
    decision = next(
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    assert decision.action == RecoveryAction.RETRY_SAME_COMMIT
    recovery_event = next(
        event
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    evaluation_event = next(
        event
        for event in harness.events()
        if event.event_type == EventType.EVALUATION_COMPLETED
    )
    assert evaluation_event.causation_event_id == recovery_event.event_id


def test_patch_gate_exception_retries_gate_against_same_candidate(
    harness, baseline_evaluation
):
    gate = RaisingOncePatchGate(FakePatchGate(harness.event_store.artifact_store))
    harness.patch_gate = gate
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    assert len(gate.candidates) == 2
    assert gate.candidates[0] == gate.candidates[1]
    decision = next(
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    assert decision.action == RecoveryAction.RETRY_SAME_COMMIT
    assert decision.reason_code == "TRANSIENT_PATCH_GATE_RETRY"
    recovery_event = next(
        event
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    patch_check_event = next(
        event
        for event in harness.events()
        if event.event_type == EventType.PATCH_CHECKED
    )
    assert patch_check_event.causation_event_id == recovery_event.event_id


def test_evaluation_decision_exception_retries_decider_without_reevaluation(
    harness, baseline_evaluation
):
    evaluator = RaisingOnceDecisionEvaluator(harness.evaluator)
    harness.evaluator = evaluator
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    assert len(evaluator.decide_inputs) >= 2
    assert evaluator.decide_inputs[0] == evaluator.decide_inputs[1]
    decision = next(
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    assert decision.reason_code == "TRANSIENT_EVALUATOR_RETRY"
    recovery_event = next(
        event
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    experiment_decision = next(
        event
        for event in harness.events()
        if event.event_type == EventType.EXPERIMENT_DECIDED
        and event.causation_event_id == recovery_event.event_id
    )
    assert experiment_decision.causation_event_id == recovery_event.event_id


def test_output_gate_exception_retries_gate_without_rerunning_candidate(
    harness, baseline_evaluation
):
    runner = FakeExecutionRunner(harness.event_store.artifact_store)
    gate = RaisingOnceOutputGate(FakeOutputGate())
    harness.runner = runner
    harness.output_gate = gate
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    assert len(gate.results) == 6
    assert gate.results[0] == gate.results[1]
    execution_starts = [
        event
        for event in harness.events()
        if event.event_type == EventType.EXECUTION_STARTED
    ]
    assert len(execution_starts) == 5
    decision = next(
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    assert decision.action == RecoveryAction.RETRY_SAME_COMMIT
    assert decision.reason_code == "TRANSIENT_OUTPUT_GATE_RETRY"
    recovery_event = next(
        event
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    output_event = next(
        event
        for event in harness.events()
        if event.event_type == EventType.OUTPUT_CHECKED
    )
    assert output_event.causation_event_id == recovery_event.event_id


def test_approved_runtime_adjustment_is_applied_without_code_repair(
    harness, baseline_evaluation
):
    artifacts = harness.event_store.artifact_store
    runner = OomOnceRunner(artifacts)
    harness.config.allowed_runtime_adjustments = {
        "batch_size": {"next_value": 32}
    }
    harness.runner = runner
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.experiments["exp_001"].status == ExperimentStatus.ACCEPTED
    assert state.experiments["exp_001"].repair_count == 0
    assert runner.requests[1].runtime_settings == {"batch_size": 32}
    assert runner.requests[0].seed == runner.requests[1].seed
    decision = next(
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    assert decision.action == RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING


def test_unadjustable_oom_rolls_back_to_terminal_without_rerun(
    harness, baseline_evaluation
):
    artifacts = harness.event_store.artifact_store
    runner = OomOnceRunner(artifacts)
    harness.runner = runner
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert len(runner.requests) == 1
    assert state.experiments["exp_001"].status == ExperimentStatus.INVALID
    decision = next(
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    )
    assert decision.action == RecoveryAction.ROLLBACK
