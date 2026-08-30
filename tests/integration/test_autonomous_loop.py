from __future__ import annotations

import asyncio

import pytest

from tacorank.coding import CodingWorkerError
from tacorank.orchestrator.convergence import StopDecision
from tacorank.orchestrator.fakes import (
    FakeCodingWorker,
    FakeEvaluator,
    FakeExecutionRunner,
    FakeResearchPlanner,
)
from tacorank.orchestrator.router import ResumablePlanningError
from tacorank.orchestrator.state import ExperimentStatus
from tacorank.recovery import RecoveryManager
from tacorank.schemas import (
    ArtifactKind,
    CheckResult,
    CheckStatus,
    EventType,
    Stability,
    PlannerAction,
    PlannerOutput,
    PatchCheckResult,
    RecoveryAction,
    SubmissionCheckedPayload,
    TrustVerdict,
    Violation,
)


class SequentialPlanner:
    def __init__(self, parent_commit_sha: str) -> None:
        self.parent_commit_sha = parent_commit_sha
        self.contexts = []

    async def propose(self, context):
        self.contexts.append(context)
        number = len(self.contexts)
        delegate = FakeResearchPlanner(
            self.parent_commit_sha,
            experiment_id="exp_%03d" % number,
        )
        output = await delegate.propose(context)
        spec = output.spec.model_copy(
            update={"duplicate_key": "feature_cross:user_item:v%d" % number}
        )
        return output.model_copy(update={"spec": spec})


class SameMechanismReimplementationPlanner:
    """Model the tree planner selecting its one bounded reimplementation."""

    def __init__(self, parent_commit_sha: str) -> None:
        self.parent_commit_sha = parent_commit_sha

    async def propose(self, context):
        output = await FakeResearchPlanner(
            self.parent_commit_sha,
            experiment_id="exp_002",
        ).propose(context)
        return output.model_copy(
            update={
                "spec": output.spec.model_copy(
                    update={
                        "duplicate_key": "feature_cross:user_item:v1",
                        "hypothesis": (
                            "A corrected implementation of the approved cross "
                            "should alter within-user ordering."
                        ),
                        "change_summary": (
                            "Reimplement the same approved mechanism from its "
                            "trusted parent."
                        ),
                    }
                )
            }
        )


class NonImprovingEvaluator(FakeEvaluator):
    async def evaluate(self, request):
        result = await super().evaluate(request)
        if request.fidelity.value == "full":
            metrics = {name: 0.601 for name in self.metric_names}
            result = result.model_copy(
                update={
                    "metric_set": result.metric_set.model_copy(
                        update={
                            "metrics": metrics,
                            "primary_score": metrics[self.primary_metric_name],
                        }
                    ),
                    "baseline_delta": 0.001,
                    "parent_delta": 0.001,
                    "previous_best_delta": 0.0,
                    "trust": result.trust.model_copy(
                        update={
                            "seed_mean": (
                                0.601
                                if result.trust.stability.value == "confirmed"
                                else None
                            )
                        }
                    ),
                }
            )
        return result


class NoOpEvaluator(FakeEvaluator):
    """Return a structurally valid evaluation with unchanged predictions."""

    async def evaluate(self, request):
        result = await super().evaluate(request)
        return result.__class__.model_validate(
            {
                **result.model_dump(mode="json"),
                "prediction_change": {
                    "spearman_vs_parent": 1.0,
                    "changed_row_fraction": 0.0,
                },
                "seed_evidence_event_ids": [],
                "trust": {
                    **result.trust.model_dump(mode="json"),
                    "verdict": TrustVerdict.NO_OP,
                    "stability": Stability.NOT_APPLICABLE,
                    "flags": ["NO_PREDICTION_CHANGE"],
                    "seed_mean": None,
                    "seed_stderr": None,
                    "seed_count": 1,
                },
            }
        )


class NoOpRepairingCodingWorker(FakeCodingWorker):
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


class ExhaustedNoOpRepairingCodingWorker(FakeCodingWorker):
    async def repair_patch(self, context, decision):
        raise CodingWorkerError(
            "TRAE_STEP_LIMIT_EXCEEDED",
            "Trae exhausted the bounded 20-step no-op repair task",
        )


class BlockedPlanner:
    async def propose(self, context):
        return PlannerOutput(
            action=PlannerAction.BLOCKED,
            reason_code="portfolio_exhausted",
            reason="No reviewed non-duplicate method remains.",
            supporting_event_ids=context.source_event_ids,
        )


class InvalidProviderPlanner:
    async def propose(self, context):
        return PlannerOutput(
            action=PlannerAction.BLOCKED,
            reason_code="INVALID_PROVIDER_PLAN",
            reason="Provider output failed bounded plan validation.",
            supporting_event_ids=context.source_event_ids,
        )


class IntegrityRejectingPatchGate:
    async def check(self, candidate):
        return PatchCheckResult(
            run_id=candidate.run_id,
            experiment_id=candidate.experiment_id,
            attempt=candidate.attempt,
            patch_commit_sha=candidate.patch_commit_sha,
            diff_sha256=candidate.diff_sha256,
            accepted=False,
            checks=[CheckResult(name="secret_scan", status=CheckStatus.FAIL)],
            violations=[
                Violation(
                    code="SECRET_DETECTED",
                    message="Candidate patch crossed the credential boundary.",
                )
            ],
        )


class FinalAwareFakeRunner(FakeExecutionRunner):
    def __init__(self, artifacts):
        super().__init__(artifacts)
        self.requests = []
        self.final_prediction = None

    async def run(self, request, observer):
        self.requests.append(request)
        result = await super().run(request, observer)
        if request.command_id == "candidate_final_infer":
            self.final_prediction = result.prediction_artifact
        elif request.command_id == "submission_check":
            assert self.final_prediction is not None
            result = result.model_copy(
                update={"prediction_artifact": self.final_prediction}
            )
        return result


class FakeBaselineFinalSubmission:
    def __init__(self, artifacts, run_id: str) -> None:
        self.artifacts = artifacts
        self.run_id = run_id

    async def prepare_baseline(self):
        artifact = self.artifacts.write(
            artifact_id="baseline_submission",
            kind=ArtifactKind.SUBMISSION,
            relative_path="runs/%s/artifacts/baseline/final/submission.csv"
            % self.run_id,
            content=b"row_id,user_id,video_id,score\n0,u0,v0,0.5\n",
            content_type="text/csv",
        )
        return SubmissionCheckedPayload(
            accepted=True,
            submission_artifact=artifact,
            checks=[CheckResult(name="official_submission", status=CheckStatus.PASS)],
        )


def test_outer_loop_uses_memory_and_counts_distinct_terminal_iterations(
    harness, baseline_evaluation
):
    planner = SequentialPlanner(harness.config.baseline_commit_sha)
    harness.config.max_experiments = 10
    harness.planner = planner
    harness.evaluator = NonImprovingEvaluator(
        harness.config.metric_names,
        harness.config.primary_metric_name,
        harness.event_store,
    )
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_until_stopped())

    assert state.stop_reason_code == "converged"
    assert state.experiments_proposed == 3
    assert state.consecutive_non_improving_full_evaluations == 3
    assert [len(context.family_history) for context in planner.contexts] == [0, 1, 2]
    assert harness.events()[-1].event_type == EventType.RUN_STOPPED


def test_no_op_repairs_once_then_returns_evidence_to_planner(
    harness, baseline_evaluation
):
    planner = SequentialPlanner(harness.config.baseline_commit_sha)
    harness.planner = planner
    worker = NoOpRepairingCodingWorker(harness.event_store.artifact_store)
    harness.coding_worker = worker
    harness.evaluator = NoOpEvaluator(
        harness.config.metric_names,
        harness.config.primary_metric_name,
        harness.event_store,
    )
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.phase == "planning"
    assert state.experiments["exp_001"].status == ExperimentStatus.NO_OP
    assert len(worker.repair_calls) == 1
    recovery_decisions = [
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    ]
    assert [decision.action for decision in recovery_decisions] == [
        RecoveryAction.TRAE_REPAIR,
        RecoveryAction.RETURN_TO_PLANNER,
    ]
    assert not any(
        event.event_type == EventType.EXPERIMENT_DECIDED
        and event.payload.decision.decision.value == "prune"
        for event in harness.events()
    )
    planner_context = harness.context_builder.build_planner(harness.events())
    summary = planner_context.family_history[0]
    assert summary.trust_verdict == TrustVerdict.NO_OP
    assert summary.decision is None
    assert summary.status == ExperimentStatus.NO_OP.value

    # Recovery did not decide the branch outcome: a planner-selected modified
    # plan with the same semantic key is admitted exactly once after the
    # return-to-planner evidence.
    harness.planner = SameMechanismReimplementationPlanner(
        harness.config.baseline_commit_sha
    )
    second_state = asyncio.run(harness.run_one_experiment())
    assert second_state.experiments["exp_002"].status == ExperimentStatus.NO_OP
    assert len(worker.repair_calls) == 2


def test_exhausted_no_op_repair_returns_to_planner_without_stopping_run(
    harness, baseline_evaluation
):
    harness.planner = SequentialPlanner(harness.config.baseline_commit_sha)
    harness.coding_worker = ExhaustedNoOpRepairingCodingWorker(
        harness.event_store.artifact_store
    )
    harness.evaluator = NoOpEvaluator(
        harness.config.metric_names,
        harness.config.primary_metric_name,
        harness.event_store,
    )
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.status.value == "running"
    assert state.phase == "planning"
    assert state.experiments["exp_001"].status == ExperimentStatus.NO_OP
    assert not any(
        event.event_type == EventType.RUN_STOPPED for event in harness.events()
    )
    decisions = [
        event.payload.decision
        for event in harness.events()
        if event.event_type == EventType.RECOVERY_DECIDED
    ]
    assert [decision.action for decision in decisions] == [
        RecoveryAction.TRAE_REPAIR,
        RecoveryAction.RETURN_TO_PLANNER,
    ]
    assert decisions[-1].reason_code == "NO_OP_REPAIR_WORKER_EXHAUSTED"
    planner_context = harness.context_builder.build_planner(harness.events())
    assert planner_context.family_history[-1].trust_flags == [
        "NO_PREDICTION_CHANGE"
    ]


def test_blocked_planner_stops_without_spinning(harness, baseline_evaluation):
    harness.planner = BlockedPlanner()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_until_stopped())

    assert state.stop_reason_code == "no_legal_proposal"
    assert state.experiments_proposed == 0
    assert [event.event_type for event in harness.events()][-2:] == [
        EventType.PLANNER_RECOMMENDED,
        EventType.RUN_STOPPED,
    ]


def test_invalid_provider_plan_is_resumable_and_not_a_false_convergence(
    harness, baseline_evaluation
):
    harness.planner = InvalidProviderPlanner()
    harness.bootstrap(baseline_evaluation)

    with pytest.raises(ResumablePlanningError, match="bounded plan validation"):
        asyncio.run(harness.run_until_stopped())

    assert harness.state().status.value == "ready"
    assert harness.state().phase == "planner_context"
    assert harness.state().stop_reason_code is None

    harness.planner = BlockedPlanner()
    state = asyncio.run(harness.run_until_stopped())
    assert state.stop_reason_code == "no_legal_proposal"


def test_integrity_violation_is_recorded_and_stops_the_run(
    harness, baseline_evaluation
):
    harness.patch_gate = IntegrityRejectingPatchGate()
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_until_stopped())

    assert state.stop_reason_code == "fatal_integrity"
    assert state.experiments_proposed == 1
    assert [event.event_type for event in harness.events()][-2:] == [
        EventType.LESSON_RECORDED,
        EventType.RUN_STOPPED,
    ]


def test_selected_candidate_cleanly_reproduces_and_checks_final_submission(
    harness, baseline_evaluation
):
    runner = FinalAwareFakeRunner(harness.event_store.artifact_store)
    harness.runner = runner
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    harness.stop(StopDecision(True, "test_complete", "Finish the test search."))

    state = asyncio.run(harness.finalize())

    assert state.status.value == "finalized"
    assert state.final_experiment_id == "exp_001"
    assert [request.command_id for request in runner.requests[-3:]] == [
        "clean_reproduce",
        "candidate_final_infer",
        "submission_check",
    ]
    assert [event.event_type for event in harness.events()][-2:] == [
        EventType.FINAL_SELECTED,
        EventType.SUBMISSION_CHECKED,
    ]


def test_baseline_best_uses_protected_official_submission(harness, baseline_evaluation):
    harness.final_submission_provider = FakeBaselineFinalSubmission(
        harness.event_store.artifact_store, harness.config.run_id
    )
    harness.bootstrap(baseline_evaluation)
    harness.stop(StopDecision(True, "no_legal_proposal", "No candidate is legal."))

    state = asyncio.run(harness.finalize())

    assert state.status.value == "finalized"
    assert state.final_experiment_id == "baseline"
    final = next(
        event
        for event in harness.events()
        if event.event_type == EventType.FINAL_SELECTED
    )
    baseline = next(
        event
        for event in harness.events()
        if event.event_type == EventType.BASELINE_VERIFIED
    )
    assert final.payload.reproduction_evaluation_event_id == baseline.event_id
