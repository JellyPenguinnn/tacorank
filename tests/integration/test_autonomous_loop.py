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
from tacorank.orchestrator.router import OrchestrationError, ResumablePlanningError
from tacorank.orchestrator.finalize import FinalizationError
from tacorank.orchestrator.state import ExperimentStatus
from tacorank.providers.research_provider import ProviderError
from tacorank.recovery import RecoveryManager
from tacorank.schemas import (
    ArtifactKind,
    CheckResult,
    CheckStatus,
    CostEstimate,
    CostTier,
    EventType,
    ExperimentDecision,
    ExperimentDecisionKind,
    Fidelity,
    PlannerAction,
    PlannerOutput,
    PatchCheckResult,
    ResearchProposal,
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


class ParallelPlanner:
    def __init__(self, parent_commit_sha: str) -> None:
        self.parent_commit_sha = parent_commit_sha

    async def propose_parallel_direction(self, context, index, count):
        output = await FakeResearchPlanner(self.parent_commit_sha).propose(context)
        spec = output.spec.model_copy(
            update={
                "hypothesis": "Independent parallel hypothesis %d of %d." % (index + 1, count),
                "duplicate_key": "parallel:direction:%d" % index,
            }
        )
        return output.model_copy(update={"spec": spec})

    async def propose_synthesis(self, context, component_experiment_ids):
        by_id = {
            item.experiment_id: item
            for item in [context.baseline, *context.family_history]
        }
        parent_id = component_experiment_ids[0]
        parent = by_id[parent_id]
        spec = ResearchProposal(
            run_id=context.run_id,
            experiment_id="exp_pending",
            parent_experiment_id=parent_id,
            parent_commit_sha=parent.commit_sha,
            context_id=context.context_id,
            hypothesis="Align every accepted parallel improvement without interaction drift.",
            family="ensemble",
            change_summary="Combine all compatible accepted round patches.",
            expected_mechanism="Complementary changes retain independent gains.",
            success_criteria="The synthesis exceeds its strongest member.",
            falsification_condition="Any gate failure or no gain rejects synthesis.",
            estimated_cost=CostEstimate(
                llm_tokens_upper_bound=500,
                wall_time_seconds_upper_bound=60,
                gpu_seconds_upper_bound=0,
                cost_tier=CostTier.MEDIUM,
            ),
            method_card_ids=["ensemble_parallel_round_synthesis"],
            component_experiment_ids=list(component_experiment_ids[1:]),
            evidence_event_ids=context.source_event_ids,
            duplicate_key="parallel:round:synthesis",
        )
        return PlannerOutput(
            action=PlannerAction.PROPOSE,
            spec=spec,
            reason_code="parallel_round_synthesis",
            reason="Combine accepted lanes through the normal coding and gate path.",
        )


class ConcurrentCodingWorker:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.active = 0
        self.max_active = 0

    async def create_patch(self, context, spec):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return await self.delegate.create_patch(context, spec)
        finally:
            self.active -= 1

    async def repair_patch(self, context, decision):
        return await self.delegate.repair_patch(context, decision)


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


class AllImprovingEvaluator(FakeEvaluator):
    async def decide(self, result, context):
        decision = await super().decide(result, context)
        if (
            result.fidelity == Fidelity.FULL
            and result.trust.stability.value == "confirmed"
        ):
            return decision.model_copy(update={"best_eligible": True})
        return decision


def test_parallel_round_runs_independent_lanes_and_serializes_public_queries(
    harness, baseline_evaluation
):
    harness.config = harness.config.model_copy(
        update={
            "max_experiments": 3,
            "parallel_directions": 3,
            "synthesize_parallel_improvements": False,
        }
    )
    harness.planner = ParallelPlanner(harness.config.baseline_commit_sha)
    worker = ConcurrentCodingWorker(harness.coding_worker)
    harness.coding_worker = worker
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_parallel_round())

    assert state.experiments_proposed == 3
    assert worker.max_active == 3
    assert state.stop_reason_code == "experiment_budget"
    proposals = [
        event.payload.spec
        for event in harness.events()
        if event.event_type == EventType.EXPERIMENT_PROPOSED
    ]
    assert [spec.experiment_id for spec in proposals] == [
        "exp_001",
        "exp_002",
        "exp_003",
    ]
    public_indices = [
        event.payload.result.public_query_index
        for event in harness.events()
        if event.event_type == EventType.EVALUATION_COMPLETED
        and event.payload.result.population.value == "public_validation"
    ]
    assert public_indices == list(range(1, len(public_indices) + 1))


def test_parallel_round_synthesizes_all_independent_improvements(
    harness, baseline_evaluation
):
    harness.config = harness.config.model_copy(
        update={
            "max_experiments": 4,
            "parallel_directions": 3,
            "synthesize_parallel_improvements": True,
        }
    )
    harness.planner = ParallelPlanner(harness.config.baseline_commit_sha)
    harness.evaluator = AllImprovingEvaluator(
        harness.config.metric_names,
        harness.config.primary_metric_name,
        harness.event_store,
    )
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_parallel_round())

    assert state.experiments_proposed == 4
    synthesis = next(
        event.payload.spec
        for event in harness.events()
        if event.event_type == EventType.EXPERIMENT_PROPOSED
        and event.payload.spec.family == "ensemble"
    )
    assert synthesis.experiment_id == "exp_004"
    assert synthesis.parent_experiment_id == "exp_001"
    assert synthesis.component_experiment_ids == ["exp_002", "exp_003"]
    coder_context = next(
        event.payload.context
        for event in harness.events()
        if event.event_type == EventType.CONTEXT_CREATED
        and event.payload.context.role == "coder"
        and event.payload.context.experiment_id == "exp_004"
    )
    assert [item["experiment_id"] for item in coder_context.component_patches] == [
        "exp_002",
        "exp_003",
    ]


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


class PruneThenProviderFailurePlanner:
    def __init__(self, parent_commit_sha: str) -> None:
        self.parent_commit_sha = parent_commit_sha
        self.calls = 0

    async def propose(self, context):
        self.calls += 1
        if self.calls == 2:
            raise ProviderError(
                "DeepSeek request timed out after 120 seconds; "
                "Authorization: Bearer abcdefghijklmnop"
            )
        return await FakeResearchPlanner(self.parent_commit_sha).propose(context)


class ProxyPruningEvaluator(FakeEvaluator):
    async def decide(self, result, context):
        if result.fidelity == Fidelity.PROXY:
            return ExperimentDecision(
                run_id=result.run_id,
                experiment_id=result.experiment_id,
                evaluation_event_id="evt_pending",
                decision=ExperimentDecisionKind.PRUNE,
                reason_code="negative_proxy",
                fidelity_completed=Fidelity.PROXY,
                parent_eligible=False,
                best_eligible=False,
            )
        return await super().decide(result, context)


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


def test_provider_failure_after_prune_is_run_level_and_preserves_original_error(
    harness, baseline_evaluation
):
    planner = PruneThenProviderFailurePlanner(harness.config.baseline_commit_sha)
    harness.planner = planner
    harness.evaluator = ProxyPruningEvaluator(
        harness.config.metric_names,
        harness.config.primary_metric_name,
        harness.event_store,
    )
    harness.bootstrap(baseline_evaluation)

    pruned = asyncio.run(harness.run_one_experiment())

    assert pruned.experiments["exp_001"].status == ExperimentStatus.PRUNED
    assert pruned.active_experiment_id is None
    assert pruned.active_attempt is None
    assert pruned.active_fidelity is None

    with pytest.raises(
        OrchestrationError, match="PLANNER_PROVIDER_FAILURE"
    ):
        asyncio.run(harness.run_to_completion())
    state = harness.state()
    events = list(harness.events())
    planning_failure = next(
        event for event in events if event.event_type == EventType.PLANNING_FAILED
    )

    assert state.status.value == "failed"
    assert state.stop_reason_code == "PLANNER_PROVIDER_FAILURE"
    assert planning_failure.payload.result.error_class == "ProviderError"
    assert "DeepSeek request timed out after 120 seconds" in (
        planning_failure.payload.result.error_summary
    )
    assert "abcdefghijklmnop" not in planning_failure.payload.result.error_summary
    assert "[REDACTED]" in planning_failure.payload.result.error_summary
    assert not hasattr(planning_failure.payload.result, "experiment_id")
    assert not any(event.event_type == EventType.ADAPTER_FAILED for event in events)
    assert events[-2].event_type == EventType.PLANNING_FAILED
    assert events[-1].event_type == EventType.RUN_STOPPED
    assert not any(event.event_type == EventType.FINAL_SELECTED for event in events)
    assert state.experiments["exp_001"].terminal_event_id is not None


def test_integrity_violation_is_recorded_and_stops_the_run(
    harness, baseline_evaluation
):
    harness.patch_gate = IntegrityRejectingPatchGate()
    harness.recovery_manager = RecoveryManager()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_until_stopped())

    assert state.stop_reason_code == "fatal_integrity"
    assert state.status.value == "failed"
    assert state.experiments_proposed == 1
    assert [event.event_type for event in harness.events()][-2:] == [
        EventType.LESSON_RECORDED,
        EventType.RUN_STOPPED,
    ]
    with pytest.raises(FinalizationError, match="stopped run"):
        asyncio.run(harness.finalize())


def test_selected_candidate_cleanly_reproduces_and_checks_final_submission(
    harness, baseline_evaluation
):
    runner = FinalAwareFakeRunner(harness.event_store.artifact_store)
    harness.runner = runner
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    harness.stop(
        StopDecision(True, "no_legal_proposal", "Finish the test search.")
    )

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
