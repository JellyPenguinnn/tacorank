from __future__ import annotations

import asyncio

import pytest

from tacorank.coding import CodingWorkerError
from tacorank.context.builder import ContextBuildError
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
    ExperimentDecisionKind,
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
    ResearchCampaign,
    TrustVerdict,
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
        if context.research_campaign is not None:
            campaign = context.research_campaign
            family = campaign.family_order[0]
            spec = spec.model_copy(
                update={
                    "campaign_id": campaign.campaign_id,
                    "family": family,
                    "variant_id": "%s_%02d" % (family, number),
                    "variant_instruction": campaign.family_directives[family],
                    "variant_parameters": {"formulation": "bpr"},
                }
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


class NegativeNonImprovingEvaluator(NonImprovingEvaluator):
    async def evaluate(self, request):
        result = await super().evaluate(request)
        if request.fidelity.value == "full":
            result = result.model_copy(
                update={
                    "trust": result.trust.model_copy(
                        update={"verdict": TrustVerdict.NEGATIVE}
                    )
                }
            )
        return result

    async def decide(self, result, context):
        decision = await super().decide(result, context)
        if result.fidelity.value == "full":
            decision = decision.model_copy(
                update={
                    "decision": ExperimentDecisionKind.REJECT,
                    "best_eligible": False,
                    "next_fidelity": None,
                }
            )
        return decision


class NegativeAuditEvaluator(FakeEvaluator):
    async def evaluate(self, request):
        result = await super().evaluate(request)
        if request.fidelity.value == "full":
            result = result.model_copy(
                update={
                    "diagnostics": result.diagnostics.model_copy(
                        update={
                            "validation_arm_deltas": {
                                "val_a": 0.02,
                                "val_b": -0.01,
                            },
                            "validation_arm_gap": 0.03,
                        }
                    )
                }
            )
        return result


class AuditRankingEvaluator(FakeEvaluator):
    async def evaluate(self, request):
        result = await super().evaluate(request)
        if request.fidelity.value == "full":
            val_b = 0.02 if request.experiment_id == "exp_002" else 0.01
            result = result.model_copy(
                update={
                    "diagnostics": result.diagnostics.model_copy(
                        update={
                            "validation_arm_deltas": {
                                "val_a": 0.02,
                                "val_b": val_b,
                            },
                            "validation_arm_gap": abs(0.02 - val_b),
                        }
                    )
                }
            )
        return result


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


class InvalidOncePlanner:
    def __init__(self):
        self.calls = 0

    async def propose(self, context):
        self.calls += 1
        if self.calls == 1:
            return await InvalidProviderPlanner().propose(context)
        return await BlockedPlanner().propose(context)


class RaisingPlanner:
    async def propose(self, context):
        del context
        raise ProviderError("planner transport returned malformed JSON")


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


def test_negative_full_results_consume_convergence_patience(
    harness, baseline_evaluation
):
    planner = SequentialPlanner(harness.config.baseline_commit_sha)
    harness.config.max_experiments = 10
    harness.planner = planner
    harness.evaluator = NegativeNonImprovingEvaluator(
        harness.config.metric_names,
        harness.config.primary_metric_name,
        harness.event_store,
    )
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_until_stopped())

    assert state.stop_reason_code == "converged"
    assert state.experiments_proposed == 3
    assert state.consecutive_non_improving_full_evaluations == 3


def test_global_patience_still_bounds_explicit_depth_campaign(
    harness, baseline_evaluation
):
    planner = SequentialPlanner(harness.config.baseline_commit_sha)
    harness.config.max_experiments = 10
    harness.config.research_campaign = ResearchCampaign(
        campaign_id="objective_depth_4",
        family_order=["objective"],
        family_budgets={"objective": 4},
        family_method_card_ids={"objective": ["objective_pairwise_bpr"]},
        family_directives={"objective": "Adapt objective parameters from evidence."},
        proxy_checkpoint_interval=3,
    )
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
    assert len(planner.contexts) == 3


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
    assert len(
        [
            event
            for event in harness.events()
            if event.event_type == EventType.PLANNER_RECOMMENDED
        ]
    ) == 3

    harness.planner = BlockedPlanner()
    state = asyncio.run(harness.run_until_stopped())
    assert state.stop_reason_code == "no_legal_proposal"


def test_one_invalid_provider_plan_does_not_interrupt_campaign(
    harness, baseline_evaluation
):
    planner = InvalidOncePlanner()
    harness.planner = planner
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_until_stopped())

    assert planner.calls == 2
    assert state.status.value == "stopped"
    assert state.stop_reason_code == "no_legal_proposal"


def test_planner_provider_failure_is_recorded_before_specific_stop(
    harness, baseline_evaluation
):
    harness.planner = RaisingPlanner()
    harness.bootstrap(baseline_evaluation)

    state = asyncio.run(harness.run_one_experiment())

    assert state.stop_reason_code == "PLANNER_PROVIDER_FAILURE"
    assert state.experiments_proposed == 0
    assert [event.event_type for event in harness.events()][-2:] == [
        EventType.PLANNING_FAILED,
        EventType.RUN_STOPPED,
    ]
    failure = harness.events()[-2].payload.result
    assert failure.error_class == "ProviderError"
    assert failure.error_summary == "planner transport returned malformed JSON"


def test_binding_failure_is_durable_and_never_becomes_generic_adapter_stop(
    harness, baseline_evaluation, monkeypatch
):
    harness.bootstrap(baseline_evaluation)

    def fail_binding(proposal):
        del proposal
        raise ContextBuildError("immutable parent implementation is unavailable")

    monkeypatch.setattr(harness.context_builder, "bind_implementation", fail_binding)

    state = asyncio.run(harness.run_one_experiment())

    assert state.stop_reason_code == "PLANNER_BINDING_FAILURE"
    assert state.experiments_proposed == 0
    assert [event.event_type for event in harness.events()][-2:] == [
        EventType.PLANNING_FAILED,
        EventType.RUN_STOPPED,
    ]
    assert (
        harness.events()[-2].payload.result.error_summary
        == "immutable parent implementation is unavailable"
    )


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


def test_protected_audit_can_override_public_best_with_baseline(
    harness, baseline_evaluation
):
    runner = FinalAwareFakeRunner(harness.event_store.artifact_store)
    harness.runner = runner
    harness.evaluator = NegativeAuditEvaluator(
        harness.config.metric_names,
        harness.config.primary_metric_name,
        harness.event_store,
    )
    harness.final_submission_provider = FakeBaselineFinalSubmission(
        harness.event_store.artifact_store, harness.config.run_id
    )
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    harness.stop(StopDecision(True, "experiment_budget", "Finish the test search."))
    request_count = len(runner.requests)

    state = asyncio.run(harness.finalize())

    assert state.best_experiment_id == "exp_001"
    assert state.final_experiment_id == "baseline"
    assert len(runner.requests) == request_count


def test_protected_audit_can_select_trusted_candidate_beyond_public_best(
    harness, baseline_evaluation
):
    planner = SequentialPlanner(harness.config.baseline_commit_sha)
    runner = FinalAwareFakeRunner(harness.event_store.artifact_store)
    harness.planner = planner
    harness.runner = runner
    harness.evaluator = AuditRankingEvaluator(
        harness.config.metric_names,
        harness.config.primary_metric_name,
        harness.event_store,
    )
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    asyncio.run(harness.run_one_experiment())
    harness.stop(StopDecision(True, "experiment_budget", "Finish the test search."))

    state = asyncio.run(harness.finalize())

    assert state.best_experiment_id == "exp_001"
    assert state.final_experiment_id == "exp_002"
    assert [request.command_id for request in runner.requests[-3:]] == [
        "clean_reproduce",
        "candidate_final_infer",
        "submission_check",
    ]
