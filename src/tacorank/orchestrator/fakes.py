"""Deterministic fake adapters for the integration contract test."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from ..artifacts import ArtifactStore
from ..evaluation.adapter import (
    ordered_prediction_sha256,
    ordered_row_identity_sha256,
)
from ..schemas import (
    ArtifactKind,
    CheckResult,
    CheckStatus,
    CoderContext,
    CostEstimate,
    CostTier,
    EvaluationDecisionContext,
    EvaluationDiagnostics,
    EvaluationRequest,
    EvaluationResult,
    ExperimentDecision,
    ExperimentDecisionKind,
    ExperimentSpec,
    Fidelity,
    Integrity,
    LessonCandidate,
    LessonCategory,
    LessonOrigin,
    MetricSet,
    MonitorAction,
    MonitorDirective,
    OutputCheckResult,
    PatchCandidate,
    PatchCheckResult,
    PlannerAction,
    PlannerContext,
    PlannerOutput,
    PredictionChange,
    RecoveryAction,
    RecoveryContext,
    RecoveryDecision,
    RecoveryPolicyContext,
    ResearchProposal,
    ResourceDelta,
    RunOutcome,
    RunRequest,
    RunResult,
    Stability,
    TelemetrySample,
    TokenMeasurement,
    TrustAssessment,
    TrustVerdict,
)
from ..run_layout import experiment_artifact_prefix


FAKE_SHA = hashlib.sha256(b"fake").hexdigest()


class FakeResearchPlanner:
    def __init__(self, parent_commit_sha: str, experiment_id: str = "exp_001"):
        self.parent_commit_sha = parent_commit_sha
        self.experiment_id = experiment_id

    async def propose(self, context: PlannerContext) -> PlannerOutput:
        spec = ResearchProposal(
            run_id=context.run_id,
            experiment_id=self.experiment_id,
            parent_experiment_id="baseline",
            parent_commit_sha=self.parent_commit_sha,
            context_id=context.context_id,
            hypothesis="Adding a deterministic user-item cross improves ranking.",
            family="feature_cross",
            change_summary="Add one bounded categorical feature cross.",
            expected_mechanism="The cross captures a missing interaction.",
            success_criteria="Trusted full primary score exceeds the baseline.",
            falsification_condition="Proxy and full scores do not improve.",
            estimated_cost=CostEstimate(
                llm_tokens_upper_bound=500,
                wall_time_seconds_upper_bound=60,
                gpu_seconds_upper_bound=0,
                cost_tier=CostTier.LOW,
            ),
            method_card_ids=[],
            evidence_event_ids=context.source_event_ids,
            duplicate_key="feature_cross:user_item:v1",
        )
        return PlannerOutput(
            action=PlannerAction.PROPOSE,
            spec=spec,
            reason_code="fake_candidate",
            reason="Deterministic fake proposal for the vertical slice.",
            resource_delta=ResourceDelta(
                llm_input_tokens=100,
                llm_output_tokens=60,
                token_measurement=TokenMeasurement.PROVIDER,
            ),
        )


class FakeCodingWorker:
    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts

    async def create_patch(self, context: CoderContext, spec: ExperimentSpec) -> PatchCandidate:
        diff = (
            b"diff --git a/solution/candidate.py b/solution/candidate.py\n"
            b"+feature_cross = True\n"
        )
        trajectory = b"fake coding trajectory\n"
        prefix = experiment_artifact_prefix(
            spec.run_id, spec.experiment_id, attempt=1
        )
        diff_ref = self.artifacts.write(
            artifact_id="diff_%s_1" % spec.experiment_id,
            kind=ArtifactKind.DIFF,
            relative_path=prefix + "/patch.diff",
            content=diff,
            content_type="text/x-diff",
        )
        trajectory_ref = self.artifacts.write(
            artifact_id="trajectory_%s_1" % spec.experiment_id,
            kind=ArtifactKind.TRAJECTORY,
            relative_path=prefix + "/trajectory.txt",
            content=trajectory,
            content_type="text/plain",
        )
        proposal_id = spec.evidence_event_ids[-1] if spec.evidence_event_ids else "evt_000001"
        # The orchestrator replaces this with the actual proposal event id before
        # validation, keeping the fake independent of the ledger implementation.
        return PatchCandidate(
            run_id=spec.run_id,
            experiment_id=spec.experiment_id,
            attempt=1,
            experiment_spec_event_id=proposal_id,
            context_id=context.context_id,
            base_commit_sha=spec.parent_commit_sha,
            patch_commit_sha="a" * 40,
            diff_sha256=diff_ref.sha256,
            changed_files=spec.target_files,
            diff_artifact=diff_ref,
            trajectory_artifact=trajectory_ref,
            trae_version="fake-1",
            model_id="fake-model",
            steps_used=1,
            resource_delta=ResourceDelta(
                llm_input_tokens=120,
                llm_output_tokens=80,
                token_measurement=TokenMeasurement.ESTIMATED,
            ),
        )

    async def repair_patch(
        self, context: RecoveryContext, decision: RecoveryDecision
    ) -> PatchCandidate:
        raise AssertionError("the successful fake lifecycle does not invoke repair")


class FakePatchGate:
    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts

    async def check(self, candidate: PatchCandidate) -> PatchCheckResult:
        prefix = experiment_artifact_prefix(
            candidate.run_id, candidate.experiment_id, attempt=candidate.attempt
        )
        receipt = self.artifacts.write(
            artifact_id="receipt_%s_%d" % (candidate.experiment_id, candidate.attempt),
            kind=ArtifactKind.VERIFICATION_RECEIPT,
            relative_path=prefix + "/patch-receipt.json",
            content=b'{"accepted":true}\n',
            content_type="application/json",
        )
        return PatchCheckResult(
            run_id=candidate.run_id,
            experiment_id=candidate.experiment_id,
            attempt=candidate.attempt,
            patch_commit_sha=candidate.patch_commit_sha,
            diff_sha256=candidate.diff_sha256,
            accepted=True,
            receipt_id="receipt_%s_%d" % (candidate.experiment_id, candidate.attempt),
            receipt_artifact=receipt,
            checks=[CheckResult(name="fake_gate_a", status=CheckStatus.PASS)],
        )


class FakeHealthObserver:
    def observe(self, sample: TelemetrySample) -> MonitorDirective:
        return MonitorDirective(action=MonitorAction.CONTINUE)


class FakeExecutionRunner:
    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts

    async def run(self, request: RunRequest, observer: FakeHealthObserver) -> RunResult:
        directive = observer.observe(
            TelemetrySample(
                timestamp=datetime.now(timezone.utc),
                run_id=request.run_id,
                experiment_id=request.experiment_id,
                attempt=request.attempt,
                elapsed_ms=1,
                process_alive=True,
                last_output_age_ms=0,
                cpu_percent=1.0,
                rss_mb=10,
                disk_free_mb=100,
                recent_output_tail="ok",
            )
        )
        assert directive.action == MonitorAction.CONTINUE
        suffix = "%s_%d_%s" % (request.experiment_id, request.attempt, request.fidelity.value)
        prefix = experiment_artifact_prefix(
            request.run_id, request.experiment_id, attempt=request.attempt
        )
        log = self.artifacts.write(
            artifact_id="log_" + suffix,
            kind=ArtifactKind.LOG,
            relative_path=prefix + "/%s-training.log" % request.fidelity.value,
            content=b"fake execution completed\n",
            content_type="text/plain",
        )
        telemetry = self.artifacts.write(
            artifact_id="telemetry_" + suffix,
            kind=ArtifactKind.LOG,
            relative_path=prefix + "/%s-telemetry.json" % request.fidelity.value,
            content=b'{"healthy":true}\n',
            content_type="application/json",
        )
        predictions = self.artifacts.write(
            artifact_id="predictions_" + suffix,
            kind=ArtifactKind.PREDICTIONS,
            relative_path=prefix + "/%s-predictions.csv" % request.fidelity.value,
            content=b"row_id,user_id,video_id,score\n0,u0,v0,0.5\n",
            content_type="text/csv",
        )
        checkpoint = self.artifacts.write(
            artifact_id="checkpoint_" + suffix,
            kind=ArtifactKind.CHECKPOINT,
            relative_path=prefix + "/%s-checkpoint.bin" % request.fidelity.value,
            content=b"fake checkpoint",
            content_type="application/octet-stream",
        )
        return RunResult(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt=request.attempt,
            fidelity=request.fidelity,
            patch_commit_sha=request.patch_commit_sha,
            outcome=RunOutcome.SUCCESS,
            exit_code=0,
            log_artifact=log,
            telemetry_artifact=telemetry,
            checkpoint_artifact=checkpoint,
            prediction_artifact=predictions,
            resource_delta=ResourceDelta(wall_time_ms=10, cpu_time_ms=8),
        )


class FakeOutputGate:
    async def check(self, result: RunResult) -> OutputCheckResult:
        assert result.prediction_artifact is not None
        return OutputCheckResult(
            run_id=result.run_id,
            experiment_id=result.experiment_id,
            attempt=result.attempt,
            prediction_artifact=result.prediction_artifact,
            ordered_row_identity_sha256=ordered_row_identity_sha256(
                (0,), ("u0",), ("v0",)
            ),
            ordered_prediction_sha256=ordered_prediction_sha256(
                (0,), ("u0",), ("v0",), (0.5,)
            ),
            accepted=True,
            checks={"schema": CheckStatus.PASS, "finite": CheckStatus.PASS},
            score_stats={"min": 0.5, "max": 0.5},
        )


class FakeEvaluator:
    def __init__(self, metric_names, primary_metric_name, event_store):
        self.metric_names = list(metric_names)
        self.primary_metric_name = primary_metric_name
        self.event_store = event_store

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        primary = 0.61 if request.fidelity == Fidelity.PROXY else 0.62
        metrics = {name: primary for name in self.metric_names}
        seed_evidence_event_ids = []
        stability = Stability.NOT_APPLICABLE
        if request.fidelity == Fidelity.FULL:
            execution = next(
                event.payload.request
                for event in reversed(
                    self.event_store.read_events(repair_tail=True)
                )
                if event.event_type.value == "execution.started"
                and event.payload.request.experiment_id == request.experiment_id
                and event.payload.request.attempt == request.attempt
            )
            if execution.command_id != "clean_reproduce":
                seed_evidence_event_ids = [
                    event.event_id
                    for event in self.event_store.read_events(repair_tail=True)
                    if event.event_type.value == "evaluation.completed"
                    and event.payload.result.experiment_id == request.experiment_id
                    and event.payload.result.population == request.population
                    and event.payload.result.fidelity == request.fidelity
                ]
            stability = (
                Stability.CONFIRMED
                if len(seed_evidence_event_ids) >= 2
                else Stability.SINGLE_SEED
            )
        confirmed = stability == Stability.CONFIRMED
        diagnostics = EvaluationDiagnostics()
        if request.fidelity == Fidelity.FULL:
            diagnostics = EvaluationDiagnostics(
                proxy_parent_delta=0.01,
                proxy_full_delta_gap=0.01,
                validation_arm_deltas={"val_a": 0.02, "val_b": 0.01},
                validation_arm_gap=0.01,
                temporal_delta_slope=-0.003,
                gain_concentration_top10pct=0.75,
                slice_deltas={
                    "user_history.cold": -0.01,
                    "user_history.hot": 0.02,
                },
                best_slice="user_history.hot",
                worst_slice="user_history.cold",
                failure_hypotheses=[
                    "Cohort weakness: the largest measured slice regression is user_history.cold."
                ],
                limitations=[
                    "Direct train/validation gap unavailable: contract v1 emits no protected train predictions.",
                    "Associational diagnostics do not prove causality; confirm hypotheses with a controlled ablation.",
                ],
            )
        return EvaluationResult(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt=request.attempt,
            population=request.population,
            fidelity=request.fidelity,
            seed=request.seed,
            public_query_index=request.public_query_index,
            evaluator_sha256=request.evaluator_sha256,
            contract_sha256=request.contract_sha256,
            metric_set=MetricSet(
                metrics=metrics,
                primary_metric_name=self.primary_metric_name,
                primary_score=metrics[self.primary_metric_name],
            ),
            baseline_delta=primary - request.baseline_summary[self.primary_metric_name],
            parent_delta=primary - request.parent_summary.get(self.primary_metric_name, primary),
            previous_best_delta=primary
            - request.previous_best_summary.get(self.primary_metric_name, primary),
            baseline_metric_deltas={
                name: value - request.baseline_summary[name]
                for name, value in metrics.items()
            },
            parent_metric_deltas={
                name: value - request.parent_summary.get(name, value)
                for name, value in metrics.items()
            },
            previous_best_metric_deltas={
                name: value - request.previous_best_summary.get(name, value)
                for name, value in metrics.items()
            },
            prediction_change=PredictionChange(
                spearman_vs_parent=0.9,
                changed_row_fraction=0.1,
            ),
            diagnostic_metrics={
                "spearman_vs_fm_baseline": 0.8,
                "user_rankable_fraction": 1.0,
            },
            trust=TrustAssessment(
                verdict=TrustVerdict.ACCEPTED,
                stability=stability,
                integrity=Integrity.CLEAN,
                flags=[],
                eta_applied=0.0016,
                seed_mean=primary if confirmed else None,
                seed_stderr=0.0 if confirmed else None,
                seed_count=len(seed_evidence_event_ids) + 1,
            ),
            diagnostics=diagnostics,
            seed_evidence_event_ids=seed_evidence_event_ids,
        )

    async def decide(
        self, result: EvaluationResult, context: EvaluationDecisionContext
    ) -> ExperimentDecision:
        if result.fidelity == Fidelity.PROXY:
            decision = ExperimentDecisionKind.PROMOTE
            next_fidelity = Fidelity.FULL
            best_eligible = False
        elif result.trust.stability != Stability.CONFIRMED:
            decision = ExperimentDecisionKind.PROMOTE
            next_fidelity = Fidelity.FULL
            best_eligible = False
        else:
            decision = ExperimentDecisionKind.ACCEPT
            next_fidelity = None
            best_eligible = result.metric_set.primary_score > context.previous_best_score
        evaluation_event = next(
            (
                event
                for event in reversed(
                    self.event_store.read_events(repair_tail=True)
                )
                if event.event_type.value == "evaluation.completed"
                and event.payload.result.experiment_id == result.experiment_id
                and event.payload.result.attempt == result.attempt
            ),
            None,
        )
        lesson_candidate = None
        if result.trust.stability == Stability.CONFIRMED:
            assert evaluation_event is not None
            lesson_candidate = LessonCandidate(
                origin=LessonOrigin.RESEARCH,
                category=LessonCategory.RESEARCH_RESULT,
                tags=["feature_cross", "confirmed"],
                summary=(
                    "The deterministic feature-cross experiment produced a "
                    "confirmed clean result."
                ),
                applicability="The frozen fake full public-validation frame.",
                avoid_when="Do not generalize beyond the recorded evaluation frame.",
                confidence=0.9,
                source_event_ids=(
                    list(result.seed_evidence_event_ids)
                    + [evaluation_event.event_id]
                ),
                source_commit_shas=["c" * 40],
                measured_under_frame_experiment_id="baseline",
            )
        return ExperimentDecision(
            run_id=result.run_id,
            experiment_id=result.experiment_id,
            evaluation_event_id="evt_pending",
            decision=decision,
            reason_code="fake_trusted_result",
            fidelity_completed=result.fidelity,
            parent_eligible=True,
            best_eligible=best_eligible,
            next_fidelity=next_fidelity,
            lesson_candidate=lesson_candidate,
        )


class FakeRecoveryManager:
    async def decide(self, failure_event_id, result, context: RecoveryPolicyContext):
        return RecoveryDecision(
            run_id=context.run_id,
            experiment_id=context.experiment_id,
            failure_event_id=failure_event_id,
            repair_attempt=1,
            action=RecoveryAction.ABANDON,
            reason_code="fake_abandon",
            instructions="Stop the fake failed experiment.",
            same_error_count=1,
            remaining_repair_budget=context.remaining_repair_budget,
        )
