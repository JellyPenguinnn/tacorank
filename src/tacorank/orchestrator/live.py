"""Production adapter composition for the canonical TacoRank harness.

This module is intentionally the only place where ledger-owned identities are
translated into the narrower production adapter contracts.  No adapter is
allowed to guess an attempt, receipt, population, or reference prediction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import Field, field_validator, model_validator

from benchmarks.kuairand_pure.evaluator_adapter import create_evaluator_adapter
from benchmarks.kuairand_pure.submission_adapter import validate_submission

from ..artifacts import ArtifactStore
from ..coding import CandidateIdentity, TraeCodingWorker, TraeConfig
from ..config import ContractError, RunConfig, VerifiedContract
from ..evaluation import (
    DecisionContext,
    DiagnosticFeatures,
    EvaluationInputs,
    EvaluationService,
    MetricSet as DomainMetricSet,
    OutputGateEvidence,
)
from ..evaluation.adapter import PopulationManifest, ordered_row_identity_sha256
from ..evaluation.comparisons import compare_metric_sets
from ..evaluation.proxy import split_validation_indices
from ..evaluation.types import (
    EvaluationResult as DomainEvaluationResult,
    PredictionChange as DomainPredictionChange,
    TrustAssessment as DomainTrustAssessment,
)
from ..execution import (
    CanonicalArtifactStoreAdapter,
    ContainerMountPolicy,
    ContainerReadOnlyMount,
    DockerSandbox,
    ExecutionRunner,
    PipelineCommandInputs,
    ReceiptArtifactBinding,
    RunnerPolicy,
    SandboxPolicy,
    SealedExecutionVerifier,
    default_command_registry,
)
from ..git import WorktreeManager
from ..memory.event_store import EventStore
from ..memory.projections import project
from ..recovery import RecoveryManager
from ..reflection import build_research_lesson
from ..run_layout import experiment_artifact_prefix, run_artifact_root
from ..safety import (
    DataAccessPolicy,
    DataViewPolicy,
    DockerEntrypointSmokeCheck,
    ExecutionSealExpectation,
    InterfaceRequirement,
    OutputColumn,
    OutputContract,
    OutputGate,
    PatchGate,
    ProtectedManifest,
    ReceiptStore,
)
from ..schemas import (
    ArtifactKind,
    CheckResult,
    CheckStatus,
    EvaluationDecisionContext,
    EvaluationRequest,
    EvaluationResult,
    ExperimentDecision,
    Fidelity,
    Integrity,
    Population,
    Stability,
    StrictModel,
    SubmissionCheckedPayload,
    TrustAssessment,
    TrustVerdict,
    normalize_relative_path,
)
from ..sre import SREObserver


_INPUT_COMMAND_IDS = frozenset(
    {
        "baseline_full",
        "candidate_smoke",
        "candidate_proxy",
        "candidate_full",
        "candidate_final_infer",
        "clean_reproduce",
    }
)
_POPULATION_KEYS = frozenset({"smoke", "proxy", "full"})


class LiveAdapterConfig(StrictModel):
    """Operator-reviewed deployment values omitted from the frozen run spec."""

    schema_version: str = "1.0"
    worktree_root: Path
    required_submodules: List[str] = Field(default_factory=list)
    trae: Dict[str, Any]
    contract_root: Path
    input_roots: Dict[str, Path]
    baseline_entrypoint: str
    candidate_entrypoint: str
    submission_check_entrypoint: str
    python_executable: Path
    container_python_executable: str
    docker_executable: Path
    docker_host: str
    docker_image: str
    docker_image_environment_sha256: str
    docker_cpu_count: float = Field(default=2.0, gt=0)
    docker_tmpfs_size_mb: int = Field(default=256, gt=0)
    output_quota_max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    data_manifest_path: Path
    population_csvs: Dict[str, Path]
    baseline_prediction_csvs: Dict[str, Path]
    baseline_final_prediction_csv: Path
    baseline_parity_receipt_path: Optional[Path] = None
    candidate_allowed_columns: List[str] = Field(default_factory=list)
    protected_columns: List[str] = Field(default_factory=lambda: ["label"])
    hidden_path_tokens: List[str] = Field(
        default_factory=lambda: ["hidden_labels", "final_labels", "test_labels"]
    )
    future_column_patterns: List[str] = Field(
        default_factory=lambda: [r"(?:^|_)future(?:_|$)"]
    )
    allowed_import_roots: Optional[List[str]] = None
    allowed_capability_imports: List[str] = Field(default_factory=list)
    allowed_dependency_changes: List[str] = Field(default_factory=list)

    @field_validator("required_submodules")
    @classmethod
    def validate_submodules(cls, values: List[str]) -> List[str]:
        normalized = [normalize_relative_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("required_submodules must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_mappings(self) -> "LiveAdapterConfig":
        if set(self.input_roots) != _INPUT_COMMAND_IDS:
            raise ValueError("input_roots must exactly match production pipeline commands")
        if set(self.population_csvs) != _POPULATION_KEYS:
            raise ValueError("population_csvs must exactly contain smoke, proxy, and full")
        if set(self.baseline_prediction_csvs) != {"proxy", "full"}:
            raise ValueError("baseline_prediction_csvs must exactly contain proxy and full")
        return self

    @classmethod
    def load(cls, path: Path) -> "LiveAdapterConfig":
        source = path.resolve(strict=True)
        raw = json.loads(source.read_text(encoding="utf-8"))
        base = source.parent
        for key in (
            "worktree_root",
            "contract_root",
            "python_executable",
            "docker_executable",
            "data_manifest_path",
            "baseline_final_prediction_csv",
        ):
            raw[key] = str(_resolve_config_path(base, raw[key]))
        if raw.get("baseline_parity_receipt_path") is not None:
            raw["baseline_parity_receipt_path"] = str(
                _resolve_config_path(base, raw["baseline_parity_receipt_path"])
            )
        for mapping_name in (
            "input_roots",
            "population_csvs",
            "baseline_prediction_csvs",
        ):
            raw[mapping_name] = {
                key: str(_resolve_config_path(base, value))
                for key, value in raw[mapping_name].items()
            }
        trae = dict(raw["trae"])
        for key in (
            "config_file",
            "docker_executable",
            "trae_install_root",
            "trae_install_identity_file",
            "trae_runtime_root",
            "python_dotenv_metadata_file",
        ):
            if trae.get(key) is not None:
                trae[key] = str(_resolve_config_path(base, trae[key]))
        if "command_prefix" in trae and trae["command_prefix"]:
            command = list(trae["command_prefix"])
            command[0] = str(_resolve_config_path(base, command[0]))
            trae["command_prefix"] = command
        raw["trae"] = trae
        return cls.model_validate(raw)


@dataclass(frozen=True)
class LiveAdapters:
    coding_worker: Any
    patch_gate: Any
    runner: Any
    health_observer: Any
    recovery_manager: Any
    output_gate: Any
    evaluator: Any
    baseline: EvaluationResult
    final_submission_provider: Any


def _trae_config_from_mapping(values: Mapping[str, Any]) -> TraeConfig:
    """Normalize JSON-shaped live values at the Trae dataclass boundary."""

    try:
        return TraeConfig.from_mapping(values)
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("live Trae configuration has invalid field types") from error


@dataclass(frozen=True)
class _PopulationData:
    rows: Tuple[Mapping[str, Any], ...]
    labels: Tuple[int, ...]
    diagnostic_features: Optional[DiagnosticFeatures] = None


class ProtectedBaselineFinalSubmission:
    """Copy and re-check the manifest-attested official FM test submission."""

    def __init__(
        self,
        *,
        config: RunConfig,
        artifact_store: ArtifactStore,
        source: Path,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self.config = config
        self.artifact_store = artifact_store
        self.source = source
        self.rows = tuple(rows)

    async def prepare_baseline(self) -> SubmissionCheckedPayload:
        checked = validate_submission(self.source, self.rows)
        artifact = self.artifact_store.write(
            artifact_id="baseline_final_submission",
            kind=ArtifactKind.SUBMISSION,
            relative_path=(
                run_artifact_root(self.config.run_id) + "/baseline/final/submission.csv"
            ),
            content=self.source.read_bytes(),
            content_type="text/csv",
        )
        return SubmissionCheckedPayload(
            accepted=True,
            submission_artifact=artifact,
            checks=[
                CheckResult(
                    name="official_submission_contract",
                    status=CheckStatus.PASS,
                    summary=(
                        "%d ordered rows; %.6f unique-score fraction"
                        % (checked.rows, checked.unique_score_fraction)
                    ),
                )
            ],
        )


class LedgerCandidateIdentityResolver:
    def __init__(self, event_store: EventStore) -> None:
        self.event_store = event_store

    def _proposal_event_id(self, experiment_id: str) -> str:
        matches = [
            event
            for event in self.event_store.read_events(repair_tail=True)
            if event.payload.type == "experiment.proposed"
            and event.payload.spec.experiment_id == experiment_id
        ]
        if len(matches) != 1:
            raise ContractError("coding identity requires exactly one proposal event")
        return matches[0].event_id

    def for_initial(self, context: Any, spec: Any) -> CandidateIdentity:
        if context.experiment_id != spec.experiment_id:
            raise ContractError("coder context and experiment spec identities differ")
        events = self.event_store.read_events(repair_tail=True)
        if any(
            event.payload.type == "patch.created"
            and event.payload.candidate.experiment_id == spec.experiment_id
            for event in events
        ):
            raise ContractError("initial coding identity cannot follow a sealed patch")
        events_by_id = {event.event_id: event for event in events}
        coding_retries = 0
        for event in events:
            if event.payload.type != "recovery.decided":
                continue
            decision = event.payload.decision
            action = getattr(decision.action, "value", decision.action)
            if (
                decision.experiment_id != spec.experiment_id
                or action != "retry_same_commit"
            ):
                continue
            failure = events_by_id.get(decision.failure_event_id)
            if (
                failure is not None
                and failure.payload.type == "adapter.failed"
                and failure.payload.result.failure_stage == "coding"
            ):
                coding_retries += 1
        return CandidateIdentity(
            1 + coding_retries,
            self._proposal_event_id(spec.experiment_id),
        )

    def for_repair(self, context: Any, decision: Any) -> CandidateIdentity:
        if context.experiment_id != decision.experiment_id:
            raise ContractError("recovery context and decision identities differ")
        return CandidateIdentity(
            int(decision.repair_attempt) + 1,
            self._proposal_event_id(decision.experiment_id),
        )


class WorktreePatchGate:
    """Construct Gate A against the exact experiment worktree, never the controller checkout."""

    def __init__(
        self,
        *,
        worktrees: WorktreeManager,
        repository_root: Path,
        event_store: EventStore,
        editable_roots: Sequence[str],
        protected_manifest: ProtectedManifest,
        receipt_store: ReceiptStore,
        data_access_policy: DataAccessPolicy,
        allowed_command_ids: Sequence[str],
        artifact_roots: Sequence[str],
        allowed_import_roots: Optional[Sequence[str]],
        allowed_capability_imports: Sequence[str],
        allowed_dependency_changes: Sequence[str],
        smoke_check: Any = None,
    ) -> None:
        self.worktrees = worktrees
        self.repository_root = repository_root
        self.event_store = event_store
        self.options = {
            "editable_roots": tuple(editable_roots),
            "protected_manifest": protected_manifest,
            "receipt_store": receipt_store,
            "data_access_policy": data_access_policy,
            "allowed_command_ids": tuple(allowed_command_ids),
            "artifact_roots": tuple(artifact_roots),
            "artifact_repository_root": repository_root,
            "allowed_import_roots": allowed_import_roots,
            "allowed_capability_imports": tuple(allowed_capability_imports),
            "allowed_dependency_changes": tuple(allowed_dependency_changes),
            "smoke_check": smoke_check,
            "interface_requirements": (
                InterfaceRequirement(
                    "solution/candidate.py", "run", parameters=("invocation",)
                ),
            ),
        }

    async def check(self, candidate: Any) -> Any:
        workspace = self.worktrees.path_for(candidate.run_id, candidate.experiment_id)
        # The cumulative-diff root is the approved ExperimentSpec parent.  It
        # is supplied by the wrapper because PatchCandidate deliberately does
        # not duplicate controller-owned proposal fields.
        spec = self._experiment_spec(candidate.experiment_id)
        return await PatchGate(repository_root=workspace, **self.options).check(
            candidate,
            experiment_root_commit_sha=spec.parent_commit_sha,
            authorized_changed_files=spec.target_files,
        )

    def _experiment_spec(self, experiment_id: str) -> Any:
        matches = [
            event.payload.spec
            for event in self.event_store.read_events(repair_tail=True)
            if event.payload.type == "experiment.proposed"
            and event.payload.spec.experiment_id == experiment_id
        ]
        if len(matches) != 1:
            raise ContractError("Gate A cannot resolve the approved ExperimentSpec")
        return matches[0]


class LedgerReceiptArtifactResolver:
    def __init__(self, event_store: EventStore) -> None:
        self.event_store = event_store

    def resolve(self, request: Any) -> ReceiptArtifactBinding:
        matches = []
        proposal_parent = None
        for event in self.event_store.read_events(repair_tail=True):
            if (
                event.payload.type == "experiment.proposed"
                and event.payload.spec.experiment_id == request.experiment_id
            ):
                proposal_parent = event.payload.spec.parent_commit_sha
            if event.payload.type != "patch.checked":
                continue
            result = event.payload.result
            if (
                result.experiment_id == request.experiment_id
                and result.accepted
                and result.receipt_id == request.patch_receipt_id
                and result.patch_commit_sha == request.patch_commit_sha
            ):
                matches.append(result)
        if len(matches) != 1 or proposal_parent is None:
            raise ContractError("execution request does not resolve to one accepted Gate A receipt")
        result = matches[0]
        return ReceiptArtifactBinding(
            artifact_ref=result.receipt_artifact,
            patch_attempt=result.attempt,
            experiment_root_commit_sha=proposal_parent,
        )


class LedgerSubmissionArtifactResolver:
    """Resolve only the selected commit's accepted final-inference output."""

    def __init__(self, event_store: EventStore) -> None:
        self.event_store = event_store

    def resolve(self, request: Any) -> Any:
        events = self.event_store.read_events(repair_tail=True)
        state = project(events)
        if (
            request.command_id != "submission_check"
            or request.experiment_id != state.best_experiment_id
            or request.patch_commit_sha != state.best_commit_sha
        ):
            raise ContractError("submission check does not target the selected commit")
        started_by_attempt = {
            event.payload.request.attempt: event.payload.request
            for event in events
            if event.payload.type == "execution.started"
            and event.payload.request.experiment_id == request.experiment_id
        }
        for event in reversed(events):
            if event.payload.type != "output.checked":
                continue
            output = event.payload.result
            if output.experiment_id != request.experiment_id or not output.accepted:
                continue
            source_request = started_by_attempt.get(output.attempt)
            if source_request is not None and (
                source_request.command_id == "candidate_final_infer"
                and source_request.patch_commit_sha == request.patch_commit_sha
            ):
                return output.prediction_artifact
        raise ContractError("selected commit has no accepted final-inference output")


class LedgerOutputGate:
    def __init__(
        self,
        *,
        repository_root: Path,
        event_store: EventStore,
        populations: Mapping[str, _PopulationData],
        artifact_roots: Sequence[str],
        max_single_score_fraction: float,
    ) -> None:
        self.repository_root = repository_root
        self.event_store = event_store
        self.artifact_roots = tuple(artifact_roots)
        self.gates = {
            key: OutputGate(
                repository_root=repository_root,
                artifact_roots=artifact_roots,
                contract=_output_contract(
                    population.rows,
                    max_single_score_fraction=max_single_score_fraction,
                ),
            )
            for key, population in populations.items()
        }

    def _request_and_receipt(self, result: Any) -> Tuple[Any, Any]:
        request = None
        receipt = None
        for event in self.event_store.read_events(repair_tail=True):
            if event.payload.type == "execution.started":
                candidate = event.payload.request
                if (
                    candidate.experiment_id == result.experiment_id
                    and candidate.attempt == result.attempt
                    and candidate.fidelity == result.fidelity
                    and candidate.patch_commit_sha == result.patch_commit_sha
                ):
                    request = candidate
            elif event.payload.type == "patch.checked":
                candidate = event.payload.result
                if (
                    candidate.accepted
                    and request is not None
                    and candidate.receipt_id == request.patch_receipt_id
                ):
                    receipt = candidate
        if request is None:
            raise ContractError("Gate B cannot resolve the controller-owned execution request")
        if receipt is None:
            receipt = next(
                (
                    event.payload.result
                    for event in self.event_store.read_events(repair_tail=True)
                    if event.payload.type == "patch.checked"
                    and event.payload.result.accepted
                    and event.payload.result.receipt_id == request.patch_receipt_id
                ),
                None,
            )
        if receipt is None or receipt.receipt_artifact is None:
            raise ContractError("Gate B cannot resolve the accepted patch receipt")
        return request, receipt

    async def check(self, result: Any) -> Any:
        request, receipt = self._request_and_receipt(result)
        key = (
            "final"
            if request.command_id == "candidate_final_infer"
            else result.fidelity.value
        )
        gate = self.gates.get(key)
        if gate is None:
            raise ContractError("no Gate B population is frozen for %s" % key)
        expectation = ExecutionSealExpectation(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            execution_attempt=request.attempt,
            producer_commit_sha=request.patch_commit_sha,
            command_id=request.command_id,
            data_manifest_sha256=request.data_manifest_sha256,
            patch_receipt_id=request.patch_receipt_id,
            patch_receipt_sha256=receipt.receipt_artifact.sha256,
        )
        return await gate.check(result, expected_execution=expectation)


class ProtectedEvaluationBridge:
    """Adapt canonical ledger requests to the protected evaluation domain."""

    def __init__(
        self,
        *,
        config: RunConfig,
        event_store: EventStore,
        populations: Mapping[str, _PopulationData],
        baseline_predictions: Mapping[str, Path],
        evaluator_adapter: Any,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> None:
        self.config = config
        self.event_store = event_store
        self.populations = dict(populations)
        self.baseline_predictions = dict(baseline_predictions)
        self.evaluator_adapter = evaluator_adapter
        self.artifact_store = artifact_store
        self._domain_results: Dict[Tuple[str, int, str], DomainEvaluationResult] = {}
        self._active_references: Optional[Tuple[DomainMetricSet, DomainMetricSet, DomainMetricSet]] = None
        self.service = EvaluationService(
            evaluator_adapter,
            output_gate_resolver=self._resolve_gate,
            seed_result_resolver=self._resolve_seed_result,
        )

    def baseline_evaluation(self, contract_sha256: str) -> EvaluationResult:
        population = self.populations["full"]
        batch = self._batch_from_path(
            self.baseline_predictions["full"], population, "baseline_full"
        )
        metric_set = self.evaluator_adapter.score(
            batch,
            population.labels,
            self.config.evaluator_sha256,
            contract_sha256,
            Population.PUBLIC_VALIDATION,
        )
        return EvaluationResult(
            run_id=self.config.run_id,
            experiment_id="baseline",
            attempt=1,
            population=Population.PUBLIC_VALIDATION,
            fidelity=Fidelity.FULL,
            seed=self.config.seed_schedule[0],
            public_query_index=1,
            evaluator_sha256=self.config.evaluator_sha256,
            contract_sha256=contract_sha256,
            metric_set=metric_set.to_canonical(),
            baseline_delta=0.0,
            parent_delta=0.0,
            previous_best_delta=0.0,
            prediction_change=1.0,
            trust=TrustAssessment(
                verdict=TrustVerdict.ACCEPTED,
                stability=Stability.CONFIRMED,
                integrity=Integrity.CLEAN,
            ),
        )

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        key = request.fidelity.value
        population = self.populations[key]
        predictions = self._batch_from_artifact(request.prediction_artifact, population)
        baseline_batch = self._batch_from_path(
            self.baseline_predictions[key], population, "baseline_%s" % key
        )
        baseline = self._score_reference(baseline_batch, population, request)
        parent_id = self._parent_experiment_id(request.experiment_id)
        parent_batch = self._reference_batch(parent_id, key, population)
        parent = self._score_reference(parent_batch, population, request)
        state = project(self.event_store.read_events(repair_tail=True))
        best_batch = self._reference_batch(state.best_experiment_id, key, population)
        previous_best = self._score_reference(best_batch, population, request)
        execution_request = self._execution_request(request.output_checked_event_id)
        internal_proxy = self._internal_proxy_evidence(
            request, execution_request.patch_commit_sha
        )
        seed_events = (
            tuple(
                event.event_id
                for event in self.event_store.read_events(repair_tail=True)
                if event.payload.type == "evaluation.completed"
                and event.payload.result.experiment_id == request.experiment_id
                and event.payload.result.population == request.population
                and event.payload.result.fidelity == request.fidelity
                and event.payload.result.attempt < request.attempt
                and self._execution_request(event.causation_event_id).patch_commit_sha
                == execution_request.patch_commit_sha
            )
            if execution_request.command_id != "clean_reproduce"
            else ()
        )
        gate = self._resolve_gate(request.output_checked_event_id)
        inputs = EvaluationInputs(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt=request.attempt,
            output_gate=gate,
            predictions=predictions,
            population=request.population,
            fidelity=request.fidelity,
            seed=request.seed,
            public_query_index=request.public_query_index,
            evaluator_sha256=request.evaluator_sha256,
            contract_sha256=request.contract_sha256,
            data_manifest_sha256=self.config.data_manifest_sha256,
            labels=population.labels,
            baseline=baseline,
            parent=parent,
            previous_best=previous_best,
            parent_scores=parent_batch.scores,
            previous_best_scores=best_batch.scores,
            diagnostic_features=population.diagnostic_features,
            baseline_scores=baseline_batch.scores,
            training_diagnostics=self._training_diagnostics(
                request.output_checked_event_id
            ),
            seed_evaluation_event_ids=seed_events,
            internal_proxy_delta=(
                internal_proxy.parent_delta if internal_proxy is not None else None
            ),
            internal_proxy_ci_lower=(
                internal_proxy.trust.parent_delta_ci_lower
                if internal_proxy is not None
                else None
            ),
            internal_proxy_ci_upper=(
                internal_proxy.trust.parent_delta_ci_upper
                if internal_proxy is not None
                else None
            ),
        )
        self._active_references = (baseline, parent, previous_best)
        try:
            domain = self.service.evaluate(inputs)
        finally:
            self._active_references = None
        self._domain_results[(request.experiment_id, request.attempt, key)] = domain
        canonical = domain.to_canonical()
        if self.artifact_store is not None:
            artifact = self._write_diagnostics_artifact(domain)
            canonical = canonical.model_copy(update={"metrics_artifact": artifact})
        return canonical

    async def decide(
        self, result: EvaluationResult, context: EvaluationDecisionContext
    ) -> ExperimentDecision:
        key = (result.experiment_id, result.attempt, result.fidelity.value)
        domain = self._domain_results.get(key)
        if domain is None:
            raise ContractError("decision requires the exact protected evaluation result")
        evaluation_event = next(
            (
                event
                for event in reversed(self.event_store.read_events(repair_tail=True))
                if event.payload.type == "evaluation.completed"
                and event.payload.result.experiment_id == result.experiment_id
                and event.payload.result.attempt == result.attempt
                and event.payload.result.fidelity == result.fidelity
            ),
            None,
        )
        if evaluation_event is None:
            raise ContractError("decision requires a persisted evaluation event")
        prior_confirmations = len(domain.seed_evidence_event_ids)
        decision = __import__(
            "tacorank.evaluation.decisions", fromlist=["decide"]
        ).decide(
            domain,
            DecisionContext(
                evaluation_event_id=evaluation_event.event_id,
                supporting_event_ids=(
                    evaluation_event.causation_event_id,
                    evaluation_event.event_id,
                ),
                seed_evidence_event_ids=domain.seed_evidence_event_ids,
                confirmations_remaining=max(
                    0, self.config.max_confirmation_attempts - prior_confirmations
                ),
                promote_inconclusive_proxy=context.promote_inconclusive_proxy,
            ),
        )
        canonical = decision.to_canonical()
        spec = self._experiment_spec(result.experiment_id)
        execution = self._execution_request(evaluation_event.causation_event_id)
        lesson = build_research_lesson(
            domain,
            tuple(domain.seed_evidence_event_ids) + (evaluation_event.event_id,),
            (execution.patch_commit_sha,),
            spec.family,
            spec.hypothesis,
            spec.expected_mechanism,
            (
                "The frozen %s/%s evaluation frame for the %s stage."
                % (result.population.value, result.fidelity.value, spec.target_stage)
            ),
            (
                "Do not generalize beyond this frame; the proposal's falsification "
                "condition was: %s" % spec.falsification_condition
            ),
            self._research_frame_id(spec, decision.decision.value),
        )
        if lesson is not None:
            canonical = canonical.model_copy(update={"lesson_candidate": lesson})
        return canonical

    def _experiment_spec(self, experiment_id: str) -> Any:
        event = next(
            (
                item
                for item in self.event_store.read_events(repair_tail=True)
                if item.payload.type == "experiment.proposed"
                and item.payload.spec.experiment_id == experiment_id
            ),
            None,
        )
        if event is None:
            raise ContractError("evaluation cannot resolve its experiment proposal")
        return event.payload.spec

    def _research_frame_id(self, spec: Any, decision: str) -> str:
        if spec.family == "objective" and decision == "accept":
            return spec.experiment_id
        proposals = {
            event.payload.spec.experiment_id: event.payload.spec
            for event in self.event_store.read_events(repair_tail=True)
            if event.payload.type == "experiment.proposed"
        }
        parent_id = spec.parent_experiment_id
        while parent_id and parent_id != "baseline":
            parent = proposals.get(parent_id)
            if parent is None:
                break
            if parent.family == "objective":
                return parent.experiment_id
            parent_id = parent.parent_experiment_id
        return "baseline"

    def _resolve_gate(self, event_id: str) -> OutputGateEvidence:
        event = next(
            (
                item
                for item in self.event_store.read_events(repair_tail=True)
                if item.event_id == event_id and item.payload.type == "output.checked"
            ),
            None,
        )
        if event is None:
            raise KeyError(event_id)
        result = event.payload.result
        execution = next(
            item.payload.request
            for item in reversed(self.event_store.read_events(repair_tail=True))
            if item.payload.type == "execution.started"
            and item.payload.request.experiment_id == result.experiment_id
            and item.payload.request.attempt == result.attempt
        )
        population = (
            Population.INTERNAL_PROXY
            if execution.fidelity == Fidelity.PROXY
            else Population.PUBLIC_VALIDATION
        )
        return OutputGateEvidence(
            event_id=event.event_id,
            accepted=result.accepted,
            prediction_artifact_id=result.prediction_artifact.artifact_id,
            prediction_artifact_sha256=result.prediction_artifact.sha256,
            population=population,
            ordered_row_identity_sha256=result.ordered_row_identity_sha256,
            ordered_prediction_sha256=result.ordered_prediction_sha256,
        )

    def _execution_request(self, output_event_id: str) -> Any:
        output = next(
            (
                event
                for event in self.event_store.read_events(repair_tail=True)
                if event.event_id == output_event_id
                and event.payload.type == "output.checked"
            ),
            None,
        )
        if output is None:
            raise ContractError("evaluation output evidence is missing")
        finished_id = output.causation_event_id
        finished = next(
            (
                event
                for event in self.event_store.read_events(repair_tail=True)
                if event.event_id == finished_id
                and event.payload.type == "execution.finished"
            ),
            None,
        )
        if finished is None:
            raise ContractError("evaluation execution evidence is missing")
        started = next(
            (
                event.payload.request
                for event in self.event_store.read_events(repair_tail=True)
                if event.event_id == finished.causation_event_id
                and event.payload.type == "execution.started"
            ),
            None,
        )
        if started is None:
            raise ContractError("evaluation request evidence is missing")
        return started

    def _training_diagnostics(self, output_event_id: str) -> Mapping[str, float]:
        output = next(
            (
                event
                for event in self.event_store.read_events(repair_tail=True)
                if event.event_id == output_event_id
                and event.payload.type == "output.checked"
            ),
            None,
        )
        finished = next(
            (
                event
                for event in self.event_store.read_events(repair_tail=True)
                if output is not None
                and event.event_id == output.causation_event_id
                and event.payload.type == "execution.finished"
            ),
            None,
        )
        reference = (
            finished.payload.result.checkpoint_artifact
            if finished is not None
            else None
        )
        if reference is None or self.artifact_store is None:
            return {}
        path = self.artifact_store.verify(reference)
        if path.stat().st_size > 64 * 1024:
            raise ContractError("training diagnostics exceed the bounded size")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ContractError("training diagnostics must be a JSON object")
        allowed = {
            "train_rows",
            "interaction_coverage",
            "loss_start",
            "loss_end",
            "pairwise_accuracy",
            "gradient_norm",
            "residual_mean",
            "residual_std",
        }
        values: Dict[str, float] = {}
        for key in allowed:
            value = document.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ContractError("training diagnostics must be finite")
            values[key] = parsed
        return values

    def _resolve_seed_result(self, event_id: str) -> DomainEvaluationResult:
        event = next(
            (
                item
                for item in self.event_store.read_events(repair_tail=True)
                if item.event_id == event_id and item.payload.type == "evaluation.completed"
            ),
            None,
        )
        if event is None or self._active_references is None:
            raise KeyError(event_id)
        canonical = event.payload.result
        baseline, parent, previous_best = self._active_references
        metrics = DomainMetricSet(
            canonical.metric_set.metrics,
            canonical.metric_set.primary_metric_name,
            canonical.metric_set.primary_score,
        )
        change = canonical.prediction_change
        return DomainEvaluationResult(
            run_id=canonical.run_id,
            experiment_id=canonical.experiment_id,
            attempt=canonical.attempt,
            population=canonical.population,
            fidelity=canonical.fidelity,
            seed=canonical.seed,
            public_query_index=canonical.public_query_index,
            evaluator_sha256=canonical.evaluator_sha256,
            contract_sha256=canonical.contract_sha256,
            data_manifest_sha256=self.config.data_manifest_sha256,
            metric_set=metrics,
            baseline_delta=compare_metric_sets(metrics, baseline),
            parent_delta=compare_metric_sets(metrics, parent),
            previous_best_delta=compare_metric_sets(metrics, previous_best),
            prediction_change=DomainPredictionChange(
                change.spearman_vs_parent,
                change.changed_row_fraction,
                None,
                1.0,
            ),
            diagnostic_metrics=dict(canonical.diagnostic_metrics),
            trust=DomainTrustAssessment(
                verdict=canonical.trust.verdict,
                stability=canonical.trust.stability,
                integrity=canonical.trust.integrity,
                flags=tuple(canonical.trust.flags),
                eta_applied=canonical.trust.eta_applied,
                seed_mean=canonical.trust.seed_mean,
                seed_stderr=canonical.trust.seed_stderr,
                seed_count=canonical.trust.seed_count,
                parent_delta_mean=canonical.trust.parent_delta_mean,
                parent_delta_stderr=canonical.trust.parent_delta_stderr,
                parent_delta_ci_lower=canonical.trust.parent_delta_ci_lower,
                parent_delta_ci_upper=canonical.trust.parent_delta_ci_upper,
                best_delta_mean=canonical.trust.best_delta_mean,
                best_delta_stderr=canonical.trust.best_delta_stderr,
                best_delta_ci_lower=canonical.trust.best_delta_ci_lower,
                best_delta_ci_upper=canonical.trust.best_delta_ci_upper,
                minimum_practical_gain=canonical.trust.minimum_practical_gain,
            ),
            diagnostics=canonical.diagnostics,
            seed_evidence_event_ids=tuple(canonical.seed_evidence_event_ids),
        )

    def _internal_proxy_evidence(
        self, request: EvaluationRequest, patch_commit_sha: str
    ) -> Optional[EvaluationResult]:
        if (
            request.population != Population.PUBLIC_VALIDATION
            or request.fidelity != Fidelity.FULL
        ):
            return None
        for event in reversed(self.event_store.read_events(repair_tail=True)):
            if event.payload.type != "evaluation.completed":
                continue
            result = event.payload.result
            if (
                result.experiment_id != request.experiment_id
                or result.population != Population.INTERNAL_PROXY
                or result.fidelity != Fidelity.PROXY
            ):
                continue
            execution = self._execution_request(event.causation_event_id)
            if (
                execution.patch_commit_sha == patch_commit_sha
                and execution.seed == request.seed
            ):
                return result
        return None

    def _internal_proxy_delta(
        self, request: EvaluationRequest, patch_commit_sha: str
    ) -> Optional[float]:
        """Compatibility projection for callers that only need the delta."""

        result = self._internal_proxy_evidence(request, patch_commit_sha)
        return result.parent_delta if result is not None else None

    def _write_diagnostics_artifact(
        self, result: DomainEvaluationResult
    ) -> Any:
        assert self.artifact_store is not None
        content = (
            json.dumps(
                {
                    "schema_version": "1.0",
                    "run_id": result.run_id,
                    "experiment_id": result.experiment_id,
                    "attempt": result.attempt,
                    "population": result.population.value,
                    "fidelity": result.fidelity.value,
                    "diagnostic_metrics": dict(result.diagnostic_metrics),
                    "diagnostics": result.diagnostics.model_dump(mode="json"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        prefix = experiment_artifact_prefix(
            result.run_id, result.experiment_id, attempt=result.attempt
        )
        return self.artifact_store.write(
            artifact_id="metrics_%s_%03d_%s"
            % (result.experiment_id, result.attempt, result.fidelity.value),
            kind=ArtifactKind.METRICS,
            relative_path="%s/evaluation/%s-diagnostics.json"
            % (prefix, result.fidelity.value),
            content=content,
            content_type="application/json",
        )

    def _score_reference(
        self, batch: Any, population: _PopulationData, request: EvaluationRequest
    ) -> DomainMetricSet:
        return self.evaluator_adapter.score(
            batch,
            population.labels,
            request.evaluator_sha256,
            request.contract_sha256,
            request.population,
        )

    def _parent_experiment_id(self, experiment_id: str) -> Optional[str]:
        event = next(
            item
            for item in self.event_store.read_events(repair_tail=True)
            if item.payload.type == "experiment.proposed"
            and item.payload.spec.experiment_id == experiment_id
        )
        return event.payload.spec.parent_experiment_id

    def _reference_batch(
        self,
        experiment_id: Optional[str],
        key: str,
        population: _PopulationData,
    ) -> Any:
        if not experiment_id or experiment_id == "baseline":
            return self._batch_from_path(
                self.baseline_predictions[key], population, "baseline_%s" % key
            )
        output = next(
            (
                event.payload.result
                for event in reversed(self.event_store.read_events(repair_tail=True))
                if event.payload.type == "output.checked"
                and event.payload.result.experiment_id == experiment_id
                and event.payload.result.accepted
                and self._output_fidelity(event) == key
            ),
            None,
        )
        if output is None:
            raise ContractError(
                "reference experiment %s has no accepted %s prediction" % (experiment_id, key)
            )
        return self._batch_from_artifact(output.prediction_artifact, population)

    def _output_fidelity(self, output_event: Any) -> str:
        finished = next(
            item.payload.result
            for item in reversed(self.event_store.read_events(repair_tail=True))
            if item.event_id == output_event.causation_event_id
        )
        return finished.fidelity.value

    def _batch_from_artifact(self, artifact: Any, population: _PopulationData) -> Any:
        artifact.verify_file(self.config.repository_root, self.config.artifact_roots)
        return self._batch_from_path(
            self.config.repository_root / artifact.path,
            population,
            artifact.artifact_id,
        )

    @staticmethod
    def _batch_from_path(path: Path, population: _PopulationData, artifact_id: str) -> Any:
        checked = validate_submission(path, population.rows)
        return checked.prediction_batch(artifact_id)


def build_live_adapters(
    *,
    config: RunConfig,
    verified: VerifiedContract,
    live: LiveAdapterConfig,
    event_store: EventStore,
    artifact_store: ArtifactStore,
) -> LiveAdapters:
    """Build every production port or fail before the run is bootstrapped."""

    root = config.repository_root.resolve(strict=True)
    _require_clean_baseline(root, config.baseline_commit_sha)
    _require_live_files(config, live)
    _verify_data_manifest(config, live)
    user_history, item_popularity = _load_training_profiles(
        live.input_roots["candidate_full"] / "train.csv"
    )
    populations = {
        key: _load_population(
            path,
            feature_path=live.input_roots["candidate_%s" % key] / "score.csv",
            user_history=user_history,
            item_popularity=item_popularity,
        )
        for key, path in live.population_csvs.items()
    }
    final_rows = _load_submission_rows(live.contract_root / "submission_rows.csv")
    populations["final"] = _PopulationData(final_rows, ())
    population_manifests = _protected_population_manifests(populations)
    worktrees = WorktreeManager(
        root, live.worktree_root, required_submodules=live.required_submodules
    )
    worktrees.preflight(config.baseline_commit_sha)
    manifest = ProtectedManifest.from_markdown(
        root / config.protected_paths_path,
        root,
        contract_paths=(config.contract_path,),
        data_manifest_sha256=config.data_manifest_sha256,
        expected_manifest_sha256=verified.protected_paths_sha256,
    )
    artifact_root = run_artifact_root(config.run_id)
    receipts = ReceiptStore(
        root,
        artifact_root=artifact_root,
        include_run_id=False,
    )
    data_policy = DataAccessPolicy(
        views=tuple(
            DataViewPolicy(
                view_id=command_id,
                allowed_columns=tuple(live.candidate_allowed_columns),
                allowed_path_prefixes=("/inputs",),
            )
            for command_id in sorted(live.input_roots)
        ),
        protected_columns=tuple(live.protected_columns),
        hidden_path_tokens=tuple(live.hidden_path_tokens),
        future_column_patterns=tuple(live.future_column_patterns),
    )
    trae_config = _trae_config_from_mapping(live.trae)
    coding_worker = TraeCodingWorker(
        worktrees=worktrees,
        artifact_repository_root=root,
        config=trae_config,
        identity_resolver=LedgerCandidateIdentityResolver(event_store),
    )
    coding_worker.preflight()
    gate_a_smoke = DockerEntrypointSmokeCheck(
        docker_executable=live.docker_executable,
        docker_host=live.docker_host,
        image=live.docker_image,
        container_python_executable=live.container_python_executable,
        entrypoint=live.candidate_entrypoint,
        timeout_seconds=min(120, config.timeout_profiles.get("standard", 600)),
        memory_limit_mb=min(2048, trae_config.docker_memory_limit_mb),
        pids_limit=min(64, trae_config.docker_pids_limit),
        cpu_limit=min(1.0, live.docker_cpu_count),
        tmpfs_limit_mb=min(128, live.docker_tmpfs_size_mb),
    )
    patch_gate = WorktreePatchGate(
        worktrees=worktrees,
        repository_root=root,
        event_store=event_store,
        editable_roots=config.editable_roots,
        protected_manifest=manifest,
        receipt_store=receipts,
        data_access_policy=data_policy,
        allowed_command_ids=config.command_ids,
        artifact_roots=config.artifact_roots,
        allowed_import_roots=live.allowed_import_roots,
        allowed_capability_imports=live.allowed_capability_imports,
        allowed_dependency_changes=live.allowed_dependency_changes,
        smoke_check=gate_a_smoke,
    )
    pipeline = PipelineCommandInputs(
        contract_root=live.contract_root,
        input_roots=live.input_roots,
        baseline_entrypoint=live.baseline_entrypoint,
        candidate_entrypoint=live.candidate_entrypoint,
        submission_check_entrypoint=live.submission_check_entrypoint,
    )
    commands = default_command_registry(
        pipeline,
        python_executable=str(live.python_executable),
        container_python_executable=live.container_python_executable,
    )
    execution_artifacts = CanonicalArtifactStoreAdapter(
        artifact_store,
        artifact_root=artifact_root,
        include_run_id=False,
    )
    read_only_roots = tuple(
        dict.fromkeys(
            [live.contract_root.resolve(strict=True)]
            + [path.resolve(strict=True) for path in live.input_roots.values()]
        )
    )
    sandbox = DockerSandbox(
        SandboxPolicy(
            allowed_workspace_roots=(live.worktree_root.resolve(strict=True),),
            allowed_artifact_roots=(execution_artifacts.artifact_root,),
            allowed_read_only_roots=read_only_roots,
            allow_network=False,
        ),
        image=live.docker_image,
        docker_executable=live.docker_executable,
        docker_host=live.docker_host,
        cpu_count=live.docker_cpu_count,
        tmpfs_size_mb=live.docker_tmpfs_size_mb,
        mount_policies=_mount_policies(config, live),
        output_quota_max_bytes=live.output_quota_max_bytes,
        output_quota_verifier=None,
        image_environment_sha256=live.docker_image_environment_sha256,
    )
    sandbox.preflight(execution_artifacts.artifact_root)
    runner = ExecutionRunner(
        repository_root=root,
        artifacts=execution_artifacts,
        commands=commands,
        sandbox=sandbox,
        workspace_resolver=worktrees.path_for,
        seal_verifier=SealedExecutionVerifier(
            worktrees=worktrees,
            receipts=receipts,
            protected_manifest=manifest,
            receipt_artifact_resolver=LedgerReceiptArtifactResolver(event_store),
        ),
        submission_artifact_resolver=LedgerSubmissionArtifactResolver(event_store),
        policy=RunnerPolicy(
            max_timeout_seconds=max(config.timeout_profiles.values()),
        ),
    )
    evaluator_adapter = create_evaluator_adapter(
        root,
        expected_evaluator_sha256=config.evaluator_sha256,
        expected_contract_sha256=verified.contract_sha256,
        expected_data_manifest_sha256=config.data_manifest_sha256,
        population_manifests=population_manifests,
    )
    evaluator = ProtectedEvaluationBridge(
        config=config,
        event_store=event_store,
        populations=populations,
        baseline_predictions=live.baseline_prediction_csvs,
        evaluator_adapter=evaluator_adapter,
        artifact_store=artifact_store,
    )
    baseline = evaluator.baseline_evaluation(verified.contract_sha256)
    config.validate_metric_set(baseline.metric_set)
    return LiveAdapters(
        coding_worker=coding_worker,
        patch_gate=patch_gate,
        runner=runner,
        health_observer=SREObserver(),
        recovery_manager=RecoveryManager(),
        output_gate=LedgerOutputGate(
            repository_root=root,
            event_store=event_store,
            populations=populations,
            artifact_roots=config.artifact_roots,
            max_single_score_fraction=config.max_single_score_fraction,
        ),
        evaluator=evaluator,
        baseline=baseline,
        final_submission_provider=ProtectedBaselineFinalSubmission(
            config=config,
            artifact_store=artifact_store,
            source=live.baseline_final_prediction_csv,
            rows=final_rows,
        ),
    )


def _load_population(
    path: Path,
    *,
    feature_path: Optional[Path] = None,
    user_history: Optional[Mapping[str, int]] = None,
    item_popularity: Optional[Mapping[str, int]] = None,
) -> _PopulationData:
    rows: List[Mapping[str, Any]] = []
    labels: List[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"row_id", "user_id", "video_id", "label"}
        if not required.issubset(reader.fieldnames or ()):
            raise ContractError(
                "population CSV must contain row_id,user_id,video_id,label"
            )
        for expected, record in enumerate(reader):
            if int(record["row_id"]) != expected:
                raise ContractError("population row_id must be contiguous and ordered")
            label = int(record["label"])
            if label not in (0, 1):
                raise ContractError("population labels must be binary")
            rows.append(
                {
                    "row_id": expected,
                    "user_id": record["user_id"],
                    "video_id": record["video_id"],
                }
            )
            labels.append(label)
    if not rows:
        raise ContractError("population CSV must not be empty")
    diagnostic_features = None
    if feature_path is not None:
        if user_history is None or item_popularity is None:
            raise ContractError("diagnostic population features require training profiles")
        diagnostic_features = _load_diagnostic_features(
            feature_path,
            rows,
            user_history=user_history,
            item_popularity=item_popularity,
        )
    return _PopulationData(tuple(rows), tuple(labels), diagnostic_features)


def _load_training_profiles(path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    user_history: Dict[str, int] = {}
    item_popularity: Dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"user_id", "video_id"}
        if not required.issubset(reader.fieldnames or ()):
            raise ContractError("training diagnostics require user_id and video_id")
        for record in reader:
            user_id = record["user_id"]
            video_id = record["video_id"]
            user_history[user_id] = user_history.get(user_id, 0) + 1
            item_popularity[video_id] = item_popularity.get(video_id, 0) + 1
    if not user_history or not item_popularity:
        raise ContractError("training diagnostics require non-empty training data")
    return user_history, item_popularity


def _load_diagnostic_features(
    path: Path,
    population_rows: Sequence[Mapping[str, Any]],
    *,
    user_history: Mapping[str, int],
    item_popularity: Mapping[str, int],
) -> DiagnosticFeatures:
    dates: List[int] = []
    durations: List[float] = []
    popularities: List[int] = []
    histories: List[int] = []
    user_ids: List[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"row_id", "date", "user_id", "video_id", "duration_ms"}
        if not required.issubset(reader.fieldnames or ()):
            raise ContractError("diagnostic score view is missing required columns")
        for expected, record in enumerate(reader):
            if expected >= len(population_rows):
                raise ContractError("diagnostic score view has extra rows")
            population = population_rows[expected]
            if (
                int(record["row_id"]) != expected
                or record["user_id"] != str(population["user_id"])
                or record["video_id"] != str(population["video_id"])
            ):
                raise ContractError("diagnostic score view does not match population order")
            try:
                date = int(record["date"])
                duration = float(record["duration_ms"])
            except (TypeError, ValueError, OverflowError) as error:
                raise ContractError("diagnostic score features must be numeric") from error
            if duration < 0 or not math.isfinite(duration):
                raise ContractError("diagnostic duration must be finite and non-negative")
            user_id = record["user_id"]
            video_id = record["video_id"]
            dates.append(date)
            durations.append(duration)
            histories.append(int(user_history.get(user_id, 0)))
            popularities.append(int(item_popularity.get(video_id, 0)))
            user_ids.append(user_id)
    if len(dates) != len(population_rows):
        raise ContractError("diagnostic score view row count does not match population")
    val_a, val_b = split_validation_indices(user_ids)
    arms = [""] * len(user_ids)
    for index in val_a:
        arms[index] = "val_a"
    for index in val_b:
        arms[index] = "val_b"
    return DiagnosticFeatures(
        dates=tuple(dates),
        duration_ms=tuple(durations),
        item_popularity=tuple(popularities),
        user_history_count=tuple(histories),
        validation_arms=tuple(arms),
    )


def _load_submission_rows(path: Path) -> Tuple[Mapping[str, Any], ...]:
    rows: List[Mapping[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != ("row_id", "user_id", "video_id"):
            raise ContractError("submission rows have an invalid header")
        for expected, record in enumerate(reader):
            if int(record["row_id"]) != expected:
                raise ContractError("submission row_id must be contiguous and ordered")
            rows.append(
                {
                    "row_id": expected,
                    "user_id": record["user_id"],
                    "video_id": record["video_id"],
                }
            )
    if not rows:
        raise ContractError("submission rows must not be empty")
    return tuple(rows)


def _protected_population_manifests(
    populations: Mapping[str, _PopulationData],
) -> Mapping[Population, PopulationManifest]:
    """Bind evaluable populations to their protected row order and identity."""

    manifests: Dict[Population, PopulationManifest] = {}
    for population, key in (
        (Population.INTERNAL_PROXY, "proxy"),
        (Population.PUBLIC_VALIDATION, "full"),
    ):
        rows = populations[key].rows
        manifests[population] = PopulationManifest(
            rows=len(rows),
            ordered_row_identity_sha256=ordered_row_identity_sha256(
                tuple(row["row_id"] for row in rows),
                tuple(row["user_id"] for row in rows),
                tuple(row["video_id"] for row in rows),
            ),
        )
    return manifests


def _output_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_single_score_fraction: float = 0.5,
) -> OutputContract:
    return OutputContract(
        columns=(
            OutputColumn("row_id", "integer"),
            OutputColumn("user_id", "string"),
            OutputColumn("video_id", "string"),
            OutputColumn("score", "number"),
        ),
        score_column="score",
        expected_rows=tuple(rows),
        identity_columns=("user_id", "video_id"),
        row_id_column="row_id",
        forbidden_columns=("label", "target"),
        maximum_single_score_fraction=max_single_score_fraction,
    )


def _mount_policies(
    config: RunConfig, live: LiveAdapterConfig
) -> Tuple[ContainerMountPolicy, ...]:
    fidelities = {
        "baseline_full": "full",
        "candidate_smoke": "smoke",
        "candidate_proxy": "proxy",
        "candidate_full": "full",
        "candidate_final_infer": "full",
        "clean_reproduce": "full",
    }
    return tuple(
        ContainerMountPolicy(
            command_id=command_id,
            fidelity=fidelities[command_id],
            data_manifest_sha256=config.data_manifest_sha256,
            mounts=(
                ContainerReadOnlyMount(
                    live.contract_root.resolve(strict=True),
                    "/contracts",
                    "contract",
                ),
                ContainerReadOnlyMount(
                    live.input_roots[command_id].resolve(strict=True),
                    "/inputs",
                    (
                        "hidden_inference_data"
                        if command_id == "candidate_final_infer"
                        else "candidate_data"
                    ),
                ),
            ),
        )
        for command_id in sorted(fidelities)
    )


def _require_live_files(config: RunConfig, live: LiveAdapterConfig) -> None:
    if config.command_ids != [
        "candidate_smoke",
        "candidate_proxy",
        "candidate_full",
    ]:
        raise ContractError(
            "live experiment routing requires candidate_smoke,candidate_proxy,candidate_full"
        )
    for label, path in (
        ("contract_root", live.contract_root),
        ("python_executable", live.python_executable),
        ("docker_executable", live.docker_executable),
        ("data_manifest_path", live.data_manifest_path),
        ("baseline final prediction", live.baseline_final_prediction_csv),
        *[("input root", value) for value in live.input_roots.values()],
        *[("population CSV", value) for value in live.population_csvs.values()],
        *[("baseline prediction", value) for value in live.baseline_prediction_csvs.values()],
    ):
        if not path.exists():
            raise ContractError("%s does not exist: %s" % (label, path))
    _verify_executable_baseline_parity(config, live)
    actual_evaluator = _sha256_file(
        config.repository_root / "kuairand-starter-kit" / "evaluate.py"
    )
    if actual_evaluator != config.evaluator_sha256:
        raise ContractError("frozen evaluator_sha256 does not match official evaluator")
    if set(config.metric_names) != {"GAUC", "nDCG@5"} or (
        config.primary_metric_name != "primary"
    ):
        raise ContractError("live KuaiRand evaluation requires GAUC,nDCG@5 and primary")
    if live.trae.get("docker_image") != live.docker_image:
        raise ContractError("coding and execution must use the same pinned Docker image")
    trae_docker = live.trae.get("docker_executable")
    if trae_docker is None or Path(str(trae_docker)).resolve() != live.docker_executable.resolve():
        raise ContractError("coding and execution must use the same Docker executable")
    if live.trae.get("docker_host") != live.docker_host:
        raise ContractError("coding and execution must use the same Docker host socket")
    for field, configured in (
        ("max_steps_cap", config.coding_step_limit),
        ("max_wall_time_seconds_cap", config.coding_wall_time_limit_seconds),
    ):
        if int(live.trae.get(field, 0)) < configured:
            raise ContractError("run coding limit exceeds the reviewed Trae %s" % field)
    reviewed_token_cap = live.trae.get("max_token_cap")
    if config.coding_token_limit is None:
        if reviewed_token_cap is not None:
            raise ContractError(
                "uncapped run coding tokens require a null reviewed Trae token cap"
            )
    elif reviewed_token_cap is not None and int(reviewed_token_cap) < config.coding_token_limit:
        raise ContractError("run coding limit exceeds the reviewed Trae max_token_cap")


def _verify_executable_baseline_parity(
    config: RunConfig, live: LiveAdapterConfig
) -> None:
    """Require executable proof before exposing baseline-parity methods."""

    receipt_path = live.baseline_parity_receipt_path
    claims_parity = "baseline_parity" in config.research_capabilities
    if receipt_path is None:
        if claims_parity:
            raise ContractError(
                "baseline_parity requires a setup-generated executable receipt"
            )
        return
    try:
        document = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("baseline parity receipt is invalid JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "candidate_entrypoint",
        "candidate_source_path",
        "candidate_source_sha256",
        "candidate_support_sha256",
        "routes",
    }:
        raise ContractError("baseline parity receipt has an invalid schema")
    if (
        document["schema_version"] != "1.0"
        or document["candidate_entrypoint"] != live.candidate_entrypoint
        or document["candidate_source_path"] != "solution/candidate.py"
    ):
        raise ContractError("baseline parity receipt identifies a different candidate")
    candidate_source = config.repository_root / "solution" / "candidate.py"
    if (
        not candidate_source.is_file()
        or candidate_source.is_symlink()
        or document["candidate_source_sha256"] != _sha256_file(candidate_source)
    ):
        raise ContractError("baseline parity candidate source identity changed")
    support = document["candidate_support_sha256"]
    expected_support = {
        relative: _sha256_file(config.repository_root / relative)
        for relative in (
            "solution/experiment_config.py",
            "solution/research_scaffold.py",
        )
    }
    if support != expected_support:
        raise ContractError("baseline parity candidate support identity changed")
    routes = document["routes"]
    expected_routes = {
        "candidate_smoke",
        "candidate_proxy",
        "candidate_full",
        "candidate_final_infer",
    }
    if not isinstance(routes, dict) or set(routes) != expected_routes:
        raise ContractError("baseline parity receipt does not cover every candidate route")
    for command_id in sorted(expected_routes):
        record = routes[command_id]
        if not isinstance(record, dict) or set(record) != {
            "fm_prediction_sha256",
            "candidate_output_sha256",
            "exact_bytes_match",
        }:
            raise ContractError("baseline parity route receipt is malformed")
        view = live.input_roots[command_id]
        prediction = view / "fm_baseline_predictions.csv"
        digest_file = view / "fm_baseline_predictions.sha256"
        if (
            not prediction.is_file()
            or prediction.is_symlink()
            or not digest_file.is_file()
            or digest_file.is_symlink()
        ):
            raise ContractError("baseline parity route is missing frozen FM inputs")
        digest = _sha256_file(prediction)
        try:
            frozen_digest = digest_file.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError) as error:
            raise ContractError("baseline parity digest file is invalid") from error
        if (
            frozen_digest != digest
            or record["fm_prediction_sha256"] != digest
            or record["candidate_output_sha256"] != digest
            or record["exact_bytes_match"] is not True
        ):
            raise ContractError("baseline parity route no longer reproduces official FM")
    if not claims_parity:
        raise ContractError(
            "verified executable FM parity exists but run capabilities omit it"
        )


def _require_clean_baseline(root: Path, baseline_commit_sha: str) -> None:
    def git(*args: str) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=False,
                close_fds=True,
                timeout=15.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ContractError("Git baseline preflight failed") from error

    head = git("rev-parse", "--verify", "HEAD^{commit}")
    baseline = git("rev-parse", "--verify", baseline_commit_sha + "^{commit}")
    if (
        head.returncode != 0
        or baseline.returncode != 0
        or head.stdout.strip() != baseline.stdout.strip()
    ):
        raise ContractError("baseline_commit_sha must equal the production checkout HEAD")
    tracked = git("status", "--porcelain=v1", "--untracked-files=no")
    if tracked.returncode != 0 or tracked.stdout:
        raise ContractError("production checkout has uncommitted tracked changes")


def _verify_data_manifest(config: RunConfig, live: LiveAdapterConfig) -> None:
    raw = live.data_manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.data_manifest_sha256:
        raise ContractError("data manifest bytes do not match data_manifest_sha256")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("data manifest is not valid UTF-8 JSON") from error
    if not isinstance(document, dict) or set(document) != {"schema_version", "files"}:
        raise ContractError("data manifest has an invalid schema")
    if document.get("schema_version") != "1.0" or not isinstance(
        document.get("files"), list
    ):
        raise ContractError("data manifest has an invalid schema")

    root = config.repository_root.resolve(strict=True)
    attested = set()
    for record in document["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise ContractError("data manifest file record is invalid")
        try:
            relative = normalize_relative_path(record["path"])
        except (TypeError, ValueError) as error:
            raise ContractError("data manifest contains an invalid path") from error
        if relative in attested:
            raise ContractError("data manifest contains duplicate paths")
        expected_hash = record["sha256"]
        expected_size = record["size_bytes"]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ContractError("data manifest file identity is invalid")
        target = root.joinpath(*relative.split("/"))
        if target.is_symlink() or not target.is_file():
            raise ContractError("data manifest file is missing: %s" % relative)
        resolved = target.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ContractError("data manifest file escaped the repository") from error
        if (
            resolved != target
            or target.stat().st_size != expected_size
            or _sha256_file(target) != expected_hash
        ):
            raise ContractError("data manifest file identity changed: %s" % relative)
        attested.add(relative)

    required_files = set()
    for directory in live.input_roots.values():
        for path in directory.rglob("*"):
            if path.is_file() and not path.is_symlink():
                required_files.add(path.resolve(strict=True))
    required_files.update(path.resolve(strict=True) for path in live.population_csvs.values())
    required_files.update(
        path.resolve(strict=True) for path in live.baseline_prediction_csvs.values()
    )
    required_files.add(live.baseline_final_prediction_csv.resolve(strict=True))
    if live.baseline_parity_receipt_path is not None:
        required_files.add(live.baseline_parity_receipt_path.resolve(strict=True))
    for path in required_files:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ContractError("all production data must be inside repository_root") from error
        if relative not in attested:
            raise ContractError("production data is not attested by the manifest: %s" % relative)


def _resolve_config_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
