"""Production adapter composition for the canonical TacoRank harness.

This module is intentionally the only place where ledger-owned identities are
translated into the narrower production adapter contracts.  No adapter is
allowed to guess an attempt, receipt, population, or reference prediction.
"""

from __future__ import annotations

import csv
import hashlib
import json
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
    EvaluationInputs,
    EvaluationService,
    MetricSet as DomainMetricSet,
    OutputGateEvidence,
)
from ..evaluation.comparisons import compare_metric_sets
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
from ..safety import (
    DataAccessPolicy,
    DataViewPolicy,
    ExecutionSealExpectation,
    OutputColumn,
    OutputContract,
    OutputGate,
    PatchGate,
    ProtectedManifest,
    ReceiptStore,
)
from ..schemas import (
    EvaluationDecisionContext,
    EvaluationRequest,
    EvaluationResult,
    ExperimentDecision,
    Fidelity,
    Integrity,
    Population,
    Stability,
    StrictModel,
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
_TRAE_PATH_KEYS = (
    "config_file",
    "docker_executable",
    "trae_install_root",
    "trae_install_identity_file",
    "trae_runtime_root",
    "python_dotenv_metadata_file",
)


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
        ):
            raw[key] = str(_resolve_config_path(base, raw[key]))
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


def _trae_config_from_mapping(values: Mapping[str, Any]) -> TraeConfig:
    """Normalize JSON-shaped live values at the Trae dataclass boundary."""

    normalized = dict(values)
    try:
        normalized["command_prefix"] = tuple(normalized["command_prefix"])
        for key in _TRAE_PATH_KEYS:
            if normalized.get(key) is not None:
                normalized[key] = Path(normalized[key])
        for key in (
            "repair_allowed_command_ids",
            "approved_environment_names",
            "credential_environment_names",
        ):
            if key in normalized:
                normalized[key] = tuple(normalized[key])
        if "credential_environment_aliases" in normalized:
            normalized["credential_environment_aliases"] = tuple(
                tuple(item) for item in normalized["credential_environment_aliases"]
            )
        return TraeConfig(**normalized)
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("live Trae configuration has invalid field types") from error


@dataclass(frozen=True)
class _PopulationData:
    rows: Tuple[Mapping[str, Any], ...]
    labels: Tuple[int, ...]


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
        return CandidateIdentity(1, self._proposal_event_id(spec.experiment_id))

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
        }

    async def check(self, candidate: Any) -> Any:
        workspace = self.worktrees.path_for(candidate.run_id, candidate.experiment_id)
        # The cumulative-diff root is the approved ExperimentSpec parent.  It
        # is supplied by the wrapper because PatchCandidate deliberately does
        # not duplicate controller-owned proposal fields.
        experiment_root = self._experiment_root(candidate.experiment_id)
        return await PatchGate(repository_root=workspace, **self.options).check(
            candidate,
            experiment_root_commit_sha=experiment_root,
        )

    def _experiment_root(self, experiment_id: str) -> str:
        matches = [
            event.payload.spec.parent_commit_sha
            for event in self.event_store.read_events(repair_tail=True)
            if event.payload.type == "experiment.proposed"
            and event.payload.spec.experiment_id == experiment_id
        ]
        if len(matches) != 1:
            raise ContractError("Gate A cannot resolve the experiment root commit")
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


class LedgerOutputGate:
    def __init__(
        self,
        *,
        repository_root: Path,
        event_store: EventStore,
        populations: Mapping[str, _PopulationData],
        artifact_roots: Sequence[str],
    ) -> None:
        self.repository_root = repository_root
        self.event_store = event_store
        self.artifact_roots = tuple(artifact_roots)
        self.gates = {
            key: OutputGate(
                repository_root=repository_root,
                artifact_roots=artifact_roots,
                contract=_output_contract(population.rows),
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
        key = result.fidelity.value
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
    ) -> None:
        self.config = config
        self.event_store = event_store
        self.populations = dict(populations)
        self.baseline_predictions = dict(baseline_predictions)
        self.evaluator_adapter = evaluator_adapter
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
        seed_events = tuple(
            event.event_id
            for event in self.event_store.read_events(repair_tail=True)
            if event.payload.type == "evaluation.completed"
            and event.payload.result.experiment_id == request.experiment_id
            and event.payload.result.population == request.population
            and event.payload.result.fidelity == request.fidelity
            and event.payload.result.attempt < request.attempt
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
            seed_evaluation_event_ids=seed_events,
        )
        self._active_references = (baseline, parent, previous_best)
        try:
            domain = self.service.evaluate(inputs)
        finally:
            self._active_references = None
        self._domain_results[(request.experiment_id, request.attempt, key)] = domain
        return domain.to_canonical()

    async def decide(
        self, result: EvaluationResult, context: EvaluationDecisionContext
    ) -> ExperimentDecision:
        del context
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
            ),
        )
        return decision.to_canonical()

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
            trust=DomainTrustAssessment(
                verdict=canonical.trust.verdict,
                stability=canonical.trust.stability,
                integrity=canonical.trust.integrity,
                flags=tuple(canonical.trust.flags),
                eta_applied=canonical.trust.eta_applied,
                seed_mean=canonical.trust.seed_mean,
                seed_stderr=canonical.trust.seed_stderr,
                seed_count=canonical.trust.seed_count,
            ),
            seed_evidence_event_ids=tuple(canonical.seed_evidence_event_ids),
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
    populations = {
        key: _load_population(path) for key, path in live.population_csvs.items()
    }
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
    receipts = ReceiptStore(root)
    data_policy = DataAccessPolicy(
        views=tuple(
            DataViewPolicy(
                view_id=command_id,
                allowed_columns=tuple(live.candidate_allowed_columns),
                allowed_path_prefixes=("/inputs/data",),
            )
            for command_id in sorted(live.input_roots)
        ),
        protected_columns=tuple(live.protected_columns),
        hidden_path_tokens=tuple(live.hidden_path_tokens),
        future_column_patterns=tuple(live.future_column_patterns),
    )
    coding_worker = TraeCodingWorker(
        worktrees=worktrees,
        artifact_repository_root=root,
        config=_trae_config_from_mapping(live.trae),
        identity_resolver=LedgerCandidateIdentityResolver(event_store),
    )
    coding_worker.preflight()
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
    execution_artifacts = CanonicalArtifactStoreAdapter(artifact_store)
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
        policy=RunnerPolicy(
            max_timeout_seconds=max(config.timeout_profiles.values()),
        ),
    )
    evaluator_adapter = create_evaluator_adapter(
        root,
        expected_evaluator_sha256=config.evaluator_sha256,
        expected_contract_sha256=verified.contract_sha256,
        expected_data_manifest_sha256=config.data_manifest_sha256,
    )
    evaluator = ProtectedEvaluationBridge(
        config=config,
        event_store=event_store,
        populations=populations,
        baseline_predictions=live.baseline_prediction_csvs,
        evaluator_adapter=evaluator_adapter,
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
        ),
        evaluator=evaluator,
        baseline=baseline,
    )


def _load_population(path: Path) -> _PopulationData:
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
    return _PopulationData(tuple(rows), tuple(labels))


def _output_contract(rows: Sequence[Mapping[str, Any]]) -> OutputContract:
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
                    "/contracts/competition",
                    "contract",
                ),
                ContainerReadOnlyMount(
                    live.input_roots[command_id].resolve(strict=True),
                    "/inputs/data",
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
        *[("input root", value) for value in live.input_roots.values()],
        *[("population CSV", value) for value in live.population_csvs.values()],
        *[("baseline prediction", value) for value in live.baseline_prediction_csvs.values()],
    ):
        if not path.exists():
            raise ContractError("%s does not exist: %s" % (label, path))
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
        ("max_token_cap", config.coding_token_limit),
        ("max_wall_time_seconds_cap", config.coding_wall_time_limit_seconds),
    ):
        if int(live.trae.get(field, 0)) < configured:
            raise ContractError("run coding limit exceeds the reviewed Trae %s" % field)


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
