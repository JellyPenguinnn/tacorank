"""Hash-protected official evaluator adapter and evaluation service."""

from dataclasses import dataclass, field
import hashlib
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Mapping, Optional, Sequence, Tuple

from .comparisons import compare_metric_sets
from .metrics import validate_metric_set
from .no_op import analyze_prediction_change
from .trust import TrustConfig, TrustEvidence, assess_trust
from .types import EvaluationResult, Fidelity, MetricSet, Population


class EvaluationIntegrityError(RuntimeError):
    """Raised when protected identities or Gate-B evidence do not match."""


@dataclass(frozen=True)
class ContractSpec:
    required_metrics: Tuple[str, ...]
    primary_metric_name: str
    primary_weights: Mapping[str, float]
    metric_ranges: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    diagnostic_metrics: Tuple[str, ...] = ()
    aggregation_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.required_metrics or len(set(self.required_metrics)) != len(
            self.required_metrics
        ):
            raise ValueError("required metrics must be unique and non-empty")
        if set(self.primary_weights) != set(self.required_metrics):
            raise ValueError("primary weights must match required metrics")


@dataclass(frozen=True)
class PopulationManifest:
    rows: int
    ordered_user_ids_sha256: Optional[str] = None


@dataclass(frozen=True)
class EvaluationInputs:
    run_id: str
    experiment_id: str
    attempt: int
    output_checked_event_id: str
    output_gate_accepted: bool
    population: Population
    fidelity: Fidelity
    seed: int
    public_query_index: Optional[int]
    evaluator_sha256: str
    contract_sha256: str
    data_manifest_sha256: str
    user_ids: Sequence[object]
    labels: Sequence[int]
    scores: Sequence[float]
    baseline: MetricSet
    parent: MetricSet
    previous_best: MetricSet
    parent_scores: Optional[Sequence[float]] = None
    seed_primary_scores: Sequence[float] = ()
    internal_proxy_delta: Optional[float] = None
    unbiased_audit_delta: Optional[float] = None
    val_b_delta: Optional[float] = None
    forbidden_inputs: Tuple[str, ...] = ()
    alignment_suspect: bool = False
    delta_correlation: Optional[float] = None
    delta_correlation_experiment_id: Optional[str] = None
    gain_concentration_top10pct: Optional[float] = None
    drift_primary_slope: Optional[float] = None
    run_stopped_event_id: Optional[str] = None


class ProtectedEvaluatorAdapter:
    """Call an official ``evaluate`` function after checking frozen hashes."""

    def __init__(
        self,
        evaluator_path: Path,
        expected_evaluator_sha256: str,
        expected_contract_sha256: str,
        contract: ContractSpec,
        contract_path: Optional[Path] = None,
        population_manifests: Optional[Mapping[Population, PopulationManifest]] = None,
        expected_data_manifest_sha256: Optional[str] = None,
    ) -> None:
        self.evaluator_path = Path(evaluator_path)
        self.expected_evaluator_sha256 = _validate_sha256(expected_evaluator_sha256)
        self.expected_contract_sha256 = _validate_sha256(expected_contract_sha256)
        self.contract = contract
        self.contract_path = Path(contract_path) if contract_path else None
        self.population_manifests = dict(population_manifests or {})
        self.expected_data_manifest_sha256 = (
            _validate_sha256(expected_data_manifest_sha256)
            if expected_data_manifest_sha256 is not None
            else None
        )

    def score(
        self,
        user_ids: Sequence[object],
        labels: Sequence[int],
        scores: Sequence[float],
        evaluator_sha256: str,
        contract_sha256: str,
        population: Population,
    ) -> MetricSet:
        self.verify_identities(evaluator_sha256, contract_sha256)
        if not (len(user_ids) == len(labels) == len(scores)) or not scores:
            raise ValueError("user IDs, labels and scores must be equal and non-empty")
        normalized_labels = [int(label) for label in labels]
        if any(label not in (0, 1) for label in normalized_labels):
            raise ValueError("official evaluator labels must be binary")
        normalized_scores = [float(score) for score in scores]
        if any(not math.isfinite(score) for score in normalized_scores):
            raise ValueError("official evaluator scores must be finite")
        self._verify_population(population, user_ids)
        module = self._load_pristine_module()
        evaluate = getattr(module, "evaluate", None)
        if not callable(evaluate):
            raise EvaluationIntegrityError("official evaluator has no evaluate function")
        raw = evaluate(user_ids, normalized_labels, normalized_scores)
        if not isinstance(raw, Mapping):
            raise EvaluationIntegrityError("official evaluator returned a non-mapping")
        return validate_metric_set(
            raw,
            self.contract.required_metrics,
            self.contract.primary_metric_name,
            self.contract.primary_weights,
            tolerance=self.contract.aggregation_tolerance,
            allow_extra=("users", "rows") + self.contract.diagnostic_metrics,
            metric_ranges=self.contract.metric_ranges,
        )

    def verify_identities(self, evaluator_sha256: str, contract_sha256: str) -> None:
        requested_evaluator = _validate_sha256(evaluator_sha256)
        requested_contract = _validate_sha256(contract_sha256)
        actual_evaluator = sha256_file(self.evaluator_path)
        if requested_evaluator != self.expected_evaluator_sha256:
            raise EvaluationIntegrityError("request evaluator hash does not match frozen hash")
        if actual_evaluator != self.expected_evaluator_sha256:
            raise EvaluationIntegrityError("protected evaluator hash mismatch")
        if requested_contract != self.expected_contract_sha256:
            raise EvaluationIntegrityError("request contract hash does not match frozen hash")
        if self.contract_path is not None:
            actual_contract = sha256_file(self.contract_path)
            if actual_contract != self.expected_contract_sha256:
                raise EvaluationIntegrityError("protected contract hash mismatch")

    def verify_data_manifest(self, data_manifest_sha256: str) -> None:
        requested = _validate_sha256(data_manifest_sha256)
        if (
            self.expected_data_manifest_sha256 is not None
            and requested != self.expected_data_manifest_sha256
        ):
            raise EvaluationIntegrityError("data manifest hash mismatch")

    def _verify_population(
        self, population: Population, user_ids: Sequence[object]
    ) -> None:
        manifest = self.population_manifests.get(population)
        if manifest is None:
            return
        if len(user_ids) != manifest.rows:
            raise EvaluationIntegrityError(
                "population row mismatch: expected %d, observed %d"
                % (manifest.rows, len(user_ids))
            )
        if manifest.ordered_user_ids_sha256 is not None:
            observed = ordered_values_sha256(user_ids)
            if observed != manifest.ordered_user_ids_sha256:
                raise EvaluationIntegrityError("population user alignment mismatch")

    def _load_pristine_module(self) -> ModuleType:
        # A fresh module prevents candidate state from persisting between calls.
        module_name = "_tacorank_official_evaluator_%s" % self.expected_evaluator_sha256[:12]
        spec = importlib.util.spec_from_file_location(module_name, str(self.evaluator_path))
        if spec is None or spec.loader is None:
            raise EvaluationIntegrityError("cannot load protected evaluator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class EvaluationService:
    """Compose official scoring, comparisons, change analysis, and trust."""

    def __init__(
        self,
        adapter: ProtectedEvaluatorAdapter,
        trust_config: Optional[TrustConfig] = None,
    ) -> None:
        self.adapter = adapter
        self.trust_config = trust_config or TrustConfig()

    def evaluate(self, request: EvaluationInputs) -> EvaluationResult:
        if not request.output_gate_accepted or not request.output_checked_event_id:
            raise EvaluationIntegrityError("verified Gate-B evidence is required")
        self._validate_route(request)
        self.adapter.verify_data_manifest(request.data_manifest_sha256)
        metric_set = self.adapter.score(
            request.user_ids,
            request.labels,
            request.scores,
            request.evaluator_sha256,
            request.contract_sha256,
            request.population,
        )
        baseline_delta = compare_metric_sets(metric_set, request.baseline)
        parent_delta = compare_metric_sets(metric_set, request.parent)
        best_delta = compare_metric_sets(metric_set, request.previous_best)
        change = analyze_prediction_change(
            request.scores,
            request.parent_scores,
            self.trust_config.no_op.score_tolerance,
        )
        seed_scores = tuple(request.seed_primary_scores) or (metric_set.primary_score,)
        trust = assess_trust(
            TrustEvidence(
                population=request.population,
                fidelity=request.fidelity,
                parent_primary=request.parent.primary_score,
                parent_delta=parent_delta.primary,
                metric_deltas=parent_delta.metrics,
                prediction_change=change,
                seed_scores=seed_scores,
                output_gate_evidence=True,
                evaluator_hash_matches=True,
                contract_hash_matches=True,
                forbidden_inputs=request.forbidden_inputs,
                alignment_suspect=request.alignment_suspect,
                internal_proxy_delta=request.internal_proxy_delta,
                unbiased_audit_delta=request.unbiased_audit_delta,
                val_b_delta=request.val_b_delta,
                delta_correlation=request.delta_correlation,
                delta_correlation_experiment_id=(
                    request.delta_correlation_experiment_id
                ),
                score_unique_fraction=change.unique_score_fraction,
                gain_concentration_top10pct=(
                    request.gain_concentration_top10pct
                ),
                drift_primary_slope=request.drift_primary_slope,
            ),
            self.trust_config,
        )
        diagnostics = {
            "score_unique_fraction": change.unique_score_fraction,
        }
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
            data_manifest_sha256=request.data_manifest_sha256,
            metric_set=metric_set,
            baseline_delta=baseline_delta,
            parent_delta=parent_delta,
            previous_best_delta=best_delta,
            prediction_change=change,
            trust=trust,
            diagnostic_metrics=diagnostics,
        )

    @staticmethod
    def _validate_route(request: EvaluationInputs) -> None:
        if request.fidelity == Fidelity.SMOKE:
            raise ValueError("smoke outputs do not use official evaluation")
        if request.population == Population.PUBLIC_VALIDATION:
            if request.fidelity != Fidelity.FULL:
                raise ValueError("public validation requires full fidelity")
            if request.public_query_index is None or request.public_query_index < 1:
                raise ValueError("public validation requires a positive query index")
        elif request.public_query_index is not None:
            raise ValueError("only public validation may carry a query index")
        if request.population == Population.INTERNAL_PROXY and request.fidelity != Fidelity.PROXY:
            raise ValueError("internal proxy population requires proxy fidelity")
        if request.population == Population.UNBIASED_AUDIT and request.fidelity != Fidelity.FULL:
            raise ValueError("unbiased audit population requires full fidelity")
        if request.population == Population.HIDDEN_FINAL:
            if request.fidelity != Fidelity.FINAL:
                raise ValueError("hidden final population requires final fidelity")
            if not request.run_stopped_event_id:
                raise EvaluationIntegrityError("hidden final requires verified run.stopped evidence")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_values_sha256(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_sha256(value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("expected a lowercase SHA-256 digest")
    return normalized
