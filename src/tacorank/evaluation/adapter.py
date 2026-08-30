"""Hash-protected official evaluator adapter and evaluation service."""

from dataclasses import dataclass, field
import hashlib
import json
import math
import operator
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Optional, Sequence, Tuple

from .comparisons import compare_metric_sets
from .diagnostics import DiagnosticFeatures, compute_evaluation_diagnostics
from .metrics import normalize_binary_labels, validate_metric_set
from .no_op import analyze_prediction_change
from .trust import TrustConfig, TrustEvidence, assess_trust
from .types import (
    EvaluationResult,
    Fidelity,
    Integrity,
    MetricDelta,
    MetricSet,
    Population,
)


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
    ordered_row_identity_sha256: str

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("population rows must be positive")
        object.__setattr__(
            self,
            "ordered_row_identity_sha256",
            _validate_sha256(self.ordered_row_identity_sha256),
        )


@dataclass(frozen=True)
class PredictionBatch:
    """Gate-B-checked predictions with their immutable artifact identity."""

    artifact_id: str
    artifact_sha256: str
    row_ids: Sequence[object]
    user_ids: Sequence[object]
    item_ids: Sequence[object]
    scores: Sequence[float]

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("prediction artifact ID must not be empty")
        row_ids = tuple(self.row_ids)
        user_ids = tuple(self.user_ids)
        item_ids = tuple(self.item_ids)
        try:
            scores = tuple(float(score) for score in self.scores)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("prediction scores must be numeric")
        object.__setattr__(
            self, "artifact_sha256", _validate_sha256(self.artifact_sha256)
        )
        lengths = {
            len(row_ids),
            len(user_ids),
            len(item_ids),
            len(scores),
        }
        if len(lengths) != 1 or not scores:
            raise ValueError("prediction rows and scores must align and be non-empty")
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("prediction scores must be finite")
        _validate_contiguous_row_ids(row_ids)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "user_ids", user_ids)
        object.__setattr__(self, "item_ids", item_ids)
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class OutputGateEvidence:
    """Verified output.checked identity supplied by the orchestrator."""

    event_id: str
    accepted: bool
    prediction_artifact_id: str
    prediction_artifact_sha256: str
    population: Population
    ordered_row_identity_sha256: str
    ordered_prediction_sha256: str

    def __post_init__(self) -> None:
        if not self.event_id or not self.prediction_artifact_id:
            raise ValueError("Gate-B event and artifact IDs must not be empty")
        object.__setattr__(
            self,
            "prediction_artifact_sha256",
            _validate_sha256(self.prediction_artifact_sha256),
        )
        object.__setattr__(
            self,
            "ordered_row_identity_sha256",
            _validate_sha256(self.ordered_row_identity_sha256),
        )
        object.__setattr__(
            self,
            "ordered_prediction_sha256",
            _validate_sha256(self.ordered_prediction_sha256),
        )


@dataclass(frozen=True)
class EvaluationInputs:
    run_id: str
    experiment_id: str
    attempt: int
    output_gate: OutputGateEvidence
    predictions: PredictionBatch
    population: Population
    fidelity: Fidelity
    seed: int
    public_query_index: Optional[int]
    evaluator_sha256: str
    contract_sha256: str
    data_manifest_sha256: str
    labels: Sequence[int]
    baseline: MetricSet
    parent: MetricSet
    previous_best: MetricSet
    parent_scores: Optional[Sequence[float]] = None
    diagnostic_features: Optional[DiagnosticFeatures] = None
    baseline_scores: Optional[Sequence[float]] = None
    seed_evaluation_event_ids: Tuple[str, ...] = ()
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
        evaluator_timeout_seconds: float = 60.0,
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
        if evaluator_timeout_seconds <= 0:
            raise ValueError("evaluator timeout must be positive")
        self.evaluator_timeout_seconds = float(evaluator_timeout_seconds)

    def score(
        self,
        predictions: PredictionBatch,
        labels: Sequence[int],
        evaluator_sha256: str,
        contract_sha256: str,
        population: Population,
    ) -> MetricSet:
        self.verify_identities(evaluator_sha256, contract_sha256)
        if len(labels) != len(predictions.scores):
            raise ValueError("protected labels and prediction rows must align")
        normalized_labels = normalize_binary_labels(
            labels, "official evaluator labels"
        )
        normalized_scores = [float(score) for score in predictions.scores]
        if any(not math.isfinite(score) for score in normalized_scores):
            raise ValueError("official evaluator scores must be finite")
        self._verify_population(population, predictions)
        raw = self._run_isolated_evaluator(
            predictions.user_ids,
            normalized_labels,
            normalized_scores,
        )
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
            raise EvaluationIntegrityError(
                "request evaluator hash does not match frozen hash"
            )
        if actual_evaluator != self.expected_evaluator_sha256:
            raise EvaluationIntegrityError("protected evaluator hash mismatch")
        if requested_contract != self.expected_contract_sha256:
            raise EvaluationIntegrityError(
                "request contract hash does not match frozen hash"
            )
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
        self, population: Population, predictions: PredictionBatch
    ) -> None:
        manifest = self.population_manifests.get(population)
        if manifest is None:
            if self.expected_data_manifest_sha256 is not None:
                raise EvaluationIntegrityError(
                    "protected population manifest is missing"
                )
            return
        if len(predictions.scores) != manifest.rows:
            raise EvaluationIntegrityError(
                "population row mismatch: expected %d, observed %d"
                % (manifest.rows, len(predictions.scores))
            )
        observed = ordered_row_identity_sha256(
            predictions.row_ids,
            predictions.user_ids,
            predictions.item_ids,
        )
        if observed != manifest.ordered_row_identity_sha256:
            raise EvaluationIntegrityError("population row alignment mismatch")

    def _run_isolated_evaluator(
        self,
        user_ids: Sequence[object],
        labels: Sequence[int],
        scores: Sequence[float],
    ) -> Mapping[str, object]:
        worker_path = Path(__file__).with_name("_isolated_worker.py")
        payload = json.dumps(
            {
                "user_ids": [str(user_id) for user_id in user_ids],
                "labels": list(labels),
                "scores": list(scores),
            },
            allow_nan=False,
            separators=(",", ":"),
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-I",
                    str(worker_path),
                    str(self.evaluator_path.resolve()),
                    self.expected_evaluator_sha256,
                ],
                input=payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.evaluator_timeout_seconds,
                cwd=str(worker_path.parent),
                env={"PATH": "/usr/bin:/bin"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EvaluationIntegrityError("isolated evaluator failed to run") from exc
        if completed.returncode != 0:
            raise EvaluationIntegrityError(
                "isolated evaluator rejected the request: %s"
                % completed.stderr.strip()[:500]
            )
        try:
            raw = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise EvaluationIntegrityError(
                "isolated evaluator returned invalid JSON"
            ) from exc
        if not isinstance(raw, Mapping):
            raise EvaluationIntegrityError("isolated evaluator returned a non-mapping")
        return raw


class EvaluationService:
    """Compose official scoring, comparisons, change analysis, and trust."""

    def __init__(
        self,
        adapter: ProtectedEvaluatorAdapter,
        trust_config: Optional[TrustConfig] = None,
        output_gate_resolver: Optional[
            Callable[[str], OutputGateEvidence]
        ] = None,
        seed_result_resolver: Optional[Callable[[str], EvaluationResult]] = None,
    ) -> None:
        self.adapter = adapter
        self.trust_config = trust_config or TrustConfig()
        self.output_gate_resolver = output_gate_resolver
        self.seed_result_resolver = seed_result_resolver

    def evaluate(self, request: EvaluationInputs) -> EvaluationResult:
        if self.output_gate_resolver is None:
            raise EvaluationIntegrityError("verified Gate-B resolver is required")
        try:
            gate = self.output_gate_resolver(request.output_gate.event_id)
        except Exception as exc:
            raise EvaluationIntegrityError(
                "Gate-B evidence could not be resolved"
            ) from exc
        if not isinstance(gate, OutputGateEvidence) or gate != request.output_gate:
            raise EvaluationIntegrityError(
                "Gate-B evidence does not match verified event"
            )
        predictions = request.predictions
        if not gate.accepted:
            raise EvaluationIntegrityError("verified Gate-B evidence is required")
        if gate.population != request.population:
            raise EvaluationIntegrityError("Gate-B population does not match request")
        if (
            gate.prediction_artifact_id != predictions.artifact_id
            or gate.prediction_artifact_sha256 != predictions.artifact_sha256
        ):
            raise EvaluationIntegrityError(
                "Gate-B evidence does not match prediction artifact"
            )
        row_identity = ordered_row_identity_sha256(
            predictions.row_ids,
            predictions.user_ids,
            predictions.item_ids,
        )
        if row_identity != gate.ordered_row_identity_sha256:
            raise EvaluationIntegrityError(
                "Gate-B evidence does not match prediction row identity"
            )
        prediction_identity = ordered_prediction_sha256(
            predictions.row_ids,
            predictions.user_ids,
            predictions.item_ids,
            predictions.scores,
        )
        if prediction_identity != gate.ordered_prediction_sha256:
            raise EvaluationIntegrityError(
                "Gate-B evidence does not match checked prediction values"
            )
        self._validate_route(request)
        self.adapter.verify_data_manifest(request.data_manifest_sha256)
        metric_set = self.adapter.score(
            predictions,
            request.labels,
            request.evaluator_sha256,
            request.contract_sha256,
            request.population,
        )
        baseline_delta = compare_metric_sets(metric_set, request.baseline)
        parent_delta = compare_metric_sets(metric_set, request.parent)
        best_delta = compare_metric_sets(metric_set, request.previous_best)
        change = analyze_prediction_change(
            predictions.scores,
            request.parent_scores,
            self.trust_config.no_op.score_tolerance,
        )
        diagnostic_summary = compute_evaluation_diagnostics(
            user_ids=predictions.user_ids,
            labels=request.labels,
            candidate_scores=predictions.scores,
            parent_scores=(
                request.parent_scores
                if request.parent_scores is not None
                else predictions.scores
            ),
            parent_delta=parent_delta,
            features=request.diagnostic_features,
            proxy_parent_delta=request.internal_proxy_delta,
            validation_gap_threshold=0.006,
            temporal_slope_threshold=self.trust_config.drift_slope_threshold,
            concentration_threshold=self.trust_config.gain_concentration_threshold,
        )
        prior_seed_results = self._resolve_seed_results(request)
        seed_metric_sets = tuple(
            result.metric_set for result in prior_seed_results
        ) + (metric_set,)
        aggregate_metric_set = _aggregate_metric_sets(seed_metric_sets)
        aggregate_parent_delta = compare_metric_sets(
            aggregate_metric_set, request.parent
        )
        seed_scores = tuple(
            result.metric_set.primary_score for result in prior_seed_results
        ) + (metric_set.primary_score,)
        trust = assess_trust(
            TrustEvidence(
                population=request.population,
                fidelity=request.fidelity,
                parent_primary=request.parent.primary_score,
                parent_delta=aggregate_parent_delta.primary,
                metric_deltas=aggregate_parent_delta.metrics,
                prediction_change=change,
                seed_scores=seed_scores,
                output_gate_evidence=True,
                evaluator_hash_matches=True,
                contract_hash_matches=True,
                forbidden_inputs=request.forbidden_inputs,
                alignment_suspect=request.alignment_suspect,
                internal_proxy_delta=request.internal_proxy_delta,
                unbiased_audit_delta=request.unbiased_audit_delta,
                val_a_delta=diagnostic_summary.validation_arm_deltas.get("val_a"),
                val_b_delta=(
                    request.val_b_delta
                    if request.val_b_delta is not None
                    else diagnostic_summary.validation_arm_deltas.get("val_b")
                ),
                delta_correlation=request.delta_correlation,
                delta_correlation_experiment_id=(
                    request.delta_correlation_experiment_id
                ),
                score_unique_fraction=change.unique_score_fraction,
                gain_concentration_top10pct=(
                    request.gain_concentration_top10pct
                    if request.gain_concentration_top10pct is not None
                    else diagnostic_summary.gain_concentration_top10pct
                ),
                drift_primary_slope=(
                    request.drift_primary_slope
                    if request.drift_primary_slope is not None
                    else diagnostic_summary.temporal_delta_slope
                ),
            ),
            self.trust_config,
        )
        diagnostic_metrics = dict(
            _prediction_quality_diagnostics(
                predictions,
                parent_scores=request.parent_scores,
                baseline_scores=request.baseline_scores,
                tolerance=self.trust_config.no_op.score_tolerance,
            )
        )
        diagnostic_metrics.update(
            {
                **(
                    {
                        "proxy_full_delta_gap": (
                            diagnostic_summary.proxy_full_delta_gap
                        ),
                    }
                    if diagnostic_summary.proxy_full_delta_gap is not None
                    else {}
                ),
                **(
                    {"validation_arm_gap": diagnostic_summary.validation_arm_gap}
                    if diagnostic_summary.validation_arm_gap is not None
                    else {}
                ),
                **(
                    {
                        "temporal_delta_slope": (
                            diagnostic_summary.temporal_delta_slope
                        )
                    }
                    if diagnostic_summary.temporal_delta_slope is not None
                    else {}
                ),
                **(
                    {
                        "gain_concentration_top10pct": (
                            diagnostic_summary.gain_concentration_top10pct
                        )
                    }
                    if diagnostic_summary.gain_concentration_top10pct is not None
                    else {}
                ),
            }
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
            data_manifest_sha256=request.data_manifest_sha256,
            metric_set=metric_set,
            baseline_delta=baseline_delta,
            parent_delta=parent_delta,
            previous_best_delta=best_delta,
            prediction_change=change,
            trust=trust,
            diagnostic_metrics=diagnostic_metrics,
            diagnostics=diagnostic_summary,
            seed_evidence_event_ids=request.seed_evaluation_event_ids,
        )

    def _resolve_seed_results(
        self, request: EvaluationInputs
    ) -> Tuple[EvaluationResult, ...]:
        event_ids = request.seed_evaluation_event_ids
        if len(set(event_ids)) != len(event_ids):
            raise EvaluationIntegrityError("seed evidence event IDs must be unique")
        if not event_ids:
            return ()
        if (
            request.population != Population.PUBLIC_VALIDATION
            or request.fidelity != Fidelity.FULL
        ):
            raise EvaluationIntegrityError(
                "seed confirmation is limited to full public validation"
            )
        if self.seed_result_resolver is None:
            raise EvaluationIntegrityError("seed evidence resolver is required")

        try:
            results = tuple(
                self.seed_result_resolver(event_id) for event_id in event_ids
            )
        except Exception as exc:
            raise EvaluationIntegrityError(
                "seed evidence could not be resolved"
            ) from exc
        seeds = {request.seed}
        previous_attempt = 0
        for event_id, result in zip(event_ids, results):
            if not isinstance(result, EvaluationResult):
                raise EvaluationIntegrityError(
                    "seed evidence %s did not resolve to an evaluation" % event_id
                )
            if (
                result.run_id != request.run_id
                or result.experiment_id != request.experiment_id
                or result.population != request.population
                or result.fidelity != request.fidelity
                or result.evaluator_sha256 != request.evaluator_sha256
                or result.contract_sha256 != request.contract_sha256
                or result.data_manifest_sha256 != request.data_manifest_sha256
            ):
                raise EvaluationIntegrityError(
                    "seed evidence %s has incompatible identity" % event_id
                )
            if result.attempt >= request.attempt:
                raise EvaluationIntegrityError(
                    "seed evidence %s is not from an earlier attempt" % event_id
                )
            if result.attempt <= previous_attempt:
                raise EvaluationIntegrityError(
                    "seed evidence must be ordered by increasing attempt"
                )
            previous_attempt = result.attempt
            if result.trust.integrity != Integrity.CLEAN:
                raise EvaluationIntegrityError(
                    "seed evidence %s does not have clean integrity" % event_id
                )
            if result.seed in seeds:
                raise EvaluationIntegrityError(
                    "seed evidence contains a duplicate seed"
                )
            seeds.add(result.seed)
            _verify_reference_metric_set(
                result.metric_set,
                result.parent_delta,
                request.parent.primary_score,
                request.parent.metrics,
                "parent",
            )
            _verify_reference_metric_set(
                result.metric_set,
                result.baseline_delta,
                request.baseline.primary_score,
                request.baseline.metrics,
                "baseline",
            )
            _verify_reference_metric_set(
                result.metric_set,
                result.previous_best_delta,
                request.previous_best.primary_score,
                request.previous_best.metrics,
                "previous best",
            )
        return results

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
        if (
            request.population == Population.INTERNAL_PROXY
            and request.fidelity != Fidelity.PROXY
        ):
            raise ValueError("internal proxy population requires proxy fidelity")
        if (
            request.population == Population.UNBIASED_AUDIT
            and request.fidelity != Fidelity.FULL
        ):
            raise ValueError("unbiased audit population requires full fidelity")
        if request.population == Population.HIDDEN_FINAL:
            if request.fidelity != Fidelity.FINAL:
                raise ValueError("hidden final population requires final fidelity")
            if not request.run_stopped_event_id:
                raise EvaluationIntegrityError(
                    "hidden final requires verified run.stopped evidence"
                )


def _prediction_quality_diagnostics(
    predictions: PredictionBatch,
    *,
    parent_scores: Optional[Sequence[float]],
    baseline_scores: Optional[Sequence[float]],
    tolerance: float,
) -> Mapping[str, float]:
    """Return label-free diagnostics that make weak candidates actionable."""

    scores = tuple(float(value) for value in predictions.scores)
    diagnostics = {
        "score_unique_fraction": len(set(scores)) / len(scores),
        "score_std": _population_std(scores),
    }
    user_fraction, user_count = _group_variation_fraction(
        predictions.user_ids, scores, tolerance=tolerance
    )
    item_fraction, item_count = _group_variation_fraction(
        predictions.item_ids,
        scores,
        tolerance=tolerance,
        distinct_members=predictions.user_ids,
    )
    diagnostics.update(
        {
            "user_rankable_fraction": user_fraction,
            "multirow_user_count": float(user_count),
            "item_personalized_fraction": item_fraction,
            "multiuser_item_count": float(item_count),
        }
    )
    if parent_scores is not None:
        parent = tuple(float(value) for value in parent_scores)
        if len(parent) != len(scores):
            raise ValueError("candidate and parent diagnostics must align")
        residuals = tuple(left - right for left, right in zip(scores, parent))
        diagnostics.update(
            {
                "mean_abs_parent_residual": sum(map(abs, residuals)) / len(residuals),
                "parent_residual_std": _population_std(residuals),
            }
        )
    if baseline_scores is not None:
        baseline_change = analyze_prediction_change(
            scores, baseline_scores, tolerance
        )
        correlation = baseline_change.spearman_vs_parent
        diagnostics["spearman_vs_fm_baseline"] = (
            float(correlation) if correlation is not None else 0.0
        )
    return diagnostics


def _population_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _group_variation_fraction(
    keys: Sequence[object],
    scores: Sequence[float],
    *,
    tolerance: float,
    distinct_members: Optional[Sequence[object]] = None,
) -> Tuple[float, int]:
    groups: dict[str, list[Tuple[float, Optional[str]]]] = {}
    members = (
        distinct_members
        if distinct_members is not None
        else (None,) * len(scores)
    )
    if not (len(keys) == len(scores) == len(members)):
        raise ValueError("diagnostic grouping inputs must align")
    for key, score, member in zip(keys, scores, members):
        groups.setdefault(str(key), []).append(
            (float(score), None if member is None else str(member))
        )
    eligible = []
    for values in groups.values():
        if len(values) < 2:
            continue
        if distinct_members is not None and len({member for _, member in values}) < 2:
            continue
        eligible.append(values)
    if not eligible:
        return 0.0, 0
    varied = sum(
        max(score for score, _ in values) - min(score for score, _ in values)
        > tolerance
        for values in eligible
    )
    return varied / len(eligible), len(eligible)


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


def ordered_row_identity_sha256(
    row_ids: Sequence[object],
    user_ids: Sequence[object],
    item_ids: Sequence[object],
) -> str:
    if not (len(row_ids) == len(user_ids) == len(item_ids)) or not row_ids:
        raise ValueError("row identities must align and be non-empty")
    _validate_contiguous_row_ids(row_ids)
    digest = hashlib.sha256()
    for row_id, user_id, item_id in zip(row_ids, user_ids, item_ids):
        record = json.dumps(
            [int(row_id), str(user_id), str(item_id)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    return digest.hexdigest()


def ordered_prediction_sha256(
    row_ids: Sequence[object],
    user_ids: Sequence[object],
    item_ids: Sequence[object],
    scores: Sequence[float],
) -> str:
    if not (
        len(row_ids) == len(user_ids) == len(item_ids) == len(scores)
    ) or not scores:
        raise ValueError("prediction identities must align and be non-empty")
    _validate_contiguous_row_ids(row_ids)
    digest = hashlib.sha256()
    for row_id, user_id, item_id, score in zip(
        row_ids, user_ids, item_ids, scores
    ):
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("prediction scores must be finite")
        record = json.dumps(
            [int(row_id), str(user_id), str(item_id), numeric_score.hex()],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    return digest.hexdigest()


def _validate_contiguous_row_ids(row_ids: Sequence[object]) -> None:
    for expected, value in enumerate(row_ids):
        if isinstance(value, bool):
            raise ValueError("row IDs must be integers")
        try:
            observed = operator.index(value)
        except TypeError:
            raise ValueError("row IDs must be integers")
        if observed != expected:
            raise ValueError("row IDs must be zero-based, contiguous, and ordered")


def _aggregate_metric_sets(metric_sets: Sequence[MetricSet]) -> MetricSet:
    if not metric_sets:
        raise ValueError("at least one metric set is required")
    first = metric_sets[0]
    names = tuple(first.metrics)
    for metric_set in metric_sets[1:]:
        if (
            metric_set.primary_metric_name != first.primary_metric_name
            or set(metric_set.metrics) != set(names)
        ):
            raise EvaluationIntegrityError("seed metric schemas do not match")
    count = len(metric_sets)
    metrics = {
        name: sum(metric_set.metrics[name] for metric_set in metric_sets) / count
        for name in names
    }
    primary = sum(metric_set.primary_score for metric_set in metric_sets) / count
    return MetricSet(metrics, first.primary_metric_name, primary)


def _verify_reference_metric_set(
    metric_set: MetricSet,
    delta: MetricDelta,
    expected_primary: float,
    expected_metrics: Mapping[str, float],
    name: str,
    tolerance: float = 1e-12,
) -> None:
    if set(metric_set.metrics) != set(delta.metrics) or set(delta.metrics) != set(
        expected_metrics
    ):
        raise EvaluationIntegrityError(
            "seed evidence uses a different %s metric schema" % name
        )
    observed_primary = float(metric_set.primary_score) - float(delta.primary)
    if abs(observed_primary - float(expected_primary)) > tolerance:
        raise EvaluationIntegrityError(
            "seed evidence uses a different %s reference" % name
        )
    for metric_name, expected_value in expected_metrics.items():
        observed_value = (
            float(metric_set.metrics[metric_name]) - float(delta.metrics[metric_name])
        )
        if abs(observed_value - float(expected_value)) > tolerance:
            raise EvaluationIntegrityError(
                "seed evidence uses a different %s reference" % name
            )


def _validate_sha256(value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        c not in "0123456789abcdef" for c in normalized
    ):
        raise ValueError("expected a lowercase SHA-256 digest")
    return normalized
