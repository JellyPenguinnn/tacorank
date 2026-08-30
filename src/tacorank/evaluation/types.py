"""Typed values produced by TacoRank's deterministic evaluation layer.

These domain dataclasses add evaluation-only evidence around TacoRank's shared
enums. Canonical ledger payloads are built with ``tacorank.schemas`` models.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Mapping, Optional, Tuple

from tacorank.schemas import (
    Decision,
    EvaluationResult as CanonicalEvaluationResult,
    ExperimentDecision as CanonicalExperimentDecision,
    Fidelity,
    Integrity,
    MetricSet as CanonicalMetricSet,
    Population,
    PredictionChange as CanonicalPredictionChange,
    Stability,
    TrustAssessment as CanonicalTrustAssessment,
    TrustVerdict as Verdict,
)


@dataclass(frozen=True)
class MetricSet:
    metrics: Mapping[str, float]
    primary_metric_name: str
    primary_score: float

    def __post_init__(self) -> None:
        if not self.primary_metric_name:
            raise ValueError("primary_metric_name must not be empty")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        normalized: Dict[str, float] = {}
        for name, value in self.metrics.items():
            if not name:
                raise ValueError("metric names must not be empty")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("metric %r must be finite" % name)
            normalized[str(name)] = numeric
        primary = float(self.primary_score)
        if not math.isfinite(primary):
            raise ValueError("primary_score must be finite")
        object.__setattr__(self, "metrics", normalized)
        object.__setattr__(self, "primary_score", primary)

    def to_canonical(self) -> CanonicalMetricSet:
        return CanonicalMetricSet(
            metrics=dict(self.metrics),
            primary_metric_name=self.primary_metric_name,
            primary_score=self.primary_score,
        )


@dataclass(frozen=True)
class MetricDelta:
    primary: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class PredictionChange:
    spearman_vs_parent: Optional[float]
    changed_row_fraction: Optional[float]
    identical_score_fraction: Optional[float]
    unique_score_fraction: float


@dataclass(frozen=True)
class SeedAggregate:
    scores: Tuple[float, ...]
    mean: float
    standard_deviation: float
    standard_error: float
    eta: float

    @property
    def count(self) -> int:
        return len(self.scores)


@dataclass(frozen=True)
class TrustAssessment:
    verdict: Verdict
    stability: Stability
    integrity: Integrity
    flags: Tuple[str, ...] = ()
    eta_applied: Optional[float] = None
    seed_mean: Optional[float] = None
    seed_stderr: Optional[float] = None
    seed_count: int = 1


@dataclass(frozen=True)
class EvaluationResult:
    run_id: str
    experiment_id: str
    attempt: int
    population: Population
    fidelity: Fidelity
    seed: int
    public_query_index: Optional[int]
    evaluator_sha256: str
    contract_sha256: str
    data_manifest_sha256: str
    metric_set: MetricSet
    baseline_delta: MetricDelta
    parent_delta: MetricDelta
    previous_best_delta: MetricDelta
    prediction_change: PredictionChange
    trust: TrustAssessment
    diagnostic_metrics: Mapping[str, float] = field(default_factory=dict)
    seed_evidence_event_ids: Tuple[str, ...] = ()

    def to_canonical(self) -> CanonicalEvaluationResult:
        if (
            self.prediction_change.spearman_vs_parent is None
            or self.prediction_change.changed_row_fraction is None
        ):
            raise ValueError("canonical evaluation requires parent prediction evidence")
        return CanonicalEvaluationResult(
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            attempt=self.attempt,
            population=self.population,
            fidelity=self.fidelity,
            seed=self.seed,
            public_query_index=self.public_query_index,
            evaluator_sha256=self.evaluator_sha256,
            contract_sha256=self.contract_sha256,
            metric_set=self.metric_set.to_canonical(),
            baseline_delta=self.baseline_delta.primary,
            parent_delta=self.parent_delta.primary,
            previous_best_delta=self.previous_best_delta.primary,
            prediction_change=CanonicalPredictionChange(
                spearman_vs_parent=self.prediction_change.spearman_vs_parent,
                changed_row_fraction=self.prediction_change.changed_row_fraction,
            ),
            diagnostic_metrics=dict(self.diagnostic_metrics),
            trust=CanonicalTrustAssessment(
                verdict=self.trust.verdict,
                stability=self.trust.stability,
                integrity=self.trust.integrity,
                flags=list(self.trust.flags),
                eta_applied=self.trust.eta_applied,
                seed_mean=self.trust.seed_mean,
                seed_stderr=self.trust.seed_stderr,
                seed_count=self.trust.seed_count,
            ),
            seed_evidence_event_ids=list(self.seed_evidence_event_ids),
        )


@dataclass(frozen=True)
class ExperimentDecision:
    run_id: str
    experiment_id: str
    evaluation_event_id: Optional[str]
    decision: Decision
    reason_code: str
    fidelity_completed: Fidelity
    parent_eligible: bool
    best_eligible: bool
    next_fidelity: Optional[Fidelity]
    supporting_event_ids: Tuple[str, ...]
    seed_evidence_event_ids: Tuple[str, ...] = ()

    def to_canonical(self) -> CanonicalExperimentDecision:
        supporting = tuple(
            dict.fromkeys(self.supporting_event_ids + self.seed_evidence_event_ids)
        )
        return CanonicalExperimentDecision(
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            evaluation_event_id=self.evaluation_event_id,
            decision=self.decision,
            reason_code=self.reason_code,
            fidelity_completed=self.fidelity_completed,
            parent_eligible=self.parent_eligible,
            best_eligible=self.best_eligible,
            next_fidelity=self.next_fidelity,
            supporting_event_ids=list(supporting),
            lesson_candidate=None,
        )
