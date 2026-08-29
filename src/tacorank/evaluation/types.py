"""Typed values produced by TacoRank's deterministic evaluation layer.

These dataclasses mirror the shared RankForge schema without taking ownership of
the orchestrator's Pydantic models.  Adapters can translate them with
``dataclasses.asdict`` or by reading attributes with the same names.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Mapping, Optional, Tuple


class Population(str, Enum):
    INTERNAL_PROXY = "internal_proxy"
    PUBLIC_VALIDATION = "public_validation"
    UNBIASED_AUDIT = "unbiased_audit"
    HIDDEN_FINAL = "hidden_final"


class Fidelity(str, Enum):
    SMOKE = "smoke"
    PROXY = "proxy"
    FULL = "full"
    FINAL = "final"


class Verdict(str, Enum):
    ACCEPTED = "accepted"
    INCONCLUSIVE = "inconclusive"
    NEGATIVE = "negative"
    NO_OP = "no_op"
    SUSPICIOUS = "suspicious"
    REDUNDANT = "redundant"


class Stability(str, Enum):
    SINGLE_SEED = "single_seed"
    CONFIRMED = "confirmed"
    UNSTABLE = "unstable"
    NOT_APPLICABLE = "not_applicable"


class Integrity(str, Enum):
    CLEAN = "clean"
    COMPROMISED = "compromised"
    INCONCLUSIVE = "inconclusive"


class Decision(str, Enum):
    PROMOTE = "promote"
    ACCEPT = "accept"
    REJECT = "reject"
    PRUNE = "prune"
    INVALID = "invalid"


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

