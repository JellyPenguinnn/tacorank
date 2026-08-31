"""Seed aggregation, run-local evidence projection, and leaderboard gating."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence, Tuple

from ..schemas import EventType, Integrity, MetricSet
from .types import SeedAggregate


def aggregate_seeds(scores: Sequence[float], eta_floor: float = 0.0016) -> SeedAggregate:
    values = tuple(float(score) for score in scores)
    if not values or any(not math.isfinite(score) for score in values):
        raise ValueError("seed scores must be a non-empty finite sequence")
    mean = statistics.mean(values)
    if len(values) == 1:
        standard_deviation = 0.0
        standard_error = 0.0
    else:
        standard_deviation = statistics.stdev(values)
        standard_error = standard_deviation / math.sqrt(len(values))
    return SeedAggregate(
        values,
        mean,
        standard_deviation,
        standard_error,
        max(2.0 * standard_error, float(eta_floor)),
    )


@dataclass(frozen=True)
class LadderResult:
    accepted: bool
    reported_score: float
    candidate_mean: float
    standard_error: float
    eta: float
    delta_vs_best: Optional[float]
    query_index: int


class Ladder:
    """Stateful Ladder-style release gate for higher-is-better metrics.

    The significance threshold is the repository's frozen pragmatic rule.  The
    optional quantization follows the original Ladder information-limiting idea.
    """

    def __init__(
        self,
        validation_rows: int,
        eta_floor: float = 0.0016,
        quantize: bool = True,
    ) -> None:
        if validation_rows <= 0:
            raise ValueError("validation_rows must be positive")
        self._precision = 1.0 / math.sqrt(validation_rows)
        self._eta_floor = float(eta_floor)
        self._quantize = bool(quantize)
        self._best_true: Optional[float] = None
        self._best_reported: Optional[float] = None
        self._queries = 0

    @property
    def best_true(self) -> Optional[float]:
        return self._best_true

    @property
    def best_reported(self) -> Optional[float]:
        return self._best_reported

    @property
    def queries(self) -> int:
        return self._queries

    def submit(self, seed_scores: Sequence[float]) -> LadderResult:
        aggregate = aggregate_seeds(seed_scores, self._eta_floor)
        self._queries += 1
        previous = self._best_true
        delta = None if previous is None else aggregate.mean - previous
        accepted = previous is None or aggregate.mean > previous + aggregate.eta
        if accepted:
            self._best_true = aggregate.mean
            if self._quantize:
                self._best_reported = (
                    round(aggregate.mean / self._precision) * self._precision
                )
            else:
                self._best_reported = aggregate.mean
        assert self._best_reported is not None
        return LadderResult(
            accepted=accepted,
            reported_score=self._best_reported,
            candidate_mean=aggregate.mean,
            standard_error=aggregate.standard_error,
            eta=aggregate.eta,
            delta_vs_best=delta,
            query_index=self._queries,
        )


def seed_independence_passes(
    observed_scores: Sequence[float],
    expected_std: float,
    tolerance_factor: float = 3.0,
    minimum_std: float = 0.0002,
) -> bool:
    aggregate = aggregate_seeds(observed_scores)
    lower = max(float(minimum_std), float(expected_std) / tolerance_factor)
    upper = float(expected_std) * tolerance_factor
    return lower <= aggregate.standard_deviation <= upper


def confirmed_seed_evaluation_events(
    events: Sequence[Any], terminal_event: Any
) -> Tuple[Any, ...]:
    """Resolve the current-run seed evidence for one terminal evaluation.

    The evaluation record names the earlier seed events explicitly.  We use
    only those IDs plus the terminal event, and require the declared count to
    match before treating the evidence as an aggregate.  Missing or malformed
    references fall back to the terminal observation instead of silently
    mixing unrelated evaluations.
    """

    result = getattr(getattr(terminal_event, "payload", None), "result", None)
    trust = getattr(result, "trust", None)
    seed_mean = getattr(trust, "seed_mean", None)
    seed_count = getattr(trust, "seed_count", 1)
    if seed_mean is None or seed_count < 3:
        return ()

    evidence_ids = list(getattr(result, "seed_evidence_event_ids", ()) or ())
    if len(evidence_ids) != seed_count - 1:
        return ()
    by_id = {
        getattr(event, "event_id", None): event
        for event in events
        if getattr(event, "event_type", None) == EventType.EVALUATION_COMPLETED
    }
    resolved = [by_id.get(event_id) for event_id in evidence_ids]
    if any(event is None for event in resolved):
        return ()

    expected_identity = (
        getattr(result, "run_id", None),
        getattr(result, "experiment_id", None),
        getattr(result, "population", None),
        getattr(result, "fidelity", None),
    )
    selected = [*resolved, terminal_event]
    for event in selected:
        candidate = event.payload.result
        identity = (
            getattr(candidate, "run_id", None),
            getattr(candidate, "experiment_id", None),
            getattr(candidate, "population", None),
            getattr(candidate, "fidelity", None),
        )
        if (
            identity != expected_identity
            or getattr(candidate.trust, "integrity", None) != Integrity.CLEAN
        ):
            return ()
    return tuple(selected)


def aggregate_metric_sets(metric_sets: Sequence[Any]) -> MetricSet:
    """Average compatible metric sets without introducing new metric semantics."""

    if not metric_sets:
        raise ValueError("at least one metric set is required")
    first = metric_sets[0]
    names = tuple(getattr(first, "metrics", {}).keys())
    primary_name = getattr(first, "primary_metric_name")
    for metric_set in metric_sets[1:]:
        if (
            getattr(metric_set, "primary_metric_name") != primary_name
            or set(getattr(metric_set, "metrics", {})) != set(names)
        ):
            raise ValueError("seed metric schemas do not match")
    count = len(metric_sets)
    metrics = {
        name: sum(float(metric_set.metrics[name]) for metric_set in metric_sets)
        / count
        for name in names
    }
    primary = sum(float(metric_set.primary_score) for metric_set in metric_sets) / count
    return MetricSet(
        metrics=metrics,
        primary_metric_name=primary_name,
        primary_score=primary,
    )


def mean_mapping(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average same-shaped protected metric-delta maps."""

    if not values or any(set(value) != set(values[0]) for value in values[1:]):
        return {}
    count = len(values)
    return {
        name: sum(float(value[name]) for value in values) / count
        for name in values[0]
    }


def stable_primary_score(result: Any) -> float:
    """Return the confirmed seed mean, or the current observation if unconfirmed."""

    trust = getattr(result, "trust", None)
    seed_mean = getattr(trust, "seed_mean", None)
    if seed_mean is not None:
        return float(seed_mean)
    return float(result.metric_set.primary_score)
