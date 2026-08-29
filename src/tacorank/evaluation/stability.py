"""Seed aggregation and adaptive leaderboard gating."""

from dataclasses import dataclass
import math
import statistics
from typing import Optional, Sequence, Tuple

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
