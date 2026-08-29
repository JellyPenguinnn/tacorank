"""P0 baseline and independent-metric parity checks."""

from dataclasses import dataclass
import statistics
from typing import Mapping, Sequence, Tuple

from .metrics import evaluate_independent
from .types import MetricSet


@dataclass(frozen=True)
class ReferenceScore:
    model: str
    population: str
    expected: float
    observed: float
    passed: bool


@dataclass(frozen=True)
class IndependentMetricCheck:
    max_abs_deviation: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class SeedIndependenceCheck:
    model: str
    seeds: Tuple[int, ...]
    observed_std: float
    expected_std: float
    tolerance_factor: float
    minimum_std: float
    passed: bool


@dataclass(frozen=True)
class BaselineVerification:
    evaluator_sha256: str
    contract_sha256: str
    data_manifest_sha256: str
    independent_metric_check: IndependentMetricCheck
    reference_scores: Tuple[ReferenceScore, ...]
    seed_independence_check: SeedIndependenceCheck
    population_manifest: Mapping[str, Mapping[str, object]]
    all_passed: bool


def verify_metric_parity(
    official: MetricSet,
    user_ids: Sequence[object],
    labels: Sequence[int],
    scores: Sequence[float],
    tolerance: float = 1e-9,
) -> IndependentMetricCheck:
    independent = evaluate_independent(user_ids, labels, scores)
    deviations = [
        abs(official.metrics[name] - float(independent[name]))
        for name in official.metrics
    ]
    deviations.append(abs(official.primary_score - float(independent["primary"])))
    maximum = max(deviations)
    return IndependentMetricCheck(maximum, tolerance, maximum < tolerance)


def verify_reference_scores(
    observed: Mapping[Tuple[str, str], float],
    expected: Mapping[Tuple[str, str], float],
    tolerance: float = 0.0001,
) -> Tuple[ReferenceScore, ...]:
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError("reference score keys mismatch; missing=%s extra=%s" % (missing, extra))
    results = []
    for model, population in sorted(expected):
        wanted = float(expected[(model, population)])
        actual = float(observed[(model, population)])
        results.append(
            ReferenceScore(
                model,
                population,
                wanted,
                actual,
                abs(actual - wanted) <= tolerance,
            )
        )
    return tuple(results)


def verify_seed_independence(
    model: str,
    seeds: Sequence[int],
    scores: Sequence[float],
    expected_std: float = 0.0008,
    tolerance_factor: float = 3.0,
    minimum_std: float = 0.0002,
) -> SeedIndependenceCheck:
    if len(seeds) != len(scores) or len(scores) < 2:
        raise ValueError("seed independence requires at least two aligned scores")
    observed_std = statistics.stdev(float(score) for score in scores)
    lower = max(minimum_std, expected_std / tolerance_factor)
    upper = expected_std * tolerance_factor
    return SeedIndependenceCheck(
        model=model,
        seeds=tuple(int(seed) for seed in seeds),
        observed_std=observed_std,
        expected_std=expected_std,
        tolerance_factor=tolerance_factor,
        minimum_std=minimum_std,
        passed=lower <= observed_std <= upper,
    )


def build_baseline_verification(
    evaluator_sha256: str,
    contract_sha256: str,
    data_manifest_sha256: str,
    metric_check: IndependentMetricCheck,
    reference_scores: Sequence[ReferenceScore],
    seed_independence_check: SeedIndependenceCheck,
    population_manifest: Mapping[str, Mapping[str, object]],
) -> BaselineVerification:
    references = tuple(reference_scores)
    all_passed = (
        metric_check.passed
        and bool(references)
        and all(reference.passed for reference in references)
        and seed_independence_check.passed
    )
    return BaselineVerification(
        evaluator_sha256=evaluator_sha256,
        contract_sha256=contract_sha256,
        data_manifest_sha256=data_manifest_sha256,
        independent_metric_check=metric_check,
        reference_scores=references,
        seed_independence_check=seed_independence_check,
        population_manifest=dict(population_manifest),
        all_passed=all_passed,
    )
