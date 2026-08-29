"""Metric-set comparison helpers."""

from typing import Mapping

from .types import MetricDelta, MetricSet


def compare_metric_sets(candidate: MetricSet, reference: MetricSet) -> MetricDelta:
    if candidate.primary_metric_name != reference.primary_metric_name:
        raise ValueError("primary metric names do not match")
    if set(candidate.metrics) != set(reference.metrics):
        raise ValueError("metric sets do not contain the same names")
    return MetricDelta(
        primary=candidate.primary_score - reference.primary_score,
        metrics={
            name: candidate.metrics[name] - reference.metrics[name]
            for name in candidate.metrics
        },
    )


def normalized_headroom(
    score: float, baseline: float, ceiling: float
) -> float:
    if ceiling <= baseline:
        raise ValueError("ceiling must be greater than baseline")
    return (float(score) - float(baseline)) / (float(ceiling) - float(baseline))
