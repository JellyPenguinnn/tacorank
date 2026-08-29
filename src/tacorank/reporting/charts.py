"""Data builders for the six static final-report charts.

Rendering is deliberately kept outside the core so matplotlib is optional.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple


@dataclass(frozen=True)
class ChartSeries:
    title: str
    x_label: str
    y_label: str
    series: Mapping[str, Tuple[float, ...]]
    categories: Tuple[str, ...] = ()


def validation_divergence(
    query_indexes: Sequence[int],
    val_a: Sequence[float],
    val_b: Sequence[float],
) -> ChartSeries:
    if not (len(query_indexes) == len(val_a) == len(val_b)):
        raise ValueError("chart inputs must align")
    return ChartSeries(
        "Validation divergence",
        "public validation query",
        "primary score",
        {
            "x": tuple(float(value) for value in query_indexes),
            "val_a": tuple(float(value) for value in val_a),
            "val_b": tuple(float(value) for value in val_b),
        },
    )


def reported_vs_raw(reported: Sequence[float], raw: Sequence[float]) -> ChartSeries:
    if len(reported) != len(raw):
        raise ValueError("chart inputs must align")
    return ChartSeries(
        "Reported versus raw score",
        "raw primary",
        "gated primary",
        {
            "raw": tuple(float(value) for value in raw),
            "reported": tuple(float(value) for value in reported),
        },
    )


def combination_gain(
    predicted: Sequence[float], measured: Sequence[float]
) -> ChartSeries:
    if len(predicted) != len(measured):
        raise ValueError("chart inputs must align")
    return ChartSeries(
        "Predicted versus measured combination gain",
        "predicted gain",
        "measured gain",
        {
            "predicted": tuple(float(value) for value in predicted),
            "measured": tuple(float(value) for value in measured),
        },
    )


def verdict_census(counts: Mapping[str, int]) -> ChartSeries:
    categories = tuple(sorted(counts))
    return ChartSeries(
        "Experiment verdict census",
        "verdict",
        "experiments",
        {"count": tuple(float(counts[name]) for name in categories)},
        categories,
    )


def headroom_progress(
    query_indexes: Sequence[int],
    gauc_fraction: Sequence[float],
    ndcg_fraction: Sequence[float],
) -> ChartSeries:
    if not (len(query_indexes) == len(gauc_fraction) == len(ndcg_fraction)):
        raise ValueError("chart inputs must align")
    return ChartSeries(
        "Headroom progress",
        "public validation query",
        "headroom captured",
        {
            "x": tuple(float(value) for value in query_indexes),
            "gauc": tuple(float(value) for value in gauc_fraction),
            "ndcg": tuple(float(value) for value in ndcg_fraction),
        },
    )


def cost_by_role(costs: Mapping[str, float]) -> ChartSeries:
    categories = tuple(sorted(costs))
    return ChartSeries(
        "Resource cost by role",
        "role",
        "cost",
        {"cost": tuple(float(costs[name]) for name in categories)},
        categories,
    )
