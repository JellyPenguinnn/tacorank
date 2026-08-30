"""Protected aggregate diagnostics for explaining evaluation movement."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from tacorank.schemas import EvaluationDiagnostics

from .metrics import evaluate_independent
from .slices import (
    delta_vector,
    duration_threshold_bucket,
    gain_concentration,
    impression_bucket,
    popularity_bucket,
    positive_bucket,
    positive_rank_diagnostics,
    slice_users,
    user_metrics,
)
from .types import MetricDelta


@dataclass(frozen=True)
class DiagnosticFeatures:
    """Manifest-attested row features available only to protected evaluation."""

    dates: Tuple[int, ...]
    duration_ms: Tuple[float, ...]
    item_popularity: Tuple[int, ...]
    user_history_count: Tuple[int, ...]
    validation_arms: Tuple[str, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.dates),
            len(self.duration_ms),
            len(self.item_popularity),
            len(self.user_history_count),
            len(self.validation_arms),
        }
        if len(lengths) != 1 or not self.dates:
            raise ValueError("diagnostic feature columns must align and be non-empty")
        if any(arm not in {"val_a", "val_b"} for arm in self.validation_arms):
            raise ValueError("diagnostic validation arms must be val_a or val_b")
        if any(value < 0 for value in self.duration_ms):
            raise ValueError("diagnostic durations must be non-negative")
        if any(value < 0 for value in self.item_popularity):
            raise ValueError("diagnostic item popularity must be non-negative")
        if any(value < 0 for value in self.user_history_count):
            raise ValueError("diagnostic history counts must be non-negative")


def compute_evaluation_diagnostics(
    *,
    user_ids: Sequence[object],
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    parent_scores: Sequence[float],
    parent_delta: MetricDelta,
    features: Optional[DiagnosticFeatures] = None,
    proxy_parent_delta: Optional[float] = None,
    train_validation_gap: Optional[float] = None,
    validation_gap_threshold: float = 0.006,
    temporal_slope_threshold: float = 0.002,
    concentration_threshold: float = 0.70,
) -> EvaluationDiagnostics:
    """Calculate aggregate diagnostics without exposing protected row evidence."""

    if validation_gap_threshold < 0 or temporal_slope_threshold < 0:
        raise ValueError("diagnostic thresholds must be non-negative")
    if not 0 <= concentration_threshold <= 1:
        raise ValueError("concentration threshold must be between zero and one")
    lengths = {
        len(user_ids),
        len(labels),
        len(candidate_scores),
        len(parent_scores),
    }
    if len(lengths) != 1 or not user_ids:
        raise ValueError("diagnostic predictions and labels must align and be non-empty")
    if features is not None and len(features.dates) != len(user_ids):
        raise ValueError("diagnostic features do not match the evaluation population")

    _, contribution_deltas = delta_vector(
        user_ids, labels, candidate_scores, parent_scores
    )
    concentration = gain_concentration(contribution_deltas)
    slices: Dict[str, float] = {}
    slices.update(
        _user_slice_deltas(
            user_ids,
            labels,
            candidate_scores,
            parent_scores,
            impression_bucket,
            "user_impressions",
        )
    )
    slices.update(
        _user_slice_deltas(
            user_ids,
            labels,
            candidate_scores,
            parent_scores,
            positive_bucket,
            "user_positives",
        )
    )

    arm_deltas: Dict[str, float] = {}
    temporal_slope: Optional[float] = None
    if features is not None:
        arm_deltas = _validation_arm_deltas(
            features.validation_arms,
            user_ids,
            labels,
            candidate_scores,
            parent_scores,
        )
        slices.update(
            {"validation_arm.%s" % name: value for name, value in arm_deltas.items()}
        )
        temporal_deltas = _temporal_deltas(
            features.dates,
            user_ids,
            labels,
            candidate_scores,
            parent_scores,
        )
        slices.update(
            {"temporal.%s" % date: value for date, value in temporal_deltas.items()}
        )
        temporal_slope = _linear_slope(
            [temporal_deltas[date] for date in sorted(temporal_deltas)]
        )

        history_by_user = _constant_user_values(
            user_ids, features.user_history_count, "history count"
        )
        history_low, history_high = _tertile_thresholds(tuple(history_by_user.values()))
        slices.update(
            _user_slice_deltas(
                user_ids,
                labels,
                candidate_scores,
                parent_scores,
                lambda metric: _count_bucket(
                    history_by_user[metric.user_id], history_low, history_high
                ),
                "user_history",
            )
        )

        duration_buckets = tuple(
            duration_threshold_bucket(value) for value in features.duration_ms
        )
        slices.update(
            _rank_deltas(
                user_ids,
                labels,
                candidate_scores,
                parent_scores,
                duration_buckets,
                "duration_rank",
            )
        )
        popularity_low, popularity_high = _tertile_thresholds(
            features.item_popularity
        )
        popularity_buckets = tuple(
            popularity_bucket(value, popularity_low, popularity_high)
            for value in features.item_popularity
        )
        slices.update(
            _rank_deltas(
                user_ids,
                labels,
                candidate_scores,
                parent_scores,
                popularity_buckets,
                "popularity_rank",
            )
        )

    best_slice = max(slices, key=lambda name: (slices[name], name)) if slices else None
    worst_slice = min(slices, key=lambda name: (slices[name], name)) if slices else None
    arm_gap = (
        abs(arm_deltas["val_a"] - arm_deltas["val_b"])
        if set(arm_deltas) == {"val_a", "val_b"}
        else None
    )
    proxy_gap = (
        abs(float(proxy_parent_delta) - parent_delta.primary)
        if proxy_parent_delta is not None
        else None
    )
    hypotheses = _failure_hypotheses(
        parent_delta=parent_delta,
        proxy_parent_delta=proxy_parent_delta,
        proxy_gap=proxy_gap,
        arm_deltas=arm_deltas,
        arm_gap=arm_gap,
        temporal_slope=temporal_slope,
        concentration=concentration,
        worst_slice=worst_slice,
        slices=slices,
        validation_gap_threshold=validation_gap_threshold,
        temporal_slope_threshold=temporal_slope_threshold,
        concentration_threshold=concentration_threshold,
    )
    limitations = [
        "Associational diagnostics do not prove causality; confirm hypotheses with a controlled ablation."
    ]
    if train_validation_gap is None:
        limitations.insert(
            0,
            "Direct train/validation gap unavailable: contract v1 emits no protected train predictions.",
        )
    return EvaluationDiagnostics(
        train_validation_gap=train_validation_gap,
        proxy_parent_delta=proxy_parent_delta,
        proxy_full_delta_gap=proxy_gap,
        validation_arm_deltas=arm_deltas,
        validation_arm_gap=arm_gap,
        temporal_delta_slope=temporal_slope,
        gain_concentration_top10pct=concentration,
        slice_deltas=dict(sorted(slices.items())),
        best_slice=best_slice,
        worst_slice=worst_slice,
        failure_hypotheses=hypotheses,
        limitations=limitations,
    )


def _user_slice_deltas(
    user_ids: Sequence[object],
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    parent_scores: Sequence[float],
    bucket: Callable[[object], str],
    prefix: str,
) -> Dict[str, float]:
    candidate = slice_users(user_metrics(user_ids, labels, candidate_scores), bucket)
    parent = slice_users(user_metrics(user_ids, labels, parent_scores), bucket)
    return {
        "%s.%s" % (prefix, name): candidate[name].primary - parent[name].primary
        for name in sorted(set(candidate).intersection(parent))
    }


def _rank_deltas(
    user_ids: Sequence[object],
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    parent_scores: Sequence[float],
    bucket_values: Sequence[object],
    prefix: str,
) -> Dict[str, float]:
    candidate = positive_rank_diagnostics(
        user_ids, labels, candidate_scores, bucket_values
    )
    parent = positive_rank_diagnostics(user_ids, labels, parent_scores, bucket_values)
    return {
        "%s.%s" % (prefix, name): (
            candidate[name].mean_normalized_rank - parent[name].mean_normalized_rank
        )
        for name in sorted(set(candidate).intersection(parent))
    }


def _validation_arm_deltas(
    arms: Sequence[str],
    user_ids: Sequence[object],
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    parent_scores: Sequence[float],
) -> Dict[str, float]:
    result = {}
    for arm in ("val_a", "val_b"):
        indexes = [index for index, value in enumerate(arms) if value == arm]
        if indexes:
            result[arm] = _primary_delta(
                indexes, user_ids, labels, candidate_scores, parent_scores
            )
    return result if set(result) == {"val_a", "val_b"} else {}


def _temporal_deltas(
    dates: Sequence[int],
    user_ids: Sequence[object],
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    parent_scores: Sequence[float],
) -> Dict[str, float]:
    result = {}
    for date in sorted(set(int(value) for value in dates)):
        indexes = [index for index, value in enumerate(dates) if int(value) == date]
        result[str(date)] = _primary_delta(
            indexes, user_ids, labels, candidate_scores, parent_scores
        )
    return result


def _primary_delta(
    indexes: Sequence[int],
    user_ids: Sequence[object],
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    parent_scores: Sequence[float],
) -> float:
    candidate = evaluate_independent(
        [user_ids[index] for index in indexes],
        [labels[index] for index in indexes],
        [candidate_scores[index] for index in indexes],
    )
    parent = evaluate_independent(
        [user_ids[index] for index in indexes],
        [labels[index] for index in indexes],
        [parent_scores[index] for index in indexes],
    )
    return float(candidate["primary"] - parent["primary"])


def _constant_user_values(
    user_ids: Sequence[object], values: Sequence[int], label: str
) -> Mapping[str, int]:
    result: Dict[str, int] = {}
    for raw_user_id, raw_value in zip(user_ids, values):
        user_id = str(raw_user_id)
        value = int(raw_value)
        if user_id in result and result[user_id] != value:
            raise ValueError("diagnostic %s must be constant within user" % label)
        result[user_id] = value
    return result


def _tertile_thresholds(values: Sequence[int]) -> Tuple[float, float]:
    if not values:
        raise ValueError("diagnostic thresholds require values")
    ordered = sorted(float(value) for value in values)
    low = ordered[(len(ordered) - 1) // 3]
    high = ordered[(2 * (len(ordered) - 1)) // 3]
    return low, high


def _count_bucket(value: int, low: float, high: float) -> str:
    if value <= low:
        return "cold"
    if value <= high:
        return "warm"
    return "hot"


def _linear_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator == 0:
        return 0.0
    slope = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator
    if not math.isfinite(slope):
        raise ValueError("diagnostic temporal slope is not finite")
    return slope


def _failure_hypotheses(
    *,
    parent_delta: MetricDelta,
    proxy_parent_delta: Optional[float],
    proxy_gap: Optional[float],
    arm_deltas: Mapping[str, float],
    arm_gap: Optional[float],
    temporal_slope: Optional[float],
    concentration: float,
    worst_slice: Optional[str],
    slices: Mapping[str, float],
    validation_gap_threshold: float,
    temporal_slope_threshold: float,
    concentration_threshold: float,
) -> list[str]:
    hypotheses = []
    if parent_delta.primary < 0:
        hypotheses.append(
            "Broad regression: the aggregate primary score is below the declared parent."
        )
    regressed_metrics = sorted(
        name for name, value in parent_delta.metrics.items() if value < 0
    )
    if regressed_metrics:
        hypotheses.append(
            "Metric weakness: %s regressed versus the parent."
            % ", ".join(regressed_metrics)
        )
    if proxy_parent_delta is not None:
        if proxy_parent_delta * parent_delta.primary < 0:
            hypotheses.append(
                "Generalization mismatch: proxy and full parent deltas have opposite signs."
            )
        elif proxy_gap is not None and proxy_gap >= validation_gap_threshold:
            hypotheses.append(
                "Generalization uncertainty: proxy and full effect sizes differ materially."
            )
    if set(arm_deltas) == {"val_a", "val_b"}:
        if arm_deltas["val_a"] * arm_deltas["val_b"] < 0:
            hypotheses.append(
                "Validation-arm sensitivity: Val-A and Val-B parent deltas have opposite signs."
            )
        elif arm_gap is not None and arm_gap >= validation_gap_threshold:
            hypotheses.append(
                "Validation-arm sensitivity: Val-A and Val-B effect sizes differ materially."
            )
    if temporal_slope is not None and temporal_slope < -temporal_slope_threshold:
        hypotheses.append(
            "Temporal degradation: candidate-parent performance declines on later dates."
        )
    if concentration >= concentration_threshold:
        hypotheses.append(
            "Concentrated movement: a small user cohort carries most score movement."
        )
    if worst_slice is not None and slices[worst_slice] < 0:
        hypotheses.append(
            "Cohort weakness: the largest measured slice regression is %s."
            % worst_slice
        )
    return hypotheses
