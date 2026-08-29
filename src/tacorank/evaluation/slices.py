"""Per-user attribution, delta fingerprints, and drift diagnostics."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .metrics import auc, evaluate_independent, ndcg_at_k, normalize_binary_labels


@dataclass(frozen=True)
class UserMetric:
    user_id: str
    impressions: int
    positives: int
    auc: Optional[float]
    ndcg: float


@dataclass(frozen=True)
class UserSlice:
    name: str
    users: int
    gauc_positive_weight: int
    gauc: float
    ndcg: float
    primary: float


@dataclass(frozen=True)
class RankDiagnostic:
    name: str
    positive_rows: int
    mean_normalized_rank: float


@dataclass(frozen=True)
class DeltaVectorArtifact:
    path: str
    sha256: str
    size_bytes: int
    users: int
    ordered_user_ids_sha256: str


def user_metrics(
    user_ids: Sequence[object],
    labels: Sequence[int],
    scores: Sequence[float],
    k: int = 5,
) -> Tuple[UserMetric, ...]:
    if not (len(user_ids) == len(labels) == len(scores)) or not scores:
        raise ValueError("user IDs, labels and scores must align and be non-empty")
    normalized_labels = normalize_binary_labels(labels)
    grouped: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for user_id, label, score in zip(user_ids, normalized_labels, scores):
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("scores must be finite")
        grouped[str(user_id)].append((numeric_score, label))
    output = []
    for user_id, rows in grouped.items():
        rows.sort(key=lambda row: -row[0])
        labs = [label for _, label in rows]
        positives = sum(labs)
        user_auc = None
        if 0 < positives < len(labs):
            user_auc = auc(labs, [score for score, _ in rows])
        output.append(
            UserMetric(
                user_id=user_id,
                impressions=len(rows),
                positives=positives,
                auc=user_auc,
                ndcg=ndcg_at_k(labs, k),
            )
        )
    return tuple(output)


def slice_users(
    metrics: Sequence[UserMetric],
    bucket: Callable[[UserMetric], str],
) -> Mapping[str, UserSlice]:
    grouped: Dict[str, List[UserMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[bucket(metric)].append(metric)
    output = {}
    for name, values in grouped.items():
        gauc_weight = sum(value.positives for value in values if value.auc is not None)
        gauc = (
            sum(value.positives * value.auc for value in values if value.auc is not None)
            / gauc_weight
            if gauc_weight
            else 0.5
        )
        ndcg = sum(value.ndcg for value in values) / len(values)
        output[name] = UserSlice(
            name=name,
            users=len(values),
            gauc_positive_weight=gauc_weight,
            gauc=gauc,
            ndcg=ndcg,
            primary=(gauc + ndcg) / 2.0,
        )
    return output


def impression_bucket(metric: UserMetric) -> str:
    if metric.impressions <= 2:
        return "le2"
    if metric.impressions <= 5:
        return "3_5"
    if metric.impressions <= 12:
        return "6_12"
    return "gt12"


def positive_bucket(metric: UserMetric) -> str:
    if metric.positives == 0:
        return "zero"
    if metric.positives == 1:
        return "one"
    if metric.positives == 2:
        return "two"
    return "three_plus"


def reconstruct_from_user_slices(slices: Mapping[str, UserSlice]) -> Mapping[str, float]:
    """Exactly reconstruct metrics with their distinct official denominators."""
    if not slices:
        raise ValueError("at least one slice is required")
    users = sum(value.users for value in slices.values())
    gauc_weight = sum(value.gauc_positive_weight for value in slices.values())
    gauc = (
        sum(value.gauc * value.gauc_positive_weight for value in slices.values())
        / gauc_weight
        if gauc_weight
        else 0.5
    )
    ndcg = sum(value.ndcg * value.users for value in slices.values()) / users
    return {"GAUC": gauc, "nDCG@5": ndcg, "primary": (gauc + ndcg) / 2.0}


def delta_vector(
    user_ids: Sequence[object],
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    parent_scores: Sequence[float],
    k: int = 5,
) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    """Return exact per-user contributions to the official primary delta."""
    candidate = user_metrics(user_ids, labels, candidate_scores, k)
    parent = {metric.user_id: metric for metric in user_metrics(user_ids, labels, parent_scores, k)}
    total_users = len(candidate)
    total_positive_weight = sum(
        metric.positives for metric in candidate if metric.auc is not None
    )
    ids = []
    deltas = []
    for metric in candidate:
        previous = parent[metric.user_id]
        ndcg_delta = (metric.ndcg - previous.ndcg) / total_users
        auc_delta = 0.0
        if metric.auc is not None and total_positive_weight:
            assert previous.auc is not None
            auc_delta = (
                metric.positives * (metric.auc - previous.auc) / total_positive_weight
            )
        ids.append(metric.user_id)
        deltas.append((auc_delta + ndcg_delta) / 2.0)
    return tuple(ids), tuple(deltas)


def delta_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("delta vectors must be non-empty and aligned")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0 if list(left) == list(right) else 0.0
    return numerator / (left_norm * right_norm)


def gain_concentration(delta_values: Sequence[float], fraction: float = 0.10) -> float:
    """Share of absolute movement carried by the largest absolute deltas."""
    if not delta_values or not 0 < fraction <= 1:
        raise ValueError("invalid delta vector or concentration fraction")
    magnitudes = sorted((abs(float(value)) for value in delta_values), reverse=True)
    total = sum(magnitudes)
    if total == 0:
        return 0.0
    count = max(1, int(math.ceil(len(magnitudes) * fraction)))
    return sum(magnitudes[:count]) / total


def positive_rank_diagnostics(
    user_ids: Sequence[object],
    labels: Sequence[int],
    scores: Sequence[float],
    bucket_values: Sequence[object],
) -> Mapping[str, RankDiagnostic]:
    """Mean within-user normalized rank of positives for row-level buckets.

    These are diagnostics, not additive metric decompositions.  Tied scores use
    average ranks; 1.0 is top-ranked and 0.0 is bottom-ranked.
    """
    if not (
        len(user_ids) == len(labels) == len(scores) == len(bucket_values)
    ) or not scores:
        raise ValueError("rank diagnostic inputs must align and be non-empty")
    normalized_labels = normalize_binary_labels(labels)
    grouped: Dict[str, List[Tuple[int, float, int, str]]] = defaultdict(list)
    for index, (user_id, label, score, bucket) in enumerate(
        zip(user_ids, normalized_labels, scores, bucket_values)
    ):
        grouped[str(user_id)].append((index, float(score), label, str(bucket)))
    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for rows in grouped.values():
        descending = sorted(rows, key=lambda row: -row[1])
        index = 0
        while index < len(descending):
            end = index
            while end + 1 < len(descending) and descending[end + 1][1] == descending[index][1]:
                end += 1
            average_position = (index + end) / 2.0
            normalized = 1.0 if len(rows) == 1 else 1.0 - average_position / (len(rows) - 1)
            for position in range(index, end + 1):
                _, _, label, bucket = descending[position]
                if label == 1:
                    by_bucket[bucket].append(normalized)
            index = end + 1
    return {
        name: RankDiagnostic(name, len(values), sum(values) / len(values))
        for name, values in by_bucket.items()
    }


def duration_threshold_bucket(duration_ms: float) -> str:
    duration = float(duration_ms)
    if duration < 7000:
        return "short_lt7s"
    if 0.97 * duration >= 18000:
        return "flat_18s"
    return "mid"


def popularity_bucket(popularity: float, cold_max: float, warm_max: float) -> str:
    value = float(popularity)
    if cold_max > warm_max:
        raise ValueError("cold_max must not exceed warm_max")
    if value <= cold_max:
        return "cold"
    if value <= warm_max:
        return "warm"
    return "hot"


def save_delta_vector(
    path: Path,
    user_ids: Sequence[str],
    delta_values: Sequence[float],
) -> DeltaVectorArtifact:
    """Persist the ordered float32 vector as a NumPy artifact when available."""
    if len(user_ids) != len(delta_values) or not delta_values:
        raise ValueError("delta artifact inputs must align and be non-empty")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to write delta-vector artifacts") from exc
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output), np.asarray(delta_values, dtype=np.float32), allow_pickle=False)
    actual = output if output.suffix == ".npy" else Path(str(output) + ".npy")
    digest = hashlib.sha256(actual.read_bytes()).hexdigest()
    user_digest = hashlib.sha256()
    for user_id in user_ids:
        user_digest.update(str(user_id).encode("utf-8"))
        user_digest.update(b"\n")
    return DeltaVectorArtifact(
        path=str(actual),
        sha256=digest,
        size_bytes=actual.stat().st_size,
        users=len(user_ids),
        ordered_user_ids_sha256=user_digest.hexdigest(),
    )


def daily_primary_slope(
    dates: Sequence[int],
    user_ids: Sequence[object],
    labels: Sequence[int],
    scores: Sequence[float],
) -> float:
    if not (len(dates) == len(user_ids) == len(labels) == len(scores)):
        raise ValueError("daily diagnostic inputs must align")
    distinct_dates = sorted(set(int(date) for date in dates))
    if len(distinct_dates) < 2:
        return 0.0
    values = []
    for date in distinct_dates:
        indexes = [index for index, value in enumerate(dates) if int(value) == date]
        result = evaluate_independent(
            [user_ids[index] for index in indexes],
            [labels[index] for index in indexes],
            [scores[index] for index in indexes],
        )
        values.append(float(result["primary"]))
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator
