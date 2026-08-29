"""Independent metric and diagnostic calculations.

The official benchmark evaluator remains the production scoring path.  The
functions here exist for parity tests and per-user diagnostics only.
"""

from collections import defaultdict
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .types import MetricSet


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mann-Whitney AUC with average ranks for tied scores."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    pairs = sorted(zip((float(s) for s in scores), (int(y) for y in labels)))
    if any(y not in (0, 1) for _, y in pairs):
        raise ValueError("AUC labels must be binary")
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average_rank
        index = end + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    positive_rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, pairs) if label == 1
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def ndcg_at_k(ranked_labels: Sequence[int], k: int = 5) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    labels = [int(value) for value in ranked_labels]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("nDCG labels must be binary")
    discounts = [math.log2(position + 2) for position in range(k)]
    dcg = sum(
        ((2 ** label) - 1) / discounts[index]
        for index, label in enumerate(labels[:k])
    )
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(
        ((2 ** label) - 1) / discounts[index]
        for index, label in enumerate(ideal)
    )
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate_independent(
    user_ids: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    k: int = 5,
) -> Mapping[str, float]:
    """Reproduce the starter kit evaluator for parity checks."""
    _validate_equal_nonempty(user_ids, labels, scores)
    grouped: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for user_id, label, score in zip(user_ids, labels, scores):
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("scores must be finite")
        grouped[str(user_id)].append((numeric_score, int(label)))

    weighted_auc = 0.0
    positive_weight = 0.0
    ndcg_values: List[float] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: -row[0])
        ranked_labels = [label for _, label in rows]
        positives = sum(ranked_labels)
        if 0 < positives < len(ranked_labels):
            weighted_auc += positives * auc(
                ranked_labels, [score for score, _ in rows]
            )
            positive_weight += positives
        ndcg_values.append(ndcg_at_k(ranked_labels, k))

    gauc = weighted_auc / positive_weight if positive_weight else 0.5
    ndcg = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0
    return {
        "GAUC": gauc,
        "nDCG@%d" % k: ndcg,
        "primary": (gauc + ndcg) / 2.0,
        "users": len(grouped),
        "rows": len(labels),
    }


def per_user_contributions(
    user_ids: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    k: int = 5,
) -> Mapping[str, Tuple[float, float, float]]:
    """Return ``user -> (AUC or NaN, nDCG, primary diagnostic)``.

    GAUC's positive weighting is applied only when aggregating the official
    metric.  The per-user primary here is a diagnostic fingerprint, with nDCG
    used alone for single-class groups.
    """
    _validate_equal_nonempty(user_ids, labels, scores)
    grouped: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for user_id, label, score in zip(user_ids, labels, scores):
        grouped[str(user_id)].append((float(score), int(label)))
    output: Dict[str, Tuple[float, float, float]] = {}
    for user_id, rows in grouped.items():
        rows.sort(key=lambda row: -row[0])
        labs = [label for _, label in rows]
        ndcg = ndcg_at_k(labs, k)
        positives = sum(labs)
        if 0 < positives < len(labs):
            user_auc = auc(labs, [score for score, _ in rows])
            primary = (user_auc + ndcg) / 2.0
        else:
            user_auc = float("nan")
            primary = ndcg
        output[user_id] = (user_auc, ndcg, primary)
    return output


def validate_metric_set(
    raw: Mapping[str, object],
    required_metrics: Sequence[str],
    primary_metric_name: str,
    primary_weights: Mapping[str, float],
    tolerance: float = 1e-12,
    allow_extra: Sequence[str] = ("users", "rows"),
    metric_ranges: Mapping[str, Tuple[float, float]] = None,
) -> MetricSet:
    """Validate official evaluator output against a frozen contract."""
    required = set(required_metrics)
    actual_metric_names = set(raw) - {primary_metric_name} - set(allow_extra)
    missing = required - actual_metric_names
    extra = actual_metric_names - required
    if missing:
        raise ValueError("missing required metrics: %s" % sorted(missing))
    if extra:
        raise ValueError("undeclared metrics: %s" % sorted(extra))
    if primary_metric_name not in raw:
        raise ValueError("missing primary metric %r" % primary_metric_name)

    metrics = {name: _finite(raw[name], name) for name in required_metrics}
    ranges = metric_ranges or {}
    for name, value in metrics.items():
        if name in ranges:
            lower, upper = ranges[name]
            if value < lower or value > upper:
                raise ValueError("metric %s outside [%s, %s]" % (name, lower, upper))
    if set(primary_weights) != required:
        raise ValueError("primary weights must cover exactly the required metrics")
    total_weight = sum(float(weight) for weight in primary_weights.values())
    if total_weight <= 0:
        raise ValueError("primary weights must have positive total weight")
    reproduced = sum(
        metrics[name] * float(primary_weights[name]) for name in required_metrics
    ) / total_weight
    primary = _finite(raw[primary_metric_name], primary_metric_name)
    if abs(primary - reproduced) > tolerance:
        raise ValueError(
            "primary aggregation mismatch: observed %.17g, reproduced %.17g"
            % (primary, reproduced)
        )
    return MetricSet(metrics, primary_metric_name, primary)


def _finite(value: object, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError("metric %s is not numeric" % name)
    if not math.isfinite(numeric):
        raise ValueError("metric %s must be finite" % name)
    return numeric


def _validate_equal_nonempty(*values: Sequence[object]) -> None:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("inputs must have equal lengths")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("evaluation inputs must not be empty")
