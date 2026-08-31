"""Prediction-change and silent no-op detection."""

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Hashable, Optional, Sequence

from .types import PredictionChange


@dataclass(frozen=True)
class NoOpConfig:
    score_tolerance: float = 1e-12
    spearman_threshold: float = 0.9999
    changed_row_fraction_max: float = 0.001
    primary_delta_tolerance: float = 0.0016


def analyze_prediction_change(
    scores: Sequence[float],
    parent_scores: Optional[Sequence[float]],
    tolerance: float = 1e-12,
    user_ids: Optional[Sequence[Hashable]] = None,
) -> PredictionChange:
    values = _finite_scores(scores)
    unique_fraction = len(set(values)) / len(values)
    if parent_scores is None:
        return PredictionChange(None, None, None, unique_fraction)
    parent = _finite_scores(parent_scores)
    if len(values) != len(parent):
        raise ValueError("candidate and parent predictions must align")
    changed = sum(
        abs(left - right) > tolerance for left, right in zip(values, parent)
    )
    identical = sum(left == right for left, right in zip(values, parent))
    within_user_spearman, within_user_changed = _within_user_rank_change(
        values,
        parent,
        user_ids,
    )
    return PredictionChange(
        spearman_vs_parent=spearman(values, parent),
        changed_row_fraction=changed / len(values),
        identical_score_fraction=identical / len(values),
        unique_score_fraction=unique_fraction,
        within_user_spearman_vs_parent=within_user_spearman,
        within_user_rank_changed_fraction=within_user_changed,
    )


def is_no_op(
    change: PredictionChange,
    parent_delta: float,
    config: NoOpConfig,
) -> bool:
    if (
        change.within_user_spearman_vs_parent is not None
        and change.within_user_rank_changed_fraction is not None
    ):
        return (
            change.within_user_spearman_vs_parent >= config.spearman_threshold
            and change.within_user_rank_changed_fraction
            <= config.changed_row_fraction_max
            and abs(float(parent_delta)) <= config.primary_delta_tolerance
        )
    if change.spearman_vs_parent is None or change.changed_row_fraction is None:
        return False
    return (
        change.spearman_vs_parent >= config.spearman_threshold
        and change.changed_row_fraction <= config.changed_row_fraction_max
        and abs(float(parent_delta)) <= config.primary_delta_tolerance
    )


def _within_user_rank_change(
    scores: Sequence[float],
    parent_scores: Sequence[float],
    user_ids: Optional[Sequence[Hashable]],
) -> tuple[Optional[float], Optional[float]]:
    if user_ids is None:
        return None, None
    if len(user_ids) != len(scores):
        raise ValueError("user IDs and predictions must align")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, user_id in enumerate(user_ids):
        groups[user_id].append(index)
    correlations = []
    changed = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        candidate = [scores[index] for index in indices]
        parent = [parent_scores[index] for index in indices]
        correlations.append(spearman(candidate, parent))
        if list(_average_ranks(candidate)) != list(_average_ranks(parent)):
            changed += 1
    if not correlations:
        return None, None
    return sum(correlations) / len(correlations), changed / len(correlations)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths")
    if not left:
        raise ValueError("Spearman inputs must not be empty")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_norm == 0 or right_norm == 0:
        return 1.0 if list(left) == list(right) else 0.0
    return numerator / (left_norm * right_norm)


def _average_ranks(values: Sequence[float]) -> Sequence[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[indexed[position][0]] = average
        index = end + 1
    return ranks


def _finite_scores(scores: Sequence[float]) -> Sequence[float]:
    if not scores:
        raise ValueError("predictions must not be empty")
    values = [float(value) for value in scores]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("predictions must be finite")
    return values
