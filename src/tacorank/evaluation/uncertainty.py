"""Deterministic paired uncertainty estimates for ranking-score deltas."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import DefaultDict, Iterable, List, Sequence, Tuple

from .metrics import auc, ndcg_at_k, normalize_binary_labels


@dataclass(frozen=True)
class PairedDeltaInterval:
    mean: float
    standard_error: float
    lower: float
    upper: float
    bootstrap_samples: int


def paired_user_delta_interval(
    user_ids: Sequence[str],
    labels: Sequence[int],
    candidate_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    seed: int,
    bootstrap_samples: int = 400,
    confidence: float = 0.95,
    k: int = 5,
) -> PairedDeltaInterval:
    """Estimate a primary-score delta by resampling complete user groups.

    GAUC is reconstructed with its official positive-count weighting while nDCG
    retains one vote per user. Candidate and reference contributions always use
    the same bootstrap multiplicity, preserving their paired dependence.
    """

    lengths = {
        len(user_ids),
        len(labels),
        len(candidate_scores),
        len(reference_scores),
    }
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("paired uncertainty inputs must be aligned and non-empty")
    if bootstrap_samples < 20:
        raise ValueError("paired uncertainty requires at least 20 bootstrap samples")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")

    normalized = normalize_binary_labels(labels)
    grouped: DefaultDict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for user_id, label, candidate, reference in zip(
        user_ids, normalized, candidate_scores, reference_scores
    ):
        candidate_value = float(candidate)
        reference_value = float(reference)
        if not math.isfinite(candidate_value) or not math.isfinite(reference_value):
            raise ValueError("paired uncertainty scores must be finite")
        grouped[str(user_id)].append((label, candidate_value, reference_value))

    candidate_auc_numerators = []
    reference_auc_numerators = []
    auc_denominators = []
    candidate_ndcg = []
    reference_ndcg = []
    for user_id in sorted(grouped):
        rows = grouped[user_id]
        user_labels = [row[0] for row in rows]
        candidate_values = [row[1] for row in rows]
        reference_values = [row[2] for row in rows]
        positives = sum(user_labels)
        if 0 < positives < len(rows):
            candidate_auc_numerators.append(
                positives * auc(user_labels, candidate_values)
            )
            reference_auc_numerators.append(
                positives * auc(user_labels, reference_values)
            )
            auc_denominators.append(float(positives))
        else:
            candidate_auc_numerators.append(0.0)
            reference_auc_numerators.append(0.0)
            auc_denominators.append(0.0)
        candidate_order = sorted(
            range(len(rows)), key=lambda index: -candidate_values[index]
        )
        reference_order = sorted(
            range(len(rows)), key=lambda index: -reference_values[index]
        )
        candidate_ndcg.append(
            ndcg_at_k([user_labels[index] for index in candidate_order], k)
        )
        reference_ndcg.append(
            ndcg_at_k([user_labels[index] for index in reference_order], k)
        )

    auc_deltas = [
        candidate - reference
        for candidate, reference in zip(
            candidate_auc_numerators, reference_auc_numerators
        )
    ]
    ndcg_deltas = [
        candidate - reference
        for candidate, reference in zip(candidate_ndcg, reference_ndcg)
    ]
    users = len(grouped)

    point_delta = _primary_delta(
        range(users), auc_deltas, auc_denominators, ndcg_deltas
    )
    rng = random.Random(int(seed))
    samples = [
        _primary_delta(
            (rng.randrange(users) for _ in range(users)),
            auc_deltas,
            auc_denominators,
            ndcg_deltas,
        )
        for _ in range(bootstrap_samples)
    ]

    alpha = (1.0 - confidence) / 2.0
    return PairedDeltaInterval(
        mean=point_delta,
        standard_error=_sample_standard_deviation(samples),
        lower=_linear_quantile(samples, alpha),
        upper=_linear_quantile(samples, 1.0 - alpha),
        bootstrap_samples=bootstrap_samples,
    )


def _primary_delta(
    selected_users: Iterable[int],
    auc_deltas: Sequence[float],
    auc_denominators: Sequence[float],
    ndcg_deltas: Sequence[float],
) -> float:
    auc_numerator = 0.0
    auc_denominator = 0.0
    ndcg_numerator = 0.0
    count = 0
    for user_index in selected_users:
        auc_numerator += auc_deltas[user_index]
        auc_denominator += auc_denominators[user_index]
        ndcg_numerator += ndcg_deltas[user_index]
        count += 1
    if count < 1:
        raise ValueError("paired uncertainty requires at least one user")
    gauc_delta = auc_numerator / auc_denominator if auc_denominator else 0.0
    ndcg_delta = ndcg_numerator / count
    return (gauc_delta + ndcg_delta) / 2.0


def _sample_standard_deviation(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight
