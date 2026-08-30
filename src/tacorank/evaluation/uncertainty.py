"""Deterministic paired uncertainty estimates for ranking-score deltas."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import DefaultDict, List, Sequence, Tuple

import numpy as np

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

    candidate_auc_array = np.asarray(candidate_auc_numerators, dtype=np.float64)
    reference_auc_array = np.asarray(reference_auc_numerators, dtype=np.float64)
    auc_denominator_array = np.asarray(auc_denominators, dtype=np.float64)
    candidate_ndcg_array = np.asarray(candidate_ndcg, dtype=np.float64)
    reference_ndcg_array = np.asarray(reference_ndcg, dtype=np.float64)
    users = len(grouped)

    point_delta = _primary_delta(
        np.ones(users, dtype=np.float64),
        candidate_auc_array,
        reference_auc_array,
        auc_denominator_array,
        candidate_ndcg_array,
        reference_ndcg_array,
    )
    rng = np.random.default_rng(int(seed))
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        selected = rng.integers(0, users, size=users)
        multiplicity = np.bincount(selected, minlength=users).astype(np.float64)
        samples[index] = _primary_delta(
            multiplicity,
            candidate_auc_array,
            reference_auc_array,
            auc_denominator_array,
            candidate_ndcg_array,
            reference_ndcg_array,
        )

    alpha = (1.0 - confidence) / 2.0
    return PairedDeltaInterval(
        mean=float(point_delta),
        standard_error=float(np.std(samples, ddof=1)),
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1.0 - alpha)),
        bootstrap_samples=bootstrap_samples,
    )


def _primary_delta(
    multiplicity: np.ndarray,
    candidate_auc: np.ndarray,
    reference_auc: np.ndarray,
    auc_denominator: np.ndarray,
    candidate_ndcg: np.ndarray,
    reference_ndcg: np.ndarray,
) -> float:
    denominator = float(np.dot(multiplicity, auc_denominator))
    if denominator:
        gauc_delta = float(
            np.dot(multiplicity, candidate_auc - reference_auc) / denominator
        )
    else:
        gauc_delta = 0.0
    ndcg_delta = float(
        np.dot(multiplicity, candidate_ndcg - reference_ndcg)
        / float(multiplicity.sum())
    )
    return (gauc_delta + ndcg_delta) / 2.0

