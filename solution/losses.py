"""Candidate-owned ranking losses for research-plan experiments."""

from __future__ import annotations

import numpy as np


def pairwise_logistic_loss(score_differences: np.ndarray) -> float:
    """Return mean ``-log(sigmoid(positive - negative))`` stably."""

    values = np.asarray(score_differences, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("pairwise score differences must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("pairwise score differences must be finite")
    return float(np.logaddexp(0.0, -values).mean())


def full_observed_listwise_loss(
    scores: np.ndarray, labels: np.ndarray
) -> float:
    """Return normalized positive-mass softmax loss for one observed list."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.shape != targets.shape or len(values) == 0:
        raise ValueError("listwise scores and labels must be aligned vectors")
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("listwise inputs must be finite")
    positive_mass = targets.sum()
    if positive_mass <= 0.0:
        return 0.0
    shifted = values - values.max()
    log_normalizer = float(np.log(np.exp(shifted).sum()))
    target_distribution = targets / positive_mass
    return float(-(target_distribution * (shifted - log_normalizer)).sum())
