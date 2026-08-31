"""Editable adaptation of the organizer's official NumPy FM baseline.

The frozen source remains ``kuairand-starter-kit/baseline.py``. This module
copies only the trainable model mathematics into the candidate-owned surface;
split selection, validation labels, evaluation, and submission logic remain
controller-owned. Research patches may modify this file in disposable
worktrees while the original baseline stays immutable.
"""

from __future__ import annotations

import numpy as np


OFFICIAL_RANK = 16
OFFICIAL_LEARNING_RATE = 0.001
OFFICIAL_L2 = 1e-6
OFFICIAL_BATCH_SIZE = 8192
OFFICIAL_MAX_EPOCHS = 40


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the baseline's numerically stable pointwise probabilities."""

    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


class OfficialFactorizationMachine:
    """Candidate-owned copy of the official five-field FM implementation."""

    def __init__(
        self,
        dimension: int,
        *,
        rank: int = OFFICIAL_RANK,
        learning_rate: float = OFFICIAL_LEARNING_RATE,
        l2: float = OFFICIAL_L2,
        seed: int = 0,
    ) -> None:
        if dimension < 1 or rank < 1:
            raise ValueError("dimension and rank must be positive")
        generator = np.random.default_rng(seed)
        self.factors = generator.normal(
            0.0, 0.01, (dimension, rank)
        ).astype(np.float32)
        self.weights = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self._factor_mean = np.zeros_like(self.factors)
        self._factor_variance = np.zeros_like(self.factors)
        self._weight_mean = np.zeros_like(self.weights)
        self._weight_variance = np.zeros_like(self.weights)
        self._step = 0

    def logits(self, features: np.ndarray) -> np.ndarray:
        embeddings = self.factors[features]
        summed = embeddings.sum(axis=1)
        interactions = 0.5 * (
            (summed ** 2).sum(axis=1)
            - (embeddings ** 2).sum(axis=(1, 2))
        )
        return self.bias + self.weights[features].sum(axis=1) + interactions

    def pointwise_step(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        if len(labels) == 0 or len(features) != len(labels):
            raise ValueError("training batch must be non-empty and aligned")
        embeddings = self.factors[features]
        summed = embeddings.sum(axis=1)
        logits = (
            self.bias
            + self.weights[features].sum(axis=1)
            + 0.5
            * (
                (summed ** 2).sum(axis=1)
                - (embeddings ** 2).sum(axis=(1, 2))
            )
        )
        probabilities = sigmoid(logits)
        gradient = ((probabilities - labels) / len(labels)).astype(np.float32)
        factor_gradient = np.zeros_like(self.factors)
        weight_gradient = np.zeros_like(self.weights)
        np.add.at(weight_gradient, features, gradient[:, None])
        np.add.at(
            factor_gradient,
            features,
            gradient[:, None, None]
            * (summed[:, None, :] - embeddings),
        )
        factor_gradient += self.l2 * self.factors
        weight_gradient += self.l2 * self.weights
        self._step += 1
        self._adam_update(
            self.factors,
            factor_gradient,
            self._factor_mean,
            self._factor_variance,
        )
        self._adam_update(
            self.weights,
            weight_gradient,
            self._weight_mean,
            self._weight_variance,
        )
        self.bias -= self.learning_rate * gradient.sum()
        return float(
            -np.mean(
                labels * np.log(probabilities + 1e-9)
                + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
            )
        )

    def predict(
        self, features: np.ndarray, *, batch_size: int = 200_000
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        scores = np.concatenate(
            [
                self.logits(features[start : start + batch_size])
                for start in range(0, len(features), batch_size)
            ]
        )
        if not np.isfinite(scores).all():
            raise ValueError("model produced non-finite scores")
        return scores

    def _adam_update(
        self,
        parameter: np.ndarray,
        gradient: np.ndarray,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> None:
        beta_one, beta_two, epsilon = 0.9, 0.999, 1e-8
        mean *= beta_one
        mean += (1.0 - beta_one) * gradient
        variance *= beta_two
        variance += (1.0 - beta_two) * (gradient * gradient)
        corrected_mean = mean / (1.0 - beta_one ** self._step)
        corrected_variance = variance / (1.0 - beta_two ** self._step)
        parameter -= self.learning_rate * corrected_mean / (
            np.sqrt(corrected_variance) + epsilon
        )


FactorizationMachine = OfficialFactorizationMachine
