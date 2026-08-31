"""Candidate-owned, label-safe training orchestration.

The controller supplies ``fidelity`` and ``seed``. This helper never reads
validation/test labels and never performs metric-driven model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from solution.features import FeatureEncoder, TrainingRow, read_training_rows
from solution.model import FactorizationMachine


@dataclass(frozen=True)
class TrainingConfig:
    rank: int = 16
    learning_rate: float = 0.001
    l2: float = 1e-6
    batch_size: int = 8192
    duration_buckets: int = 10


@dataclass(frozen=True)
class TrainedCandidate:
    model: FactorizationMachine
    encoder: FeatureEncoder
    training_rows: int
    available_rows: int
    coverage_fraction: float
    epochs: int


@dataclass(frozen=True)
class _FidelityBudget:
    maximum_rows: Optional[int]
    epochs: int


_FIDELITY_BUDGETS = {
    "smoke": _FidelityBudget(maximum_rows=25_000, epochs=1),
    "proxy": _FidelityBudget(maximum_rows=250_000, epochs=2),
    "full": _FidelityBudget(maximum_rows=None, epochs=4),
}


def fit_pointwise(
    train_path: Path,
    *,
    fidelity: str,
    seed: int,
    config: TrainingConfig = TrainingConfig(),
) -> TrainedCandidate:
    """Fit the deterministic pointwise starting model on training data only."""

    if fidelity not in _FIDELITY_BUDGETS:
        raise ValueError("unsupported fidelity: %s" % fidelity)
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    available = read_training_rows(train_path)
    budget = _FIDELITY_BUDGETS[fidelity]
    rows = _representative_rows(available, budget.maximum_rows, seed)
    encoder = FeatureEncoder.fit(rows, duration_buckets=config.duration_buckets)
    features = encoder.transform(rows)
    labels = np.asarray([row.long_view for row in rows], dtype=np.float32)
    model = FactorizationMachine(
        encoder.dimension,
        rank=config.rank,
        learning_rate=config.learning_rate,
        l2=config.l2,
        seed=seed,
    )
    generator = np.random.default_rng(seed)
    for _ in range(budget.epochs):
        indices = generator.permutation(len(rows))
        for start in range(0, len(indices), config.batch_size):
            batch = indices[start : start + config.batch_size]
            model.pointwise_step(features[batch], labels[batch])
    return TrainedCandidate(
        model=model,
        encoder=encoder,
        training_rows=len(rows),
        available_rows=len(available),
        coverage_fraction=len(rows) / len(available),
        epochs=budget.epochs,
    )


def _representative_rows(
    rows: list[TrainingRow], maximum_rows: Optional[int], seed: int
) -> list[TrainingRow]:
    if maximum_rows is None or len(rows) <= maximum_rows:
        return rows
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(rows), size=maximum_rows, replace=False))
    return [rows[int(index)] for index in indices]
