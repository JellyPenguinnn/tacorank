"""Pure rolling detectors for training-metric anomalies."""

from __future__ import annotations

import math
import statistics
from typing import Iterable, List, Optional

from tacorank.schemas import TelemetrySample


def first_nonfinite_metric(sample: TelemetrySample) -> Optional[str]:
    """Return the first non-finite emitted metric, in stable priority order."""

    for name in ("loss", "gradient_norm"):
        value = getattr(sample, name, None)
        if value is not None and not math.isfinite(float(value)):
            return name
    return None


def persistent_explosion(
    samples: Iterable[TelemetrySample],
    metric: str,
    *,
    multiplier: float = 5.0,
    baseline_samples: int = 5,
    persistence: int = 2,
    baseline_floor: float = 1e-12,
) -> bool:
    """Detect consecutive spikes relative to a finite historical median.

    The candidate spike samples are excluded from their baseline so that the
    decision does not drift as a spike persists. Missing values disable the
    current decision rather than being treated as zeros.
    """

    if multiplier <= 1:
        raise ValueError("explosion multiplier must be greater than one")
    if baseline_samples < 1 or persistence < 2:
        raise ValueError("baseline_samples must be positive and persistence >= 2")

    values: List[Optional[float]] = []
    for sample in samples:
        raw = getattr(sample, metric, None)
        values.append(None if raw is None else float(raw))

    required = baseline_samples + persistence
    if len(values) < required:
        return False
    candidates = values[-persistence:]
    if any(value is None or not math.isfinite(value) for value in candidates):
        return False

    history = [
        value
        for value in values[:-persistence]
        if value is not None and math.isfinite(value)
    ]
    if len(history) < baseline_samples:
        return False
    baseline = abs(statistics.median(history[-baseline_samples:]))
    threshold = max(baseline, baseline_floor) * multiplier
    return all(abs(value) > threshold for value in candidates if value is not None)
