"""Bounded, deterministic storage for live execution telemetry."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, List, Optional, Sequence

from tacorank.schemas import TelemetrySample


class TelemetryWindow(Sequence[TelemetrySample]):
    """A fixed-capacity window ordered from oldest to newest sample.

    ``deque(maxlen=...)`` makes eviction atomic and keeps observer memory
    independent of run duration.  The class intentionally does not use wall
    clock time; detectors consume timestamps carried by the samples instead.
    """

    def __init__(self, capacity: int = 60) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("telemetry window capacity must be a positive integer")
        self._samples: Deque[TelemetrySample] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._samples.maxlen or 0

    @property
    def latest(self) -> Optional[TelemetrySample]:
        return self._samples[-1] if self._samples else None

    def add(self, sample: TelemetrySample) -> None:
        self._samples.append(sample)

    def snapshot(self) -> List[TelemetrySample]:
        """Return an isolated snapshot suitable for deterministic analysis."""

        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.snapshot()[index]

    def __iter__(self) -> Iterator[TelemetrySample]:
        return iter(self._samples)
