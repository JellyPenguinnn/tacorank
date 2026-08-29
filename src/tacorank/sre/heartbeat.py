"""Multi-signal heartbeat/hang diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from tacorank.sre.telemetry_window import TelemetryWindow


@dataclass(frozen=True)
class HeartbeatPolicy:
    deadline_ms: int = 60_000
    minimum_stale_samples: int = 2
    cpu_idle_percent: float = 5.0
    cpu_unchanged_delta_percent: float = 0.5
    cpu_unchanged_ceiling_percent: float = 10.0
    gpu_expected: bool = False
    gpu_idle_percent: float = 5.0

    def __post_init__(self) -> None:
        if self.deadline_ms <= 0:
            raise ValueError("heartbeat deadline must be positive")
        if self.minimum_stale_samples < 2:
            raise ValueError("hang detection requires at least two stale samples")
        for name in (
            "cpu_idle_percent",
            "cpu_unchanged_delta_percent",
            "cpu_unchanged_ceiling_percent",
            "gpu_idle_percent",
        ):
            if getattr(self, name) < 0:
                raise ValueError("heartbeat utilization thresholds cannot be negative")


def likely_hang(window: TelemetryWindow, policy: HeartbeatPolicy) -> bool:
    """Return true only when stale output and idle resource signals agree.

    A missing GPU is explicitly normal unless the command profile says a GPU
    is expected. High CPU/GPU activity prevents a hang decision even when
    stdout is quiet, which protects long compute phases.
    """

    if len(window) < policy.minimum_stale_samples:
        return False
    recent = window.snapshot()[-policy.minimum_stale_samples :]
    if not all(bool(sample.process_alive) for sample in recent):
        return False
    if not all(
        int(sample.last_output_age_ms) >= policy.deadline_ms for sample in recent
    ):
        return False

    cpu_values = [float(sample.cpu_percent) for sample in recent]
    cpu_idle = all(value <= policy.cpu_idle_percent for value in cpu_values)
    cpu_low_and_unchanged = (
        max(cpu_values) <= policy.cpu_unchanged_ceiling_percent
        and max(cpu_values) - min(cpu_values) <= policy.cpu_unchanged_delta_percent
    )
    if not (cpu_idle or cpu_low_and_unchanged):
        return False

    if policy.gpu_expected:
        gpu_values = [sample.gpu_utilization_percent for sample in recent]
        if any(value is None for value in gpu_values):
            return False
        if not all(float(value) <= policy.gpu_idle_percent for value in gpu_values):
            return False

    return True
