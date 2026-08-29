"""Configurable health policy for the live execution sampling loop."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Pattern, Tuple

from tacorank.schemas import MonitorDirective
from tacorank.sre.anomaly_detection import first_nonfinite_metric, persistent_explosion
from tacorank.sre.heartbeat import HeartbeatPolicy, likely_hang
from tacorank.sre.telemetry_window import TelemetryWindow


_DEFAULT_OOM_PATTERNS = (
    r"\bCUDA out of memory\b",
    r"\bout of memory(?: error)?\b",
    r"\bMemoryError\b",
    r"\bcannot allocate memory\b",
)
_DEFAULT_POLICY_PATTERNS = (
    r"\bexecution policy violation\b",
    r"\bcontract violation\b",
    r"\bprotected path violation\b",
    r"\bunauthori[sz]ed (?:resource|access)\b",
)


@dataclass(frozen=True)
class HealthPolicyConfig:
    """Thresholds supplied by the frozen command/resource profile."""

    heartbeat: HeartbeatPolicy = field(default_factory=HeartbeatPolicy)
    cpu_memory_limit_mb: Optional[float] = None
    gpu_memory_limit_mb: Optional[float] = None
    disk_free_floor_mb: Optional[float] = None
    anomaly_multiplier: float = 5.0
    anomaly_baseline_samples: int = 5
    anomaly_persistence: int = 2
    oom_patterns: Tuple[str, ...] = _DEFAULT_OOM_PATTERNS
    policy_violation_patterns: Tuple[str, ...] = _DEFAULT_POLICY_PATTERNS

    def __post_init__(self) -> None:
        for name in (
            "cpu_memory_limit_mb",
            "gpu_memory_limit_mb",
            "disk_free_floor_mb",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError("resource thresholds cannot be negative")
        if self.anomaly_multiplier <= 1:
            raise ValueError("anomaly multiplier must be greater than one")
        if self.anomaly_baseline_samples < 1:
            raise ValueError("anomaly baseline must contain at least one sample")
        if self.anomaly_persistence < 2:
            raise ValueError("anomaly persistence must be at least two samples")


def _compiled(patterns: Tuple[str, ...]) -> Tuple[Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


class HealthPolicy:
    """Evaluate immediate failures before conservative rolling detectors."""

    def __init__(self, config: Optional[HealthPolicyConfig] = None) -> None:
        self.config = config or HealthPolicyConfig()
        self._oom_patterns = _compiled(self.config.oom_patterns)
        self._policy_patterns = _compiled(self.config.policy_violation_patterns)

    @staticmethod
    def _directive(
        action: str, reason: Optional[str], summary: Optional[str]
    ) -> MonitorDirective:
        # Shared schemas are Pydantic models; keyword construction also keeps
        # this boundary compatible with dataclass-based test doubles.
        return MonitorDirective(action=action, reason_code=reason, summary=summary)

    def evaluate(self, window: TelemetryWindow) -> MonitorDirective:
        sample = window.latest
        if sample is None:
            return self._directive("continue", None, None)

        if not bool(sample.process_alive):
            return self._directive("terminate", "PROCESS_DIED", "Process exited unexpectedly.")

        metric = first_nonfinite_metric(sample)
        if metric is not None:
            return self._directive(
                "terminate", "NUMERICAL_NONFINITE", f"Non-finite {metric} detected."
            )

        tail = sample.recent_output_tail or ""
        if any(pattern.search(tail) for pattern in self._oom_patterns):
            return self._directive(
                "terminate", "EXPLICIT_OOM", "Explicit out-of-memory failure detected."
            )

        disk_floor = self.config.disk_free_floor_mb
        if (
            disk_floor is not None
            and sample.disk_free_mb is not None
            and float(sample.disk_free_mb) < disk_floor
        ):
            return self._directive(
                "terminate", "DISK_LOW", "Available disk is below the configured safe floor."
            )

        if any(pattern.search(tail) for pattern in self._policy_patterns):
            return self._directive(
                "terminate",
                "EXECUTION_POLICY_VIOLATION",
                "Execution policy violation reported.",
            )

        cpu_limit = self.config.cpu_memory_limit_mb
        if cpu_limit is not None and float(sample.rss_mb) > cpu_limit:
            return self._directive(
                "terminate",
                "CPU_MEMORY_LIMIT",
                "Process memory exceeds the configured hard limit.",
            )

        gpu_limit = self.config.gpu_memory_limit_mb
        if (
            gpu_limit is not None
            and sample.gpu_memory_mb is not None
            and float(sample.gpu_memory_mb) > gpu_limit
        ):
            return self._directive(
                "terminate",
                "GPU_MEMORY_LIMIT",
                "GPU memory exceeds the configured hard limit.",
            )

        if likely_hang(window, self.config.heartbeat):
            return self._directive(
                "terminate", "HEARTBEAT_STALE", "Multiple signals indicate stalled progress."
            )

        anomaly_args = dict(
            multiplier=self.config.anomaly_multiplier,
            baseline_samples=self.config.anomaly_baseline_samples,
            persistence=self.config.anomaly_persistence,
        )
        if persistent_explosion(window, "loss", **anomaly_args):
            return self._directive(
                "terminate", "LOSS_EXPLOSION", "Persistent loss explosion detected."
            )
        if persistent_explosion(window, "gradient_norm", **anomaly_args):
            return self._directive(
                "terminate",
                "GRADIENT_EXPLOSION",
                "Persistent gradient-norm explosion detected.",
            )

        return self._directive("continue", None, None)
