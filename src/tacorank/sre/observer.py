"""Public live SRE observer integration surface."""

from __future__ import annotations

from typing import Optional, Protocol

from tacorank.schemas import MonitorDirective, TelemetrySample
from tacorank.sre.health_policy import HealthPolicy, HealthPolicyConfig
from tacorank.sre.telemetry_window import TelemetryWindow


class HealthObserver(Protocol):
    def observe(self, sample: TelemetrySample) -> MonitorDirective:
        ...


class SREObserver(HealthObserver):
    """Synchronously turn one telemetry sample into a bounded directive."""

    def __init__(
        self,
        *,
        window_capacity: int = 60,
        policy: Optional[HealthPolicy] = None,
        config: Optional[HealthPolicyConfig] = None,
    ) -> None:
        if policy is not None and config is not None:
            raise ValueError("provide policy or config, not both")
        self.window = TelemetryWindow(window_capacity)
        self.policy = policy or HealthPolicy(config)
        self._execution_identity: Optional[tuple[str, str, int]] = None

    def observe(self, sample: TelemetrySample) -> MonitorDirective:
        identity = (sample.run_id, sample.experiment_id, sample.attempt)
        if self._execution_identity != identity:
            self.window.clear()
            self._execution_identity = identity
        self.window.add(sample)
        return self.policy.evaluate(self.window)
