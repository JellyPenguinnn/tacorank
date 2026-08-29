"""Deterministic aggregation of action-local resource measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import ResourceDelta, TokenMeasurement


@dataclass(frozen=True)
class ResourceTotals:
    provider_input_tokens: int = 0
    provider_output_tokens: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    unmeasured_input_tokens: int = 0
    unmeasured_output_tokens: int = 0
    action_wall_time_ms: int = 0
    cpu_time_ms: int = 0
    gpu_weighted_time_ms: int = 0
    manual_interventions: int = 0
    peak_rss_mb: int = 0
    peak_gpu_memory_mb: int = 0

    @property
    def gpu_hours(self) -> float:
        return self.gpu_weighted_time_ms / 3_600_000.0

    @property
    def provider_tokens(self) -> int:
        return self.provider_input_tokens + self.provider_output_tokens

    @property
    def estimated_tokens(self) -> int:
        return self.estimated_input_tokens + self.estimated_output_tokens

    @property
    def unmeasured_tokens(self) -> int:
        return self.unmeasured_input_tokens + self.unmeasured_output_tokens

    @property
    def total_reported_tokens(self) -> int:
        """All reported tokens, regardless of measurement provenance."""

        return self.provider_tokens + self.estimated_tokens + self.unmeasured_tokens


def aggregate_resources(deltas: Iterable[ResourceDelta]) -> ResourceTotals:
    values = list(deltas)
    by_measurement = {
        measurement: [item for item in values if item.token_measurement == measurement]
        for measurement in TokenMeasurement
    }
    return ResourceTotals(
        provider_input_tokens=sum(item.llm_input_tokens for item in by_measurement[TokenMeasurement.PROVIDER]),
        provider_output_tokens=sum(item.llm_output_tokens for item in by_measurement[TokenMeasurement.PROVIDER]),
        estimated_input_tokens=sum(item.llm_input_tokens for item in by_measurement[TokenMeasurement.ESTIMATED]),
        estimated_output_tokens=sum(item.llm_output_tokens for item in by_measurement[TokenMeasurement.ESTIMATED]),
        unmeasured_input_tokens=sum(item.llm_input_tokens for item in by_measurement[TokenMeasurement.NONE]),
        unmeasured_output_tokens=sum(item.llm_output_tokens for item in by_measurement[TokenMeasurement.NONE]),
        action_wall_time_ms=sum(item.wall_time_ms for item in values),
        cpu_time_ms=sum(item.cpu_time_ms for item in values),
        gpu_weighted_time_ms=sum(item.gpu_time_ms * item.gpu_count for item in values),
        manual_interventions=sum(item.manual_interventions for item in values),
        peak_rss_mb=max((item.peak_rss_mb or 0 for item in values), default=0),
        peak_gpu_memory_mb=max((item.peak_gpu_memory_mb or 0 for item in values), default=0),
    )
