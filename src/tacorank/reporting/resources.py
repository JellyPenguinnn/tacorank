"""Resource aggregation from immutable action-local event deltas."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class ResourceSummary:
    llm_input_tokens_provider: int
    llm_output_tokens_provider: int
    llm_input_tokens_estimated: int
    llm_output_tokens_estimated: int
    action_wall_time_ms: int
    cpu_time_ms: int
    gpu_hours: float
    manual_interventions: int
    peak_rss_mb: Optional[int]
    peak_gpu_memory_mb: Optional[int]


def aggregate_resources(deltas: Iterable[Mapping[str, object]]) -> ResourceSummary:
    provider_in = provider_out = estimated_in = estimated_out = 0
    wall = cpu = gpu_weighted_ms = interventions = 0
    peak_rss = peak_gpu = None
    for delta in deltas:
        source = str(delta.get("token_measurement", "none"))
        input_tokens = int(delta.get("llm_input_tokens", 0))
        output_tokens = int(delta.get("llm_output_tokens", 0))
        if source == "provider":
            provider_in += input_tokens
            provider_out += output_tokens
        elif source == "estimated":
            estimated_in += input_tokens
            estimated_out += output_tokens
        elif source != "none":
            raise ValueError("unknown token_measurement %r" % source)
        wall += int(delta.get("wall_time_ms", 0))
        cpu += int(delta.get("cpu_time_ms", 0))
        gpu_weighted_ms += int(delta.get("gpu_time_ms", 0)) * int(
            delta.get("gpu_count", 0)
        )
        interventions += int(delta.get("manual_interventions", 0))
        rss = delta.get("peak_rss_mb")
        gpu = delta.get("peak_gpu_memory_mb")
        if rss is not None:
            peak_rss = max(peak_rss or 0, int(rss))
        if gpu is not None:
            peak_gpu = max(peak_gpu or 0, int(gpu))
    return ResourceSummary(
        provider_in,
        provider_out,
        estimated_in,
        estimated_out,
        wall,
        cpu,
        gpu_weighted_ms / 3_600_000.0,
        interventions,
        peak_rss,
        peak_gpu,
    )
