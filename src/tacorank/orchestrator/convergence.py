"""Frozen deterministic stop and convergence rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import RunConfig
from ..orchestrator.state import RunState
from ..schemas import Event, EventType, Fidelity, Population, TrustVerdict


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason_code: str = "continue"
    reason: str = "No deterministic stop condition matched."


def convergence_pressure(events: Sequence[Event], config: RunConfig) -> int:
    """Count consecutive trusted full evaluations without epsilon improvement."""

    incumbent = None
    non_improving = 0
    for event in events:
        if event.event_type == EventType.BASELINE_VERIFIED:
            incumbent = event.payload.metric_set.primary_score
        elif event.event_type == EventType.EVALUATION_COMPLETED:
            result = event.payload.result
            if not (
                result.fidelity == Fidelity.FULL
                and result.population == Population.PUBLIC_VALIDATION
                and result.trust.verdict == TrustVerdict.ACCEPTED
            ):
                continue
            score = result.metric_set.primary_score
            if incumbent is None or score > incumbent + config.convergence_epsilon:
                incumbent = score
                non_improving = 0
            else:
                non_improving += 1
    return non_improving


def stop_decision(
    state: RunState,
    events: Sequence[Event],
    config: RunConfig,
    *,
    fatal_integrity: bool = False,
    no_legal_proposal: bool = False,
) -> StopDecision:
    if fatal_integrity:
        return StopDecision(True, "fatal_integrity", "A fatal integrity condition was recorded.")
    if state.experiments_proposed >= config.max_experiments:
        return StopDecision(True, "experiment_budget", "The frozen experiment budget is exhausted.")
    if state.elapsed_wall_time_seconds >= config.wall_time_limit_seconds:
        return StopDecision(True, "wall_time_budget", "The frozen wall-clock budget is exhausted.")
    total_tokens = state.resource_totals.provider_tokens + state.resource_totals.estimated_tokens
    if config.token_limit is not None and total_tokens >= config.token_limit:
        return StopDecision(True, "token_budget", "The frozen LLM token budget is exhausted.")
    gpu_seconds = state.resource_totals.gpu_weighted_time_ms / 1000.0
    if config.gpu_seconds_limit is not None and gpu_seconds >= config.gpu_seconds_limit:
        return StopDecision(True, "gpu_budget", "The frozen GPU budget is exhausted.")
    pressure = convergence_pressure(events, config)
    if pressure >= config.convergence_patience:
        return StopDecision(
            True,
            "converged",
            "No trusted full evaluation improved the incumbent by more than epsilon "
            "for %d consecutive evaluations." % pressure,
        )
    if no_legal_proposal:
        return StopDecision(True, "no_legal_proposal", "No legal non-duplicate proposal remains.")
    return StopDecision(False)
