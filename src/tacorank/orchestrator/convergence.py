"""Frozen deterministic stop and convergence rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import RunConfig
from ..orchestrator.state import RunState
from ..schemas import (
    Event,
    EventType,
    ExperimentDecisionKind,
    Fidelity,
    Population,
    TrustVerdict,
)


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason_code: str = "continue"
    reason: str = "No deterministic stop condition matched."


def convergence_pressure(events: Sequence[Event], config: RunConfig) -> int:
    """Count terminal research iterations without epsilon improvement.

    Full-fidelity confirmation seeds belong to one experiment iteration.  They
    must establish trust, but must not independently consume convergence
    patience.  A terminal full decision is therefore the unit counted here.
    """

    incumbent = None
    non_improving = 0
    evaluations = {}
    for event in events:
        if event.event_type == EventType.BASELINE_VERIFIED:
            incumbent = event.payload.metric_set.primary_score
        elif event.event_type == EventType.EVALUATION_COMPLETED:
            result = event.payload.result
            evaluations[event.event_id] = result
        elif event.event_type == EventType.EXPERIMENT_DECIDED:
            decision = event.payload.decision
            if (
                decision.decision == ExperimentDecisionKind.PROMOTE
                or decision.fidelity_completed != Fidelity.FULL
                or decision.evaluation_event_id is None
            ):
                continue
            result = evaluations.get(decision.evaluation_event_id)
            if result is None or not (
                result.fidelity == Fidelity.FULL
                and result.population == Population.PUBLIC_VALIDATION
                and result.trust.verdict == TrustVerdict.ACCEPTED
            ):
                continue
            score = (
                result.trust.seed_mean
                if result.trust.seed_mean is not None
                else result.metric_set.primary_score
            )
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
    runtime_stop = runtime_budget_decision(state, config)
    if runtime_stop.stop:
        return runtime_stop
    pressure = convergence_pressure(events, config)
    # A frozen depth campaign owns its per-family conclusion budget. Global
    # patience must not terminate it after the first few noisy configurations.
    if config.research_campaign is None and pressure >= config.convergence_patience:
        return StopDecision(
            True,
            "converged",
            "No trusted terminal full-fidelity iteration improved the incumbent by "
            "more than epsilon for %d consecutive iterations." % pressure,
        )
    if no_legal_proposal:
        return StopDecision(True, "no_legal_proposal", "No legal non-duplicate proposal remains.")
    return StopDecision(False)


def runtime_budget_decision(state: RunState, config: RunConfig) -> StopDecision:
    """Check budgets that can be exhausted by any adapter action."""

    if state.elapsed_wall_time_seconds >= config.wall_time_limit_seconds:
        return StopDecision(True, "wall_time_budget", "The frozen wall-clock budget is exhausted.")
    # token_measurement describes provenance, not whether reported tokens count
    # against the frozen ceiling. Excluding "none" would let callers bypass the
    # stop rule simply by changing the measurement label.
    total_tokens = state.resource_totals.total_reported_tokens
    if config.token_limit is not None and total_tokens >= config.token_limit:
        return StopDecision(True, "token_budget", "The frozen LLM token budget is exhausted.")
    gpu_seconds = state.resource_totals.gpu_weighted_time_ms / 1000.0
    if config.gpu_seconds_limit is not None and gpu_seconds >= config.gpu_seconds_limit:
        return StopDecision(True, "gpu_budget", "The frozen GPU budget is exhausted.")
    return StopDecision(False)
