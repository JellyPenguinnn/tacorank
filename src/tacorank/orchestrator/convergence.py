"""Frozen deterministic stop and convergence rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..config import RunConfig
from ..orchestrator.state import RunState
from ..schemas import (
    Event,
    EventType,
    ExperimentDecisionKind,
    Fidelity,
    Integrity,
    Population,
    TrustVerdict,
)


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason_code: str = "continue"
    reason: str = "No deterministic stop condition matched."


FINALIZABLE_STOP_REASON_CODES = frozenset(
    {
        "experiment_budget",
        "wall_time_budget",
        "token_budget",
        "gpu_budget",
        "converged",
        "no_legal_proposal",
    }
)


def is_finalizable_stop_reason(reason_code: Optional[str]) -> bool:
    """Return whether a stop is a normal, frozen end-of-search condition."""

    return reason_code in FINALIZABLE_STOP_REASON_CODES


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
                and result.trust.integrity == Integrity.CLEAN
                and result.trust.verdict
                in {
                    TrustVerdict.ACCEPTED,
                    TrustVerdict.NEGATIVE,
                    TrustVerdict.INCONCLUSIVE,
                    TrustVerdict.REDUNDANT,
                }
            ):
                continue
            score = (
                result.trust.seed_mean
                if result.trust.seed_mean is not None
                else result.metric_set.primary_score
            )
            if decision.best_eligible and (
                incumbent is None or score > incumbent + config.convergence_epsilon
            ):
                incumbent = score
                non_improving = 0
            else:
                non_improving += 1
    return non_improving


def campaign_family_pressures(
    events: Sequence[Event], config: RunConfig
) -> dict[str, tuple[int, int]]:
    """Return clean terminal full counts and current pressure per campaign family."""

    campaign = config.research_campaign
    if campaign is None:
        return {}
    specifications = {
        event.payload.spec.experiment_id: event.payload.spec
        for event in events
        if event.event_type == EventType.EXPERIMENT_PROPOSED
        and event.payload.spec.campaign_id == campaign.campaign_id
    }
    evaluations = {
        event.event_id: event.payload.result
        for event in events
        if event.event_type == EventType.EVALUATION_COMPLETED
    }
    counts = {family: 0 for family in campaign.family_order}
    pressures = {family: 0 for family in campaign.family_order}
    for event in events:
        if event.event_type != EventType.EXPERIMENT_DECIDED:
            continue
        decision = event.payload.decision
        if decision.decision == ExperimentDecisionKind.PROMOTE:
            continue
        spec = specifications.get(decision.experiment_id)
        result = evaluations.get(decision.evaluation_event_id or "")
        if spec is None or result is None or not (
            result.fidelity == Fidelity.FULL
            and result.population == Population.PUBLIC_VALIDATION
            and result.trust.integrity == Integrity.CLEAN
            and result.trust.verdict
            in {
                TrustVerdict.ACCEPTED,
                TrustVerdict.NEGATIVE,
                TrustVerdict.INCONCLUSIVE,
                TrustVerdict.REDUNDANT,
            }
        ):
            continue
        family = spec.family
        if family not in counts:
            continue
        counts[family] += 1
        pressures[family] = 0 if decision.best_eligible else pressures[family] + 1
    return {family: (counts[family], pressures[family]) for family in counts}


def campaign_converged(events: Sequence[Event], config: RunConfig) -> bool:
    campaign = config.research_campaign
    if campaign is None:
        return False
    evidence = campaign_family_pressures(events, config)
    return all(
        evidence.get(family, (0, 0))[0]
        >= campaign.minimum_family_full_evaluations
        and evidence.get(family, (0, 0))[1]
        >= campaign.family_convergence_patience
        for family in campaign.family_order
    )


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
    if campaign_converged(events, config):
        return StopDecision(
            True,
            "campaign_converged",
            "Every campaign family reached its frozen minimum depth and "
            "non-improvement patience.",
        )
    # The official convergence contract applies to every run mode. Research
    # plans and legacy explicit campaigns may bound legal search, but cannot
    # turn the 50-iteration cap into a quota or bypass three-result patience.
    if pressure >= config.convergence_patience:
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
