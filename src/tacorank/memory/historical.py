"""Read-only, compatibility-filtered learning evidence from prior runs.

The current run's event ledger remains authoritative for current state.  This
module only reads completed ledgers and turns comparable public evaluations into
bounded observations for the planner and legal-choice ranker.  A malformed,
partial, hidden-label, or contract-incompatible run is ignored rather than
being allowed to influence search.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import (
    Event,
    EventType,
    Fidelity,
    Integrity,
    PlannerHistoricalSummary,
    Population,
    Stability,
    TrustVerdict,
)
from .event_store import EventStore, LedgerError


def _stable_score(result: Any) -> Optional[float]:
    trust = getattr(result, "trust", None)
    seed_mean = getattr(trust, "seed_mean", None)
    if seed_mean is not None:
        return float(seed_mean)
    metric_set = getattr(result, "metric_set", None)
    score = getattr(metric_set, "primary_score", None)
    return None if score is None else float(score)


def _diagnostic_metrics(result: Any) -> Dict[str, float]:
    raw = getattr(result, "diagnostic_metrics", None)
    if not isinstance(raw, dict):
        return {}
    values: Dict[str, float] = {}
    for name, value in raw.items():
        try:
            if value is not None and math.isfinite(float(value)):
                values[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _compatible_run(
    events: List[Event],
    *,
    contract_sha256: str,
    evaluator_sha256: str,
    baseline_score: Optional[float],
) -> bool:
    if not any(event.event_type == EventType.RUN_STOPPED for event in events):
        return False
    contract_events = [
        event
        for event in events
        if event.event_type == EventType.CONTRACT_VERIFIED
    ]
    baseline_events = [
        event
        for event in events
        if event.event_type == EventType.BASELINE_VERIFIED
    ]
    if not contract_events or not baseline_events:
        return False
    contract = contract_events[-1].payload
    baseline = baseline_events[-1].payload
    if (
        contract.contract_sha256 != contract_sha256
        or contract.evaluator_sha256 != evaluator_sha256
    ):
        return False
    prior_score = float(baseline.metric_set.primary_score)
    if baseline_score is None or not math.isclose(
        prior_score, float(baseline_score), rel_tol=0.0, abs_tol=1e-9
    ):
        return False
    # A baseline event is only useful here when its evaluation also agrees with
    # the frozen evaluator/contract recorded by the current run.
    evaluation = baseline.evaluation
    return (
        evaluation.contract_sha256 == contract_sha256
        and evaluation.evaluator_sha256 == evaluator_sha256
        and evaluation.population != Population.HIDDEN_FINAL
    )


def _run_observations(
    events: List[Event],
) -> List[PlannerHistoricalSummary]:
    specs = {
        event.payload.spec.experiment_id: event.payload.spec
        for event in events
        if event.event_type == EventType.EXPERIMENT_PROPOSED
    }
    output_by_experiment = {}
    for event in events:
        if event.event_type == EventType.OUTPUT_CHECKED:
            output_by_experiment[event.payload.result.experiment_id] = (
                event.payload.result
            )

    evaluations: Dict[str, Tuple[int, Any]] = {}
    for event in events:
        if event.event_type != EventType.EVALUATION_COMPLETED:
            continue
        result = event.payload.result
        if (
            result.population == Population.PUBLIC_VALIDATION
            and result.fidelity == Fidelity.FULL
        ):
            evaluations[result.experiment_id] = (event.seq, result)

    baseline = next(
        event.payload
        for event in reversed(events)
        if event.event_type == EventType.BASELINE_VERIFIED
    )
    scores: Dict[str, float] = {
        baseline.experiment_id: float(baseline.metric_set.primary_score)
    }
    for _, result in evaluations.values():
        score = _stable_score(result)
        if score is not None:
            scores[result.experiment_id] = score

    observations: List[PlannerHistoricalSummary] = []
    for _, result in sorted(evaluations.values(), reverse=True):
        spec = specs.get(result.experiment_id)
        output = output_by_experiment.get(result.experiment_id)
        trust = result.trust
        if spec is None or output is None or not output.accepted:
            continue
        if trust.integrity != Integrity.CLEAN:
            continue
        if trust.verdict in {
            TrustVerdict.SUSPICIOUS,
            TrustVerdict.REDUNDANT,
        }:
            continue
        child_score = scores.get(result.experiment_id)
        parent_id = spec.parent_experiment_id
        parent_score = scores.get(parent_id)
        reward = None
        if child_score is not None and parent_score is not None:
            reward = child_score - parent_score
        elif result.parent_delta is not None:
            reward = float(result.parent_delta)
        risk_adjusted = None if reward is None else reward
        if risk_adjusted is not None and trust.seed_stderr is not None:
            risk_adjusted -= 2.0 * float(trust.seed_stderr)
        observations.append(
            PlannerHistoricalSummary(
                source_run_id=events[0].run_id,
                experiment_id=result.experiment_id,
                parent_experiment_id=parent_id,
                family=spec.family,
                method_card_ids=list(spec.method_card_ids),
                stable_primary_score=child_score,
                parent_stable_primary_score=parent_score,
                reward=reward,
                risk_adjusted_reward=risk_adjusted,
                seed_count=trust.seed_count,
                seed_stderr=trust.seed_stderr,
                stability=trust.stability,
                trust_verdict=trust.verdict,
                integrity=trust.integrity,
                trust_flags=list(trust.flags),
                diagnostic_metrics=_diagnostic_metrics(result),
            )
        )
    return observations


def load_historical_feedback(
    *,
    repository_root: Path,
    current_run_id: str,
    contract_sha256: str,
    evaluator_sha256: str,
    baseline_score: Optional[float],
    max_runs: int = 8,
    max_observations: int = 32,
) -> List[PlannerHistoricalSummary]:
    """Load bounded comparable observations without mutating any ledger."""

    if max_runs <= 0 or max_observations <= 0:
        return []
    runs_root = repository_root / "runs"
    if not runs_root.is_dir():
        return []

    observations: List[PlannerHistoricalSummary] = []
    compatible_runs = 0
    for ledger_path in sorted(runs_root.glob("*/events.jsonl"), reverse=True):
        try:
            events = EventStore(ledger_path).read_events()
        except (OSError, LedgerError, ValueError):
            continue
        if not events or events[0].run_id == current_run_id:
            continue
        if not _compatible_run(
            events,
            contract_sha256=contract_sha256,
            evaluator_sha256=evaluator_sha256,
            baseline_score=baseline_score,
        ):
            continue
        compatible_runs += 1
        observations.extend(_run_observations(events))
        if compatible_runs >= max_runs:
            break
        if len(observations) >= max_observations:
            break
    return observations[:max_observations]
