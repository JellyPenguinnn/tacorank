"""Gated evaluation projection intended for the research planner."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .types import EvaluationResult, ExperimentDecision


@dataclass(frozen=True)
class EvaluationProjection:
    experiment_id: str
    parent_experiment_id: Optional[str]
    family: str
    kind: str
    fidelity_completed: str
    trusted_best_primary: float
    trusted_best_experiment_id: str
    node_moved_best: bool
    headroom_captured_pct: float
    trust_verdict: str
    trust_integrity: str
    trust_stability: str
    decision: str
    reason_code: str
    parent_eligible: bool
    best_eligible: bool
    next_fidelity: Optional[str]
    delta_band: str
    metric_split: str
    slice_digest: str
    orthogonality: float
    holdout_signal: str
    convergence_pressure: int
    scheduling_hint: str
    public_query_index: int
    full_evaluations_remaining: int
    stale_lesson_ids: Tuple[str, ...]


def build_projection(
    result: EvaluationResult,
    decision: ExperimentDecision,
    parent_experiment_id: Optional[str],
    family: str,
    kind: str,
    trusted_best_primary: float,
    trusted_best_experiment_id: str,
    baseline_primary: float,
    oracle_primary: float,
    orthogonality: float,
    val_a_val_b_gap: Optional[float],
    convergence_pressure: int,
    full_evaluations_remaining: int,
    slice_deltas: Optional[Mapping[str, float]] = None,
    stale_lesson_ids: Sequence[str] = (),
) -> EvaluationProjection:
    eta = result.trust.eta_applied or 0.0
    delta = result.parent_delta.primary
    if delta > eta:
        delta_band = "better"
    elif abs(delta) <= eta:
        delta_band = "within_noise"
    elif delta <= -0.01:
        delta_band = "much_worse"
    else:
        delta_band = "worse"
    metric_items = list(result.parent_delta.metrics.items())
    signs = [value > 0 for _, value in metric_items]
    if signs and all(signs):
        metric_split = "both_up"
    elif signs and not any(signs):
        metric_split = "both_down"
    else:
        names = [name.lower() for name, _ in metric_items]
        if len(names) == 2 and any("gauc" in name for name in names):
            gauc_index = next(index for index, name in enumerate(names) if "gauc" in name)
            metric_split = "gauc_only" if signs[gauc_index] else "ndcg_only"
        else:
            metric_split = "mixed"
    gap = abs(val_a_val_b_gap) if val_a_val_b_gap is not None else 0.0
    holdout_signal = "normal" if gap < 0.006 else "widening" if gap < 0.012 else "alert"
    digest = _slice_digest(slice_deltas or {})
    if oracle_primary <= baseline_primary:
        raise ValueError("oracle primary must exceed baseline primary")
    headroom = 100.0 * (trusted_best_primary - baseline_primary) / (oracle_primary - baseline_primary)
    return EvaluationProjection(
        experiment_id=result.experiment_id,
        parent_experiment_id=parent_experiment_id,
        family=family,
        kind=kind,
        fidelity_completed=result.fidelity.value,
        trusted_best_primary=trusted_best_primary,
        trusted_best_experiment_id=trusted_best_experiment_id,
        node_moved_best=decision.best_eligible,
        headroom_captured_pct=headroom,
        trust_verdict=result.trust.verdict.value,
        trust_integrity=result.trust.integrity.value,
        trust_stability=result.trust.stability.value,
        decision=decision.decision.value,
        reason_code=decision.reason_code,
        parent_eligible=decision.parent_eligible,
        best_eligible=decision.best_eligible,
        next_fidelity=decision.next_fidelity.value if decision.next_fidelity else None,
        delta_band=delta_band,
        metric_split=metric_split,
        slice_digest=digest,
        orthogonality=max(0.0, min(1.0, float(orthogonality))),
        holdout_signal=holdout_signal,
        convergence_pressure=convergence_pressure,
        scheduling_hint=("force_high_variance" if convergence_pressure == 2 else "free"),
        public_query_index=result.public_query_index or 0,
        full_evaluations_remaining=full_evaluations_remaining,
        stale_lesson_ids=tuple(stale_lesson_ids),
    )


def _slice_digest(deltas: Mapping[str, float]) -> str:
    largest = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)[:2]
    return "; ".join("%s %+.4f" % item for item in largest)[:120]
