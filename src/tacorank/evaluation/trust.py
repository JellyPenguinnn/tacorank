"""Ordered, deterministic trust adjudication."""

from dataclasses import dataclass, field
import math
import statistics
from typing import Mapping, Optional, Sequence, Tuple

from .no_op import NoOpConfig, is_no_op
from .stability import aggregate_seeds
from .types import (
    Fidelity,
    Integrity,
    Population,
    PredictionChange,
    Stability,
    TrustAssessment,
    Verdict,
)


@dataclass(frozen=True)
class TrustConfig:
    # Statistical uncertainty is estimated from paired deltas. This value is a
    # practical-effect threshold, not an uncertainty estimate.
    eta_floor: float = 0.0016
    minimum_practical_gain: float = 0.0001
    required_seed_count: int = 3
    baseline_seed_std: float = 0.0008
    unstable_std_multiplier: float = 3.0
    too_good_delta: float = 0.05
    minimum_unique_score_fraction: float = 0.01
    redundancy_correlation: float = 0.70
    gain_concentration_threshold: float = 0.70
    drift_slope_threshold: float = 0.002
    validation_arm_gap_threshold: float = 0.006
    # Proxy deltas below this magnitude are not reliable enough to prune or to
    # claim an improvement.  They receive one bounded full-fidelity check.
    proxy_improvement_threshold: float = 0.0016
    require_non_decreasing_metrics: bool = False
    no_op: NoOpConfig = field(default_factory=NoOpConfig)


@dataclass(frozen=True)
class TrustEvidence:
    population: Population
    fidelity: Fidelity
    parent_primary: float
    parent_delta: float
    metric_deltas: Mapping[str, float]
    prediction_change: PredictionChange
    seed_scores: Sequence[float]
    seed_parent_deltas: Sequence[float] = ()
    seed_best_deltas: Sequence[float] = ()
    paired_parent_delta_stderr: Optional[float] = None
    paired_parent_delta_ci_lower: Optional[float] = None
    paired_parent_delta_ci_upper: Optional[float] = None
    paired_best_delta_stderr: Optional[float] = None
    paired_best_delta_ci_lower: Optional[float] = None
    paired_best_delta_ci_upper: Optional[float] = None
    output_gate_evidence: bool = True
    evaluator_hash_matches: bool = True
    contract_hash_matches: bool = True
    forbidden_inputs: Tuple[str, ...] = ()
    alignment_suspect: bool = False
    internal_proxy_delta: Optional[float] = None
    internal_proxy_ci_lower: Optional[float] = None
    internal_proxy_ci_upper: Optional[float] = None
    unbiased_audit_delta: Optional[float] = None
    val_a_delta: Optional[float] = None
    val_a_delta_ci_lower: Optional[float] = None
    val_a_delta_ci_upper: Optional[float] = None
    val_b_delta: Optional[float] = None
    val_b_delta_ci_lower: Optional[float] = None
    val_b_delta_ci_upper: Optional[float] = None
    delta_correlation: Optional[float] = None
    delta_correlation_experiment_id: Optional[str] = None
    score_unique_fraction: Optional[float] = None
    gain_concentration_top10pct: Optional[float] = None
    drift_primary_slope: Optional[float] = None


def assess_trust(
    evidence: TrustEvidence, config: Optional[TrustConfig] = None
) -> TrustAssessment:
    cfg = config or TrustConfig()
    aggregate = aggregate_seeds(evidence.seed_scores, cfg.eta_floor)
    parent_deltas = evidence.seed_parent_deltas or (
        (evidence.parent_delta,)
        if evidence.population == Population.INTERNAL_PROXY
        or evidence.fidelity == Fidelity.PROXY
        else tuple(
            float(score) - evidence.parent_primary for score in evidence.seed_scores
        )
    )
    parent_interval = _delta_interval(
        parent_deltas,
        paired_stderr=evidence.paired_parent_delta_stderr,
        paired_lower=evidence.paired_parent_delta_ci_lower,
        paired_upper=evidence.paired_parent_delta_ci_upper,
    )
    best_interval = _delta_interval(
        evidence.seed_best_deltas,
        paired_stderr=evidence.paired_best_delta_stderr,
        paired_lower=evidence.paired_best_delta_ci_lower,
        paired_upper=evidence.paired_best_delta_ci_upper,
    )

    hard_flags = []
    if not evidence.evaluator_hash_matches:
        hard_flags.append("EVALUATOR_HASH_MISMATCH")
    if not evidence.contract_hash_matches:
        hard_flags.append("CONTRACT_HASH_MISMATCH")
    if not evidence.output_gate_evidence:
        hard_flags.append("OUTPUT_GATE_EVIDENCE_MISSING")
    hard_flags.extend(
        "FORBIDDEN_INPUT_DETECTED:%s" % name for name in evidence.forbidden_inputs
    )
    if hard_flags:
        return _assessment(
            Verdict.SUSPICIOUS, Stability.NOT_APPLICABLE,
            Integrity.COMPROMISED, hard_flags, aggregate
        )

    if evidence.alignment_suspect:
        return _assessment(
            Verdict.SUSPICIOUS, Stability.NOT_APPLICABLE,
            Integrity.INCONCLUSIVE, ["PREDICTION_ALIGNMENT_SUSPECT"], aggregate
        )

    if is_no_op(evidence.prediction_change, evidence.parent_delta, cfg.no_op):
        return _assessment(
            Verdict.NO_OP, Stability.NOT_APPLICABLE,
            Integrity.CLEAN, ["NO_PREDICTION_CHANGE"], aggregate
        )

    unique_fraction = (
        evidence.score_unique_fraction
        if evidence.score_unique_fraction is not None
        else evidence.prediction_change.unique_score_fraction
    )
    if unique_fraction < cfg.minimum_unique_score_fraction:
        return _assessment(
            Verdict.SUSPICIOUS, Stability.NOT_APPLICABLE,
            Integrity.COMPROMISED, ["DEGENERATE_SCORES"], aggregate
        )

    if evidence.parent_delta > cfg.too_good_delta:
        return _assessment(
            Verdict.SUSPICIOUS, Stability.NOT_APPLICABLE,
            Integrity.INCONCLUSIVE, ["TOO_GOOD_TO_BE_TRUE"], aggregate
        )
    proxy_full_conflict = _proxy_full_conflict(evidence)
    if proxy_full_conflict:
        return _assessment(
            Verdict.SUSPICIOUS, Stability.NOT_APPLICABLE,
            Integrity.INCONCLUSIVE, ["PROXY_FULL_SIGN_CONFLICT"], aggregate,
            parent_interval=parent_interval,
            best_interval=best_interval,
            minimum_practical_gain=cfg.minimum_practical_gain,
        )
    if (
        evidence.parent_delta > 0
        and evidence.unbiased_audit_delta is not None
        and evidence.unbiased_audit_delta <= 0
    ):
        return _assessment(
            Verdict.SUSPICIOUS, Stability.NOT_APPLICABLE,
            Integrity.INCONCLUSIVE, ["UNBIASED_AUDIT_SIGN_CONFLICT"], aggregate
        )
    if (
        evidence.delta_correlation is not None
        and evidence.delta_correlation > cfg.redundancy_correlation
    ):
        suffix = ""
        if evidence.delta_correlation_experiment_id:
            suffix = ":%s" % evidence.delta_correlation_experiment_id
        return _assessment(
            Verdict.REDUNDANT, Stability.NOT_APPLICABLE,
            Integrity.CLEAN,
            ["DELTA_VECTOR_REDUNDANT:%.4f%s" % (evidence.delta_correlation, suffix)],
            aggregate,
        )

    directional_flags = _directional_flags(evidence, cfg)
    if _raw_proxy_full_sign_conflict(evidence) and not proxy_full_conflict:
        directional_flags.append("PROXY_FULL_WITHIN_UNCERTAINTY")
    if "CONTRACT_DIRECTION_POLICY_FAILED" in directional_flags:
        return _assessment(
            Verdict.INCONCLUSIVE, Stability.NOT_APPLICABLE,
            Integrity.CLEAN, directional_flags, aggregate
        )
    if evidence.population == Population.INTERNAL_PROXY or evidence.fidelity == Fidelity.PROXY:
        proxy_floor = max(cfg.proxy_improvement_threshold, cfg.minimum_practical_gain)
        proxy_lower = parent_interval[2]
        proxy_upper = parent_interval[3]
        if proxy_lower > 0 and evidence.parent_delta > proxy_floor:
            return _assessment(
                Verdict.ACCEPTED, Stability.NOT_APPLICABLE, Integrity.CLEAN,
                directional_flags, aggregate, parent_interval=parent_interval,
                best_interval=best_interval,
                minimum_practical_gain=cfg.minimum_practical_gain,
            )
        if proxy_upper < 0 and evidence.parent_delta < -proxy_floor:
            return _assessment(
                Verdict.NEGATIVE, Stability.NOT_APPLICABLE, Integrity.CLEAN,
                directional_flags, aggregate, parent_interval=parent_interval,
                best_interval=best_interval,
                minimum_practical_gain=cfg.minimum_practical_gain,
            )
        return _assessment(
            Verdict.INCONCLUSIVE, Stability.NOT_APPLICABLE, Integrity.CLEAN,
            ["WITHIN_NOISE"] + directional_flags, aggregate,
            parent_interval=parent_interval, best_interval=best_interval,
            minimum_practical_gain=cfg.minimum_practical_gain,
        )

    if aggregate.count < cfg.required_seed_count:
        return _assessment(
            Verdict.ACCEPTED, Stability.SINGLE_SEED,
            Integrity.CLEAN, directional_flags, aggregate,
            parent_interval=parent_interval, best_interval=best_interval,
            minimum_practical_gain=cfg.minimum_practical_gain,
        )
    if aggregate.standard_deviation > cfg.unstable_std_multiplier * cfg.baseline_seed_std:
        return _assessment(
            Verdict.INCONCLUSIVE, Stability.UNSTABLE,
            Integrity.CLEAN, ["SEED_INSTABILITY"] + directional_flags, aggregate
        )
    aggregate_delta, _, lower, upper = parent_interval
    if lower > 0:
        flags = list(directional_flags)
        if aggregate_delta <= cfg.minimum_practical_gain:
            flags.insert(0, "CONFIRMED_POSITIVE_BELOW_PRACTICAL_GAIN")
        return _assessment(
            Verdict.ACCEPTED, Stability.CONFIRMED, Integrity.CLEAN, flags,
            aggregate, parent_interval=parent_interval,
            best_interval=best_interval,
            minimum_practical_gain=cfg.minimum_practical_gain,
        )
    if upper < 0:
        return _assessment(
            Verdict.NEGATIVE, Stability.CONFIRMED, Integrity.CLEAN,
            directional_flags, aggregate, parent_interval=parent_interval,
            best_interval=best_interval,
            minimum_practical_gain=cfg.minimum_practical_gain,
        )
    return _assessment(
        Verdict.INCONCLUSIVE, Stability.CONFIRMED, Integrity.CLEAN,
        ["WITHIN_NOISE"] + directional_flags, aggregate,
        parent_interval=parent_interval, best_interval=best_interval,
        minimum_practical_gain=cfg.minimum_practical_gain,
    )


def _directional_flags(evidence: TrustEvidence, config: TrustConfig) -> list:
    flags = []
    if (
        evidence.internal_proxy_delta is not None
        and evidence.parent_delta * evidence.internal_proxy_delta < 0
    ):
        # Proxy and full use different samples.  A sign change is useful
        # generalization evidence, but is not an integrity failure by itself.
        flags.append("PROXY_FULL_DIRECTION_CONFLICT")
    deltas = list(evidence.metric_deltas.values())
    if deltas and min(deltas) < 0 < max(deltas):
        flags.append("METRIC_DIRECTION_CONFLICT")
        if config.require_non_decreasing_metrics:
            flags.append("CONTRACT_DIRECTION_POLICY_FAILED")
    if (
        evidence.gain_concentration_top10pct is not None
        and evidence.gain_concentration_top10pct > config.gain_concentration_threshold
    ):
        flags.append("FRAGILE_CONCENTRATED_GAIN")
    if (
        evidence.drift_primary_slope is not None
        and abs(evidence.drift_primary_slope) > config.drift_slope_threshold
    ):
        flags.append("DRIFT_DETECTED")
    if (
        evidence.val_a_delta is not None
        and evidence.val_b_delta is not None
    ):
        if evidence.val_a_delta * evidence.val_b_delta <= 0:
            flags.append("VALIDATION_ARM_SIGN_CONFLICT")
        if (
            abs(evidence.val_a_delta - evidence.val_b_delta)
            > config.validation_arm_gap_threshold
        ):
            flags.append("VALIDATION_ARM_GAP")
    return flags


def _assessment(
    verdict: Verdict,
    stability: Stability,
    integrity: Integrity,
    flags: Sequence[str],
    aggregate,
    *,
    parent_interval=None,
    best_interval=None,
    minimum_practical_gain=None,
) -> TrustAssessment:
    has_confirmed_aggregate = aggregate.count >= 3
    parent_interval = parent_interval or (None, None, None, None)
    best_interval = best_interval or (None, None, None, None)
    interval_half_width = (
        None
        if parent_interval[0] is None
        else max(
            parent_interval[0] - parent_interval[2],
            parent_interval[3] - parent_interval[0],
        )
    )
    return TrustAssessment(
        verdict=verdict,
        stability=stability,
        integrity=integrity,
        flags=tuple(flags),
        eta_applied=(
            aggregate.eta
            if interval_half_width is None
            else max(float(minimum_practical_gain or 0.0), interval_half_width)
        ),
        seed_mean=aggregate.mean if has_confirmed_aggregate else None,
        seed_stderr=(aggregate.standard_error if has_confirmed_aggregate else None),
        seed_count=aggregate.count,
        parent_delta_mean=parent_interval[0],
        parent_delta_stderr=parent_interval[1],
        parent_delta_ci_lower=parent_interval[2],
        parent_delta_ci_upper=parent_interval[3],
        best_delta_mean=best_interval[0],
        best_delta_stderr=best_interval[1],
        best_delta_ci_lower=best_interval[2],
        best_delta_ci_upper=best_interval[3],
        minimum_practical_gain=minimum_practical_gain,
    )


def _delta_interval(
    deltas: Sequence[float],
    *,
    paired_stderr: Optional[float],
    paired_lower: Optional[float],
    paired_upper: Optional[float],
) -> tuple:
    values = tuple(float(value) for value in deltas)
    if not values:
        return (None, None, None, None)
    mean = statistics.mean(values)
    seed_stderr = (
        statistics.stdev(values) / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    sampling_stderr = float(paired_stderr or 0.0)
    standard_error = max(seed_stderr, sampling_stderr)
    critical = _student_t_critical(len(values) - 1) if len(values) > 1 else 1.96
    half_width = critical * standard_error
    lower = mean - half_width
    upper = mean + half_width
    if len(values) == 1 and paired_lower is not None and paired_upper is not None:
        lower = float(paired_lower)
        upper = float(paired_upper)
        half_width = max(mean - lower, upper - mean)
        standard_error = max(standard_error, half_width / 1.96)
        lower = min(lower, mean)
        upper = max(upper, mean)
    return (mean, standard_error, lower, upper)


def _student_t_critical(degrees_of_freedom: int) -> float:
    values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    return values.get(degrees_of_freedom, 1.96 if degrees_of_freedom >= 30 else 2.262)


def _raw_proxy_full_sign_conflict(evidence: TrustEvidence) -> bool:
    full_delta = (
        evidence.val_a_delta
        if evidence.val_a_delta is not None
        else evidence.parent_delta
    )
    return (
        evidence.internal_proxy_delta is not None
        and full_delta != 0
        and evidence.internal_proxy_delta * full_delta < 0
    )


def _proxy_full_conflict(evidence: TrustEvidence) -> bool:
    if not _raw_proxy_full_sign_conflict(evidence):
        return False
    if None in (
        evidence.internal_proxy_ci_lower,
        evidence.internal_proxy_ci_upper,
        evidence.val_a_delta_ci_lower,
        evidence.val_a_delta_ci_upper,
    ):
        return False
    return bool(
        (
            evidence.internal_proxy_ci_upper < 0
            and evidence.val_a_delta_ci_lower > 0
        )
        or (
            evidence.internal_proxy_ci_lower > 0
            and evidence.val_a_delta_ci_upper < 0
        )
    )
