"""Ordered, deterministic trust adjudication."""

from dataclasses import dataclass, field
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
    eta_floor: float = 0.0016
    # Two seeds decide stability: observed candidate seed spread (~1e-4) sits
    # far below the eta floor (1.6e-3), so a third full retrain buys almost no
    # information while costing a full evaluation of wall clock.
    required_seed_count: int = 2
    baseline_seed_std: float = 0.0008
    unstable_std_multiplier: float = 3.0
    too_good_delta: float = 0.05
    minimum_unique_score_fraction: float = 0.01
    redundancy_correlation: float = 0.70
    gain_concentration_threshold: float = 0.70
    drift_slope_threshold: float = 0.002
    validation_arm_gap_threshold: float = 0.006
    # Proxy deltas below this magnitude are not reliable enough to claim an
    # improvement. The decision layer promotes only the positive side of this
    # noise band and cleanly prunes zero or negative deltas to bound resources.
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
    output_gate_evidence: bool = True
    evaluator_hash_matches: bool = True
    contract_hash_matches: bool = True
    forbidden_inputs: Tuple[str, ...] = ()
    alignment_suspect: bool = False
    internal_proxy_delta: Optional[float] = None
    unbiased_audit_delta: Optional[float] = None
    val_a_delta: Optional[float] = None
    val_b_delta: Optional[float] = None
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
    if "CONTRACT_DIRECTION_POLICY_FAILED" in directional_flags:
        return _assessment(
            Verdict.INCONCLUSIVE, Stability.NOT_APPLICABLE,
            Integrity.CLEAN, directional_flags, aggregate
        )
    if evidence.population == Population.INTERNAL_PROXY or evidence.fidelity == Fidelity.PROXY:
        proxy_noise_tolerance = cfg.proxy_improvement_threshold
        if evidence.parent_delta > proxy_noise_tolerance:
            return _assessment(
                Verdict.ACCEPTED, Stability.NOT_APPLICABLE,
                Integrity.CLEAN, directional_flags, aggregate
            )
        if evidence.parent_delta < -proxy_noise_tolerance:
            return _assessment(
                Verdict.NEGATIVE, Stability.NOT_APPLICABLE,
                Integrity.CLEAN, directional_flags, aggregate
            )
        return _assessment(
            Verdict.INCONCLUSIVE, Stability.NOT_APPLICABLE,
            Integrity.CLEAN, ["WITHIN_NOISE"] + directional_flags, aggregate
        )

    # Two different seeds with identical scores prove the candidate is
    # seed-independent; a further identical rerun adds no evidence and costs
    # one full evaluation, so a deterministic pair counts as confirmed.
    deterministic_pair = (
        aggregate.count >= 2 and aggregate.standard_deviation == 0.0
    )
    if aggregate.count < cfg.required_seed_count and not deterministic_pair:
        return _assessment(
            Verdict.ACCEPTED, Stability.SINGLE_SEED,
            Integrity.CLEAN, directional_flags, aggregate
        )
    if aggregate.standard_deviation > cfg.unstable_std_multiplier * cfg.baseline_seed_std:
        return _assessment(
            Verdict.INCONCLUSIVE, Stability.UNSTABLE,
            Integrity.CLEAN, ["SEED_INSTABILITY"] + directional_flags, aggregate
        )
    aggregate_delta = aggregate.mean - evidence.parent_primary
    if abs(aggregate_delta) <= aggregate.eta:
        directional_threshold = max(2.0 * aggregate.standard_error, 1e-12)
        if aggregate_delta > directional_threshold:
            return _assessment(
                Verdict.ACCEPTED,
                Stability.CONFIRMED,
                Integrity.CLEAN,
                ["CONFIRMED_POSITIVE_BELOW_LADDER"] + directional_flags,
                aggregate,
            )
        return _assessment(
            Verdict.INCONCLUSIVE, Stability.CONFIRMED,
            Integrity.CLEAN, ["WITHIN_NOISE"] + directional_flags, aggregate
        )
    if aggregate_delta < -aggregate.eta:
        return _assessment(
            Verdict.NEGATIVE, Stability.CONFIRMED,
            Integrity.CLEAN, directional_flags, aggregate
        )
    return _assessment(
        Verdict.ACCEPTED, Stability.CONFIRMED,
        Integrity.CLEAN, directional_flags, aggregate
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
) -> TrustAssessment:
    has_confirmed_aggregate = aggregate.count >= 3
    return TrustAssessment(
        verdict=verdict,
        stability=stability,
        integrity=integrity,
        flags=tuple(flags),
        eta_applied=aggregate.eta,
        seed_mean=aggregate.mean if has_confirmed_aggregate else None,
        seed_stderr=(aggregate.standard_error if has_confirmed_aggregate else None),
        seed_count=aggregate.count,
    )
