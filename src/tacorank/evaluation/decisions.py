"""Deterministic proxy/full experiment decision policy."""

from dataclasses import dataclass
from typing import Optional, Sequence

from .types import (
    Decision,
    EvaluationResult,
    ExperimentDecision,
    Fidelity,
    Integrity,
    Population,
    Stability,
    Verdict,
)


class NoOpRecoveryRequired(RuntimeError):
    """A no-op must visit Person 4 before a terminal decision is emitted."""


@dataclass(frozen=True)
class DecisionContext:
    evaluation_event_id: str
    supporting_event_ids: Sequence[str]
    seed_evidence_event_ids: Sequence[str] = ()
    confirmations_remaining: int = 0
    promote_inconclusive_proxy: bool = True


def decide(
    result: EvaluationResult,
    context: DecisionContext,
) -> ExperimentDecision:
    trust = result.trust
    if tuple(context.seed_evidence_event_ids) != tuple(
        result.seed_evidence_event_ids
    ):
        raise ValueError("decision seed evidence does not match evaluation")
    if (
        result.population == Population.HIDDEN_FINAL
        or result.fidelity == Fidelity.FINAL
    ):
        raise ValueError("hidden-final results do not produce experiment decisions")
    if result.population == Population.UNBIASED_AUDIT:
        raise ValueError(
            "unbiased-audit results support trust but do not produce decisions"
        )
    if trust.verdict == Verdict.NO_OP:
        raise NoOpRecoveryRequired("no-op requires implementation recovery")

    if (
        result.fidelity == Fidelity.PROXY
        or result.population == Population.INTERNAL_PROXY
    ):
        if trust.verdict == Verdict.ACCEPTED and trust.integrity == Integrity.CLEAN:
            return _decision(
                result,
                context,
                Decision.PROMOTE,
                "PROXY_PASSED",
                False,
                False,
                Fidelity.FULL,
            )
        if (
            trust.verdict == Verdict.INCONCLUSIVE
            and trust.integrity == Integrity.CLEAN
            and "WITHIN_NOISE" in trust.flags
        ):
            return _decision(
                result,
                context,
                Decision.PROMOTE,
                "PROXY_WITHIN_NOISE",
                False,
                False,
                Fidelity.FULL,
            )
        reason = (
            "PROXY_FAILED"
            if trust.verdict == Verdict.NEGATIVE
            else "INTEGRITY_UNVERIFIED"
        )
        action = (
            Decision.PRUNE
            if trust.verdict == Verdict.NEGATIVE
            else Decision.INVALID
        )
        return _decision(result, context, action, reason, False, False, None)

    if (
        result.population != Population.PUBLIC_VALIDATION
        or result.fidelity != Fidelity.FULL
    ):
        raise ValueError("full experiment decisions require public validation")

    if trust.verdict == Verdict.ACCEPTED and trust.stability == Stability.SINGLE_SEED:
        if context.confirmations_remaining <= 0:
            return _decision(
                result,
                context,
                Decision.RETAIN,
                "CONFIRMATION_BUDGET_EXHAUSTED",
                False,
                False,
                None,
            )
        return _decision(
            result,
            context,
            Decision.PROMOTE,
            "CONFIRMATION_REQUIRED",
            False,
            False,
            Fidelity.FULL,
        )

    if (
        trust.verdict == Verdict.ACCEPTED
        and trust.stability == Stability.CONFIRMED
        and trust.integrity == Integrity.CLEAN
    ):
        if trust.best_delta_mean is None or trust.best_delta_ci_lower is None:
            current_best = (
                result.metric_set.primary_score - result.previous_best_delta.primary
            )
            candidate_primary = (
                trust.seed_mean
                if trust.seed_mean is not None
                else result.metric_set.primary_score
            )
            best_eligible = (
                candidate_primary - current_best > (trust.eta_applied or 0.0)
            )
        else:
            practical_gain = trust.minimum_practical_gain or 0.0
            best_eligible = bool(
                trust.best_delta_ci_lower > 0
                and trust.best_delta_mean > practical_gain
            )
        reason = "TRUSTED_IMPROVEMENT" if best_eligible else "TRUSTED_PARENT_ONLY"
        return _decision(
            result,
            context,
            Decision.ACCEPT,
            reason,
            True,
            best_eligible,
            None,
        )

    if (
        trust.verdict == Verdict.INCONCLUSIVE
        and trust.stability == Stability.CONFIRMED
        and trust.integrity == Integrity.CLEAN
        and "WITHIN_NOISE" in trust.flags
        and trust.seed_mean is not None
    ):
        current_best = (
            result.metric_set.primary_score - result.previous_best_delta.primary
        )
        eta = trust.eta_applied or 0.0
        if trust.seed_mean >= current_best - eta:
            return _decision(
                result,
                context,
                Decision.ACCEPT,
                "EXPLORATORY_PARENT_WITHIN_TOLERANCE",
                True,
                False,
                None,
            )

    reason_by_verdict = {
        Verdict.INCONCLUSIVE: (
            "SEED_VARIANCE_HIGH"
            if trust.stability == Stability.UNSTABLE
            else "WITHIN_NOISE"
        ),
        Verdict.NEGATIVE: "CLEAR_REGRESSION",
        Verdict.REDUNDANT: "SIGNAL_ALREADY_CAPTURED",
        Verdict.SUSPICIOUS: "INTEGRITY_UNVERIFIED",
    }
    reason = reason_by_verdict.get(trust.verdict, "INTEGRITY_UNVERIFIED")
    action = {
        Verdict.SUSPICIOUS: Decision.INVALID,
        Verdict.INCONCLUSIVE: Decision.RETAIN,
    }.get(trust.verdict, Decision.REJECT)
    return _decision(result, context, action, reason, False, False, None)


def _decision(
    result: EvaluationResult,
    context: DecisionContext,
    action: Decision,
    reason: str,
    parent_eligible: bool,
    best_eligible: bool,
    next_fidelity: Optional[Fidelity],
) -> ExperimentDecision:
    return ExperimentDecision(
        run_id=result.run_id,
        experiment_id=result.experiment_id,
        evaluation_event_id=context.evaluation_event_id,
        decision=action,
        reason_code=reason,
        fidelity_completed=result.fidelity,
        parent_eligible=parent_eligible,
        best_eligible=best_eligible,
        next_fidelity=next_fidelity,
        supporting_event_ids=tuple(context.supporting_event_ids),
        seed_evidence_event_ids=tuple(context.seed_evidence_event_ids),
    )
