"""Deterministic eligibility rules for research lessons."""

from tacorank.evaluation.types import Fidelity, Population, Stability, Verdict


def lesson_allowed(
    verdict: Verdict,
    fidelity: Fidelity,
    population: Population,
    stability: Stability,
) -> bool:
    if verdict in (Verdict.NO_OP, Verdict.INCONCLUSIVE):
        return False
    if population == Population.HIDDEN_FINAL or fidelity == Fidelity.FINAL:
        return False
    if verdict == Verdict.NEGATIVE:
        return fidelity == Fidelity.FULL and population == Population.PUBLIC_VALIDATION
    if verdict == Verdict.ACCEPTED:
        return stability == Stability.CONFIRMED
    return verdict in (Verdict.REDUNDANT, Verdict.SUSPICIOUS)
