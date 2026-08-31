"""Final candidate filtering and rank averaging."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .types import Integrity, Stability, Verdict


@dataclass(frozen=True)
class CandidateEvidence:
    experiment_id: str
    commit_sha: str
    public_score: float
    val_b_score: Optional[float]
    trust_verdict: Verdict
    stability: Stability
    integrity: Integrity
    internal_holdout_agrees: bool
    unbiased_audit_agrees: bool
    clean_reproduction_passed: bool
    seed_predictions: Tuple[Sequence[float], ...] = ()


def select_final(candidates: Sequence[CandidateEvidence]) -> CandidateEvidence:
    eligible = [candidate for candidate in candidates if _eligible(candidate)]
    if not eligible:
        raise ValueError("no trusted final candidate")
    return max(
        eligible,
        key=lambda candidate: (
            candidate.val_b_score
            if candidate.val_b_score is not None
            else candidate.public_score,
            candidate.public_score,
            candidate.experiment_id,
        ),
    )


def rank_finalists(candidates: Sequence[CandidateEvidence]) -> Tuple[CandidateEvidence, ...]:
    """Rank trusted selection candidates before clean reproduction."""

    eligible = [
        candidate
        for candidate in candidates
        if candidate.trust_verdict == Verdict.ACCEPTED
        and candidate.stability == Stability.CONFIRMED
        and candidate.integrity == Integrity.CLEAN
        and candidate.internal_holdout_agrees
        and candidate.unbiased_audit_agrees
        and candidate.val_b_score is not None
    ]
    return tuple(
        sorted(
            eligible,
            key=lambda candidate: (
                candidate.val_b_score,
                candidate.public_score,
                candidate.experiment_id,
            ),
            reverse=True,
        )
    )


def rank_average(predictions: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    if not predictions:
        raise ValueError("at least one prediction vector is required")
    length = len(predictions[0])
    if length == 0 or any(len(vector) != length for vector in predictions):
        raise ValueError("prediction vectors must be non-empty and aligned")
    rank_vectors = [_normalized_average_ranks(vector) for vector in predictions]
    return tuple(
        sum(vector[index] for vector in rank_vectors) / len(rank_vectors)
        for index in range(length)
    )


def _eligible(candidate: CandidateEvidence) -> bool:
    return (
        candidate.trust_verdict == Verdict.ACCEPTED
        and candidate.stability == Stability.CONFIRMED
        and candidate.integrity == Integrity.CLEAN
        and candidate.internal_holdout_agrees
        and candidate.unbiased_audit_agrees
        and candidate.clean_reproduction_passed
        and candidate.val_b_score is not None
    )


def _normalized_average_ranks(values: Sequence[float]) -> Tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        average = (index + end) / 2.0
        for position in range(index, end + 1):
            ranks[indexed[position][0]] = average
        index = end + 1
    denominator = max(1, len(values) - 1)
    return tuple(rank / denominator for rank in ranks)
