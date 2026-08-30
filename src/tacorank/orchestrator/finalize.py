"""Pure evidence selection for sealed post-search finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..schemas import Event, EventType, Fidelity, Population, TrustVerdict


class FinalizationError(RuntimeError):
    """The ledger does not contain sufficient trusted finalization evidence."""


@dataclass(frozen=True)
class CandidateFinalizationPlan:
    experiment_id: str
    commit_sha: str
    patch_receipt_id: str
    seed: int
    best_primary_score: float
    next_attempt: int


def candidate_finalization_plan(
    events: Sequence[Event], state
) -> CandidateFinalizationPlan:
    """Resolve the exact selected candidate, receipt, seed, and next attempt."""

    if state.status.value != "stopped":
        raise FinalizationError("finalization requires a stopped run")
    if not state.best_experiment_id or not state.best_commit_sha:
        raise FinalizationError("stopped run has no verified best")
    if state.best_experiment_id == "baseline":
        raise FinalizationError(
            "baseline finalization uses protected baseline evidence"
        )

    receipt = next(
        (
            event.payload.result
            for event in reversed(events)
            if event.event_type == EventType.PATCH_CHECKED
            and event.payload.result.experiment_id == state.best_experiment_id
            and event.payload.result.patch_commit_sha == state.best_commit_sha
            and event.payload.result.accepted
        ),
        None,
    )
    if receipt is None or receipt.receipt_id is None:
        raise FinalizationError(
            "selected candidate has no exact accepted Gate A receipt"
        )

    evaluation = next(
        (
            event.payload.result
            for event in reversed(events)
            if event.event_type == EventType.EVALUATION_COMPLETED
            and event.payload.result.experiment_id == state.best_experiment_id
            and event.payload.result.fidelity == Fidelity.FULL
            and event.payload.result.population == Population.PUBLIC_VALIDATION
            and event.payload.result.trust.verdict == TrustVerdict.ACCEPTED
            and event.payload.result.metric_set.primary_score
            == state.best_primary_score
        ),
        None,
    )
    if evaluation is None:
        raise FinalizationError(
            "selected candidate has no trusted best-score evaluation"
        )

    attempts = [
        event.payload.request.attempt
        for event in events
        if event.event_type == EventType.EXECUTION_STARTED
        and event.payload.request.experiment_id == state.best_experiment_id
    ]
    return CandidateFinalizationPlan(
        experiment_id=state.best_experiment_id,
        commit_sha=state.best_commit_sha,
        patch_receipt_id=receipt.receipt_id,
        seed=evaluation.seed,
        best_primary_score=state.best_primary_score,
        next_attempt=max(attempts, default=0) + 1,
    )


def baseline_reproduction_event_id(events: Sequence[Event], state) -> str:
    """Return the hash-bound official baseline parity event used at bootstrap."""

    if state.status.value != "stopped" or state.best_experiment_id != "baseline":
        raise FinalizationError("baseline reproduction evidence is not selected")
    baseline = next(
        (event for event in events if event.event_type == EventType.BASELINE_VERIFIED),
        None,
    )
    if baseline is None:
        raise FinalizationError("official baseline parity evidence is missing")
    result = baseline.payload.evaluation
    if not (
        result.fidelity == Fidelity.FULL
        and result.population == Population.PUBLIC_VALIDATION
        and result.trust.verdict == TrustVerdict.ACCEPTED
        and result.metric_set.primary_score == state.best_primary_score
    ):
        raise FinalizationError("official baseline parity evidence is not trusted")
    return baseline.event_id
