"""Pure evidence selection for sealed post-search finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from ..evaluation.final_selection import CandidateEvidence
from ..schemas import (
    Event,
    EventType,
    ExperimentDecisionKind,
    Fidelity,
    Integrity,
    Population,
    Stability,
    TrustVerdict,
)


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
    events: Sequence[Event], state, experiment_id: str | None = None
) -> CandidateFinalizationPlan:
    """Resolve the exact selected candidate, receipt, seed, and next attempt."""

    if state.status.value != "stopped":
        raise FinalizationError("finalization requires a stopped run")
    selected_id = experiment_id or state.best_experiment_id
    if not selected_id:
        raise FinalizationError("stopped run has no verified best")
    if selected_id == "baseline":
        raise FinalizationError(
            "baseline finalization uses protected baseline evidence"
        )

    receipt = next(
        (
            event.payload.result
            for event in reversed(events)
            if event.event_type == EventType.PATCH_CHECKED
            and event.payload.result.experiment_id == selected_id
            and event.payload.result.patch_commit_sha
            == state.experiments[selected_id].latest_commit_sha
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
            and event.payload.result.experiment_id == selected_id
            and event.payload.result.fidelity == Fidelity.FULL
            and event.payload.result.population == Population.PUBLIC_VALIDATION
            and event.payload.result.trust.verdict == TrustVerdict.ACCEPTED
            and event.payload.result.trust.stability == Stability.CONFIRMED
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
        and event.payload.request.experiment_id == selected_id
    ]
    return CandidateFinalizationPlan(
        experiment_id=selected_id,
        commit_sha=state.experiments[selected_id].latest_commit_sha,
        patch_receipt_id=receipt.receipt_id,
        seed=evaluation.seed,
        best_primary_score=evaluation.metric_set.primary_score,
        next_attempt=max(attempts, default=0) + 1,
    )


def baseline_reproduction_event_id(events: Sequence[Event], state) -> str:
    """Return the hash-bound official baseline parity event used at bootstrap."""

    if state.status.value not in {"stopped", "finalizing"}:
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
        and result.metric_set.primary_score == state.baseline_primary_score
    ):
        raise FinalizationError("official baseline parity evidence is not trusted")
    return baseline.event_id


def finalization_candidates(
    events: Sequence[Event], state
) -> Tuple[CandidateEvidence, ...]:
    """Build post-search selection evidence from trusted ledger events."""

    baseline_event = next(
        event for event in events if event.event_type == EventType.BASELINE_VERIFIED
    )
    proposals = {
        event.payload.spec.experiment_id: event.payload.spec
        for event in events
        if event.event_type == EventType.EXPERIMENT_PROPOSED
    }
    evaluations = {
        event.event_id: event.payload.result
        for event in events
        if event.event_type == EventType.EVALUATION_COMPLETED
    }
    terminal_results: Dict[str, object] = {}
    for event in events:
        if event.event_type != EventType.EXPERIMENT_DECIDED:
            continue
        decision = event.payload.decision
        if (
            decision.decision == ExperimentDecisionKind.ACCEPT
            and decision.parent_eligible
            and decision.evaluation_event_id in evaluations
        ):
            terminal_results[decision.experiment_id] = evaluations[
                decision.evaluation_event_id
            ]

    val_b_cache: Dict[str, float | None] = {"baseline": 0.0}

    def cumulative_val_b(experiment_id: str) -> float | None:
        if experiment_id in val_b_cache:
            return val_b_cache[experiment_id]
        result = terminal_results.get(experiment_id)
        spec = proposals.get(experiment_id)
        if result is None or spec is None:
            val_b_cache[experiment_id] = None
            return None
        parent = cumulative_val_b(spec.parent_experiment_id or "baseline")
        delta = result.diagnostics.validation_arm_deltas.get("val_b")
        val_b_cache[experiment_id] = (
            None if parent is None or delta is None else parent + float(delta)
        )
        return val_b_cache[experiment_id]

    candidates = [
        CandidateEvidence(
            experiment_id="baseline",
            commit_sha=baseline_event.payload.commit_sha,
            public_score=state.baseline_primary_score,
            val_b_score=0.0,
            trust_verdict=TrustVerdict.ACCEPTED,
            stability=Stability.CONFIRMED,
            integrity=Integrity.CLEAN,
            internal_holdout_agrees=True,
            unbiased_audit_agrees=True,
            clean_reproduction_passed=True,
        )
    ]
    for experiment_id, result in terminal_results.items():
        node = state.experiments.get(experiment_id)
        if node is None or not node.latest_commit_sha:
            continue
        val_b = cumulative_val_b(experiment_id)
        proxy_delta = result.diagnostics.proxy_parent_delta
        parent_delta = result.parent_delta
        candidates.append(
            CandidateEvidence(
                experiment_id=experiment_id,
                commit_sha=node.latest_commit_sha,
                public_score=result.metric_set.primary_score,
                val_b_score=val_b,
                trust_verdict=result.trust.verdict,
                stability=result.trust.stability,
                integrity=result.trust.integrity,
                internal_holdout_agrees=(
                    proxy_delta is not None
                    and parent_delta is not None
                    and float(proxy_delta) * float(parent_delta) >= 0
                ),
                unbiased_audit_agrees=val_b is not None and val_b > 0,
                clean_reproduction_passed=False,
            )
        )
    return tuple(candidates)
