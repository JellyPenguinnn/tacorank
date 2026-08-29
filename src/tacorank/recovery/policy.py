"""Bounded deterministic recovery policy."""

from __future__ import annotations

from typing import Any

from .classifier import FailureClassification, classify_failure
from .operational_reflection import build_operational_lesson
from .runtime_adjustments import select_runtime_adjustment
from .self_debug import build_self_debug_instructions

MAX_REPAIR_ATTEMPTS = 2


def _action(value: str) -> Any:
    from tacorank.schemas import RecoveryAction

    return RecoveryAction(value)


class RecoveryManager:
    """Choose one auditable recovery action without executing it."""

    async def decide(self, failure_event_id: str, result: Any, context: Any) -> Any:
        """Implement the canonical asynchronous recovery-manager port."""
        if not failure_event_id:
            raise ValueError("failure_event_id is required")

        classification = classify_failure(result)
        remaining_before = min(
            MAX_REPAIR_ATTEMPTS,
            max(0, int(getattr(context, "remaining_repair_budget", 0))),
        )
        prior = getattr(context, "previous_error_fingerprints", None)
        if prior is None:
            prior = getattr(context, "prior_error_fingerprints", ())
        same_count = list(prior or ()).count(classification.fingerprint) + 1

        action, reason, instructions = self._route(
            classification, context, same_count, remaining_before
        )
        consumes_repair = action == "trae_repair"
        repair_attempt = max(1, MAX_REPAIR_ATTEMPTS - remaining_before + 1)
        remaining_after = max(0, remaining_before - int(consumes_repair))
        lesson = build_operational_lesson(
            classification,
            context,
            exhausted=action == "abandon",
            failure_event_id=failure_event_id,
        )

        from tacorank.schemas import RecoveryDecision

        return RecoveryDecision(
            run_id=context.run_id,
            experiment_id=context.experiment_id,
            failure_event_id=failure_event_id,
            repair_attempt=repair_attempt,
            action=_action(action),
            reason_code=reason,
            instructions=instructions,
            same_error_count=same_count,
            remaining_repair_budget=remaining_after,
            lesson_candidate=lesson,
        )

    @staticmethod
    def _route(
        failure: FailureClassification,
        context: Any,
        same_count: int,
        remaining: int,
    ) -> tuple[str, str, str]:
        if failure.deliberate_integrity_violation:
            return "abandon", "INTEGRITY_VIOLATION", (
                "Abandon: the failure crosses a protected integrity boundary."
            )
        if same_count >= 2:
            return "abandon", "REPEATED_ERROR_FINGERPRINT", (
                "Abandon: the same locally normalized failure occurred twice; "
                "do not spend another attempt."
            )
        if failure.failure_class in {"infrastructure_error", "hang", "timeout"}:
            return "retry_same_commit", "TRANSIENT_SAME_COMMIT_RETRY", (
                "Retry the exact sealed commit once with identical approved settings; "
                "do not invoke code repair."
            )

        adjustment = select_runtime_adjustment(failure.failure_class, context)
        if failure.failure_class in {"oom", "numerical_error"} and adjustment is not None:
            return (
                "adjust_approved_runtime_setting",
                f"APPROVED_{adjustment.name.upper()}_ADJUSTMENT",
                adjustment.instruction(),
            )
        if failure.failure_class == "oom":
            return "rollback", "OOM_NO_APPROVED_ADJUSTMENT", (
                "Rollback: no legal lower-memory setting is available in the frozen allowlist."
            )
        if remaining <= 0:
            return "abandon", "REPAIR_BUDGET_EXHAUSTED", (
                "Abandon: the maximum of two code-repair attempts has been consumed."
            )

        attempt = max(1, MAX_REPAIR_ATTEMPTS - remaining + 1)
        instructions = build_self_debug_instructions(
            failure,
            context,
            attempt,
            remaining - 1,
        )
        return "trae_repair", f"REPAIRABLE_{failure.reason_code}", instructions


async def decide_recovery(failure_event_id: str, result: Any, context: Any) -> Any:
    return await RecoveryManager().decide(failure_event_id, result, context)
