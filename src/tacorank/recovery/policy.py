"""Bounded deterministic recovery policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .classifier import FailureClassification, classify_failure
from .operational_reflection import build_operational_lesson
from .runtime_adjustments import select_runtime_adjustment
from .self_debug import build_self_debug_instructions

MAX_REPAIR_ATTEMPTS = 2


def _has_run_budget(value: Any) -> bool:
    if isinstance(value, Mapping):
        numeric = [item for item in value.values() if isinstance(item, (int, float))]
        return not numeric or any(item > 0 for item in numeric)
    return int(value) > 0


def _action(value: str) -> Any:
    from tacorank.schemas import RecoveryAction

    try:
        return RecoveryAction(value)
    except (TypeError, ValueError):
        return getattr(RecoveryAction, value.upper(), value)


class RecoveryManager:
    """Choose one auditable recovery action without executing it."""

    def decide(self, *args: Any, failure_event_id: str | None = None) -> Any:
        """Accept ``(result, context)`` or legacy ``(event_id, result, context)``."""
        if len(args) == 3 and isinstance(args[0], str):
            failure_event_id, result, context = args
        elif len(args) == 2:
            result, context = args
        elif len(args) == 3:
            result, context, positional_event_id = args
            failure_event_id = failure_event_id or positional_event_id
        else:
            raise TypeError("decide expects (result, context) or (failure_event_id, result, context)")
        classification = classify_failure(result)
        event_id = failure_event_id or getattr(context, "failure_event_id", None)
        if not event_id:
            raise ValueError("failure_event_id is required")

        used = max(0, int(getattr(context, "repair_attempts_used", 0)))
        configured_max = int(getattr(context, "max_repair_attempts", MAX_REPAIR_ATTEMPTS))
        limit = min(MAX_REPAIR_ATTEMPTS, max(0, configured_max))
        remaining_before = max(0, limit - used)
        prior = list(getattr(context, "prior_error_fingerprints", ()) or ())
        same_count = prior.count(classification.fingerprint) + 1

        action, reason, instructions = self._route(
            classification, context, same_count, remaining_before
        )
        consumes_repair = action == "trae_repair"
        repair_attempt = used + 1 if consumes_repair else used
        remaining_after = max(0, remaining_before - int(consumes_repair))
        exhausted = action == "abandon"
        lesson = build_operational_lesson(classification, context, exhausted=exhausted)

        from tacorank.schemas import RecoveryDecision

        return RecoveryDecision(
            run_id=getattr(context, "run_id"),
            experiment_id=getattr(context, "experiment_id"),
            failure_event_id=event_id,
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
        if same_count >= 2:
            return "abandon", "REPEATED_ERROR_FINGERPRINT", (
                "Abandon: the same normalized failure occurred twice; do not spend another repair call."
            )
        if not _has_run_budget(getattr(context, "remaining_run_budget", 1)):
            return "abandon", "RUN_BUDGET_EXHAUSTED", "Abandon: no sealed run budget remains."
        if failure.deliberate_integrity_violation:
            return "abandon", "INTEGRITY_VIOLATION", (
                "Abandon: the failure crosses a protected integrity boundary."
            )
        if failure.failure_class in {"infrastructure_error", "hang"}:
            if int(getattr(context, "same_commit_retries_used", 0)) < 1:
                return "retry_same_commit", "TRANSIENT_SAME_COMMIT_RETRY", (
                    "Retry the exact sealed commit once with identical approved settings; do not invoke code repair."
                )
            return "abandon", "SAME_COMMIT_RETRY_EXHAUSTED", (
                "Abandon: the one exact same-commit retry has already been used."
            )
        adjustment = select_runtime_adjustment(failure.failure_class, context)
        if failure.failure_class in {"oom", "timeout", "numerical_error"} and adjustment is not None:
            return "adjust_approved_runtime_setting", f"APPROVED_{adjustment.name.upper()}_ADJUSTMENT", adjustment.instruction()
        if failure.failure_class == "oom":
            return "rollback", "OOM_NO_APPROVED_ADJUSTMENT", (
                "Rollback: no legal lower-memory setting is available in the frozen allowlist."
            )
        if failure.failure_class == "timeout":
            return "abandon", "TIMEOUT_TOO_COSTLY", (
                "Abandon: progress does not authorize a larger timeout and no approved profile fits the budget."
            )
        if remaining <= 0:
            return "abandon", "REPAIR_BUDGET_EXHAUSTED", (
                "Abandon: the maximum of two code-repair attempts has been consumed."
            )
        attempt = int(getattr(context, "repair_attempts_used", 0)) + 1
        instructions = build_self_debug_instructions(
            failure,
            context,
            attempt,
            remaining - 1,
        )
        return "trae_repair", f"REPAIRABLE_{failure.reason_code}", instructions


def decide_recovery(result: Any, context: Any, failure_event_id: str | None = None) -> Any:
    return RecoveryManager().decide(result, context, failure_event_id=failure_event_id)
