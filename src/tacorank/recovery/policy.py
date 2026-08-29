"""Bounded deterministic recovery policy."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .classifier import FailureClassification, classify_failure
from .operational_reflection import build_operational_lesson
from .runtime_adjustments import select_runtime_adjustment
from .self_debug import build_self_debug_instructions

MAX_REPAIR_ATTEMPTS = 2


class RecoveryManager:
    """Choose one auditable recovery action without executing it."""

    async def decide(self, failure_event_id: str, result: Any, context: Any) -> Any:
        if not failure_event_id:
            raise ValueError("failure_event_id is required")
        if context.failure_event_id != failure_event_id:
            raise ValueError("recovery context cites a different failure event")

        classification = classify_failure(result)
        maximum = min(MAX_REPAIR_ATTEMPTS, int(context.max_repair_attempts))
        used = int(context.repair_attempts_used)
        remaining_before = int(context.remaining_repair_budget)
        same_count = (
            list(context.previous_error_fingerprints).count(
                classification.fingerprint
            )
            + 1
        )

        action, reason, instructions, adjustments = self._route(
            classification, context, same_count, remaining_before
        )
        consumes_repair = action == "trae_repair"
        remaining_after = remaining_before - int(consumes_repair)
        repair_attempt = min(max(1, used + 1), max(1, maximum))
        lesson = build_operational_lesson(
            classification,
            context,
            exhausted=action == "abandon",
            failure_event_id=failure_event_id,
        )

        from tacorank.schemas import RecoveryAction, RecoveryDecision

        return RecoveryDecision(
            run_id=context.run_id,
            experiment_id=context.experiment_id,
            failure_event_id=failure_event_id,
            repair_attempt=repair_attempt,
            action=RecoveryAction(action),
            reason_code=reason,
            instructions=instructions,
            same_error_count=same_count,
            remaining_repair_budget=remaining_after,
            runtime_adjustments=adjustments,
            lesson_candidate=lesson,
        )

    @staticmethod
    def _route(
        failure: FailureClassification,
        context: Any,
        same_count: int,
        remaining: int,
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        if failure.deliberate_integrity_violation:
            return (
                "abandon",
                "INTEGRITY_VIOLATION",
                "Abandon: the failure crosses a protected integrity boundary.",
                {},
            )
        if same_count >= 2:
            return (
                "abandon",
                "REPEATED_ERROR_FINGERPRINT",
                "Abandon: the same locally normalized failure occurred twice.",
                {},
            )

        adjustment = select_runtime_adjustment(failure.failure_class, context)
        if failure.failure_class in {"oom", "numerical_error"} and adjustment is not None:
            return (
                "adjust_approved_runtime_setting",
                "APPROVED_%s_ADJUSTMENT" % adjustment.name.upper(),
                adjustment.instruction(),
                {adjustment.name: adjustment.value},
            )
        if failure.failure_class == "oom":
            return (
                "rollback",
                "OOM_NO_APPROVED_ADJUSTMENT",
                "Rollback: no legal lower-memory setting is available.",
                {},
            )

        transient = failure.failure_class in {"infrastructure_error", "hang"}
        timeout_retry = failure.failure_class == "timeout" and failure.made_progress
        if transient or timeout_retry:
            if int(context.same_commit_retries_used) < 1:
                return (
                    "retry_same_commit",
                    "TRANSIENT_SAME_COMMIT_RETRY",
                    "Retry the exact sealed commit once with identical settings.",
                    {},
                )
            return (
                "abandon",
                "SAME_COMMIT_RETRY_EXHAUSTED",
                "Abandon: the exact same-commit retry has already been used.",
                {},
            )
        if failure.failure_class == "timeout":
            return (
                "abandon",
                "TIMEOUT_WITHOUT_PROGRESS",
                "Abandon: timeout evidence does not show reliable progress.",
                {},
            )
        if remaining <= 0:
            return (
                "abandon",
                "REPAIR_BUDGET_EXHAUSTED",
                "Abandon: the code-repair budget has been consumed.",
                {},
            )

        attempt = int(context.repair_attempts_used) + 1
        instructions = build_self_debug_instructions(
            failure, context, attempt, remaining - 1
        )
        return (
            "trae_repair",
            "REPAIRABLE_%s" % failure.reason_code,
            instructions,
            {},
        )


async def decide_recovery(failure_event_id: str, result: Any, context: Any) -> Any:
    return await RecoveryManager().decide(failure_event_id, result, context)
