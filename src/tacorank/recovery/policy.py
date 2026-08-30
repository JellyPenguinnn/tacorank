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
        consumes_repair = action in {
            "trae_repair",
            "restart_from_trusted_parent",
        }
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
        if failure.candidate_integrity_violation:
            if same_count >= 2 or remaining <= 0:
                return (
                    "abandon",
                    "CANDIDATE_INTEGRITY_RETRY_EXHAUSTED",
                    "Abandon only this experiment: the clean coding restart "
                    "was exhausted. Preserve the violation evidence and return "
                    "control to planning; do not weaken a safety gate.",
                    {},
                )
            attempt = int(context.repair_attempts_used) + 1
            return (
                "restart_from_trusted_parent",
                "CANDIDATE_INTEGRITY_CLEAN_RESTART",
                build_self_debug_instructions(
                    failure, context, attempt, remaining - 1
                )
                + "\nDiscard the rejected candidate completely and restart from "
                "the declared trusted parent commit. Treat the cited integrity "
                "finding as a hard constraint. Do not edit, bypass, or weaken "
                "Gate A, Gate B, protected paths, data policy, network policy, "
                "or contract checks.",
                {},
            )
        if failure.deliberate_integrity_violation:
            return (
                "abandon",
                "INTEGRITY_VIOLATION",
                "Abandon: the failure crosses a protected integrity boundary.",
                {},
            )
        if failure.disk_quota_failure:
            return (
                "abandon",
                "DISK_QUOTA_EXHAUSTED",
                "Abandon: storage or the reviewed output quota is exhausted; free space or use a new runtime.",
                {},
            )
        for dimension, value in context.remaining_run_budget.items():
            if value <= 0:
                return (
                    "abandon",
                    "RUN_%s_BUDGET_EXHAUSTED" % dimension.upper(),
                    "Abandon: the frozen %s run budget is exhausted." % dimension,
                    {},
                )

        if failure.control_plane_failure:
            return (
                "abandon",
                "CONTROL_PLANE_INVARIANT_FAILURE",
                "Stop experiment recovery: the failure belongs to %s and may "
                "indicate inconsistent safety or orchestration evidence. Preserve "
                "the artifacts for operator review; do not ask Trae to edit "
                "candidate code." % failure.owner,
                {},
            )

        if failure.owner == "solution_verifier" and not failure.owner_retryable:
            return (
                "abandon",
                "SOLUTION_VERIFIER_RETRY_EXHAUSTED",
                "Abandon this experiment: the solution verifier exhausted its "
                "own bounded protocol/provider retries. Preserve the candidate "
                "trajectory and verifier diagnostics; do not rerun Trae or ask "
                "it to change code without a valid verifier finding.",
                {},
            )

        if failure.failure_class == "no_op":
            if same_count >= 2 or remaining <= 0:
                return (
                    "return_to_planner",
                    "NO_OP_RECOVERY_EXHAUSTED",
                    "The bounded no-op repair did not change predictions; return "
                    "the recorded evidence to the research planner.",
                    {},
                )
            attempt = int(context.repair_attempts_used) + 1
            return (
                "trae_repair",
                "REPAIRABLE_NO_OP_WIRING",
                build_self_debug_instructions(
                    failure, context, attempt, remaining - 1
                ),
                {},
            )

        no_op_repair_in_progress = any(
            item.get("action") == "trae_repair"
            and item.get("reason_code") == "REPAIRABLE_NO_OP_WIRING"
            for item in context.attempt_history
        )
        if (
            getattr(context, "failure_stage", None) == "coding"
            and no_op_repair_in_progress
        ):
            return (
                "return_to_planner",
                "NO_OP_REPAIR_WORKER_EXHAUSTED",
                "The bounded no-op repair worker did not produce a valid "
                "replacement patch; preserve both records and return the "
                "unchanged-prediction evidence to the research planner.",
                {},
            )

        if failure.owner_retryable:
            owner_code = failure.owner.upper()
            if same_count >= 2 or int(context.same_commit_retries_used) >= 1:
                return (
                    "abandon",
                    "%s_RETRY_EXHAUSTED" % owner_code,
                    "Abandon: the one bounded retry of %s is exhausted; preserve "
                    "the candidate and failure evidence without asking Trae to "
                    "change code." % failure.owner,
                    {},
                )
            if failure.owner == "coding_worker":
                instruction = (
                    "Retry the same coding assignment once. Give Trae this exact "
                    "diagnostic: %s. Ask it to state DIAGNOSIS, REPAIR_PLAN, and "
                    "VERIFICATION, then produce the smallest valid candidate "
                    "patch without changing the accepted hypothesis or protected "
                    "contract."
                    % (failure.evidence or failure.reason_code)
                )
            else:
                instruction = (
                    "Retry only the %s stage once against the same immutable "
                    "inputs. The owner must correct or reissue its response using "
                    "this diagnostic: %s. Preserve the candidate; do not ask Trae "
                    "to edit code unless a later valid result identifies a "
                    "candidate defect."
                    % (failure.owner, failure.evidence or failure.reason_code)
                )
            return (
                "retry_same_commit",
                "TRANSIENT_%s_RETRY" % owner_code,
                instruction,
                {},
            )

        if same_count >= 2:
            return (
                "abandon",
                "REPEATED_ERROR_FINGERPRINT",
                "Abandon: the same locally normalized failure occurred twice.",
                {},
            )

        if getattr(context, "failure_stage", None) == "coding":
            return (
                "abandon",
                "CODING_WORKER_FAILURE",
                "Abandon: the coding worker did not produce a valid patch candidate.",
                {},
            )

        adjustment = select_runtime_adjustment(failure.failure_class, context)
        if failure.failure_class in {"oom", "numerical_error", "timeout"} and adjustment is not None:
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

        if not failure.trae_repairable:
            return (
                "abandon",
                "OWNER_REPAIR_UNAVAILABLE",
                "Abandon: the failure belongs to %s and contains no validated "
                "candidate-code defect for Trae to repair." % failure.owner,
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
