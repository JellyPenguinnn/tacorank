"""Deterministic classification of completed operational failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fingerprints import fingerprint_result, normalize_text
from ..safety.path_policy import (
    CANDIDATE_SCOPED_INTEGRITY_CODES,
    DELIBERATE_INTEGRITY_CODES,
)


TRANSIENT_CODING_ERROR_CODES = frozenset(
    {
        "TRAE_LAUNCH_FAILED",
        "TRAE_TIMEOUT",
        "TRAE_DOCKER_UNAVAILABLE",
        "TRAE_PROVIDER_TIMEOUT",
        "TRAE_PROVIDER_UNAVAILABLE",
    }
)

# These failures belong to the reviewer/provider protocol.  They are not
# evidence that the candidate implementation is defective, so recovery must
# never turn them into a Trae edit prompt.
VERIFIER_PROTOCOL_ERROR_CODES = frozenset(
    {
        "SOLUTION_VERIFIER_MALFORMED",
    }
)

TRANSIENT_PROVIDER_ERROR_CODES = frozenset(
    {
        "TRAE_PROVIDER_TIMEOUT",
        "TRAE_PROVIDER_UNAVAILABLE",
        "PROVIDER_TIMEOUT",
        "PROVIDER_UNAVAILABLE",
        "EVALUATOR_PROVIDER_TIMEOUT",
        "EVALUATOR_PROVIDER_UNAVAILABLE",
    }
)

# The coding agent produced no usable candidate or protocol record.  There is
# no sealed patch to repair, but the same coding assignment may be reissued
# once with the recorded diagnostic.
CORRECTABLE_CODING_AGENT_ERROR_CODES = frozenset(
    {
        "NO_PATCH",
        "SOLUTION_REVISION_NO_CHANGE",
        "TRAE_PROCESS_FAILED",
        "TRAE_REPORTED_FAILURE",
        "TRAJECTORY_ENCODING",
        "TRAJECTORY_MALFORMED",
        "TRAJECTORY_MISSING",
    }
)

# A mismatch in these identities is ambiguous: it can indicate corruption or
# a controller/gate defect.  An experiment-scoped coding worker is not allowed
# to "repair" the evidence or the safety mechanism that detected it.
CONTROL_PLANE_INVARIANT_CODES = frozenset(
    {
        "CONTRACT_HASH_MISMATCH",
        "DIFF_MISMATCH",
        "OUTPUT_ARTIFACT_MISMATCH",
        "OUTPUT_IDENTITY_MISMATCH",
        "OUTPUT_PRODUCER_MISMATCH",
        "PATCH_RECEIPT_MISMATCH",
        "EXECUTION_SEAL_MISMATCH",
    }
)

DISK_QUOTA_ERROR_CODES = frozenset(
    {
        "DISK_QUOTA_EXHAUSTED",
        "DISK_SPACE_EXHAUSTED",
        "DISK_LOW",
        "OBSERVED_DISK_FREE_FLOOR",
        "OUTPUT_QUOTA_EXCEEDED",
        "ENOSPC",
    }
)


@dataclass(frozen=True)
class FailureClassification:
    failure_class: str
    reason_code: str
    fingerprint: str
    evidence: str
    deliberate_integrity_violation: bool = False
    made_progress: bool = False
    transient_coding_failure: bool = False
    disk_quota_failure: bool = False
    owner: str = "coding_worker"
    owner_retryable: bool = False
    trae_repairable: bool = False
    control_plane_failure: bool = False
    candidate_integrity_violation: bool = False


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _failed_checks(result: Any) -> list[str]:
    checks = getattr(result, "checks", None)
    if isinstance(checks, dict):
        return [
            str(name)
            for name, status in checks.items()
            if str(_value(status)).lower() == "fail"
        ]
    return [
        str(getattr(check, "name", ""))
        for check in (checks or ())
        if str(_value(getattr(check, "status", ""))).lower() == "fail"
    ]


def _violation_text(result: Any) -> str:
    return " ".join(
        f"{getattr(v, 'code', '')} {getattr(v, 'message', v)}"
        for v in (getattr(result, "violations", ()) or ())
    )


def _violation_codes(result: Any) -> set[str]:
    return {
        str(getattr(violation, "code", "")).strip().upper()
        for violation in (getattr(result, "violations", ()) or ())
        if getattr(violation, "code", None)
    }


def classify_failure(result: Any) -> FailureClassification:
    """Classify a supported result without consulting hidden metrics or an LLM."""
    stage = str(_value(getattr(result, "failure_stage", ""))).strip().lower()
    error_code = str(getattr(result, "error_class", "")).strip().upper()
    owner = {
        "coding": "coding_worker",
        "patch_gate": "patch_gate",
        "execution": "execution_runner",
        "output_gate": "output_gate",
        "evaluation": "evaluator",
        "recovery": "recovery_control_plane",
    }.get(stage, "coding_worker")
    owner_retryable = False
    trae_repairable = False
    control_plane = stage == "recovery"

    trust = getattr(result, "trust", None)
    if trust is not None:
        verdict = str(_value(getattr(trust, "verdict", ""))).lower()
        if verdict != "no_op":
            raise ValueError("only an evaluation verdict of no_op is a recovery input")
        failure_class, reason = "no_op", "NO_OP_WIRING"
        evidence = " ".join(getattr(trust, "flags", ()) or ()) or "verified prediction no-op"
        owner = "coding_worker"
        trae_repairable = True
    elif hasattr(result, "outcome"):
        outcome = str(_value(getattr(result, "outcome", ""))).lower()
        if outcome in {"success", "cancelled"}:
            raise ValueError(f"{outcome or 'successful'} execution is not a recovery input")
        valid = {
            "code_error", "interface_error", "contract_error", "numerical_error",
            "oom", "timeout", "hang", "infrastructure_error",
        }
        failure_class = outcome if outcome in valid else "code_error"
        reason = {
            "code_error": "CODE_ERROR",
            "interface_error": "INTERFACE_ERROR",
            "contract_error": "CONTRACT_ERROR",
            "numerical_error": "NUMERICAL_ERROR",
            "oom": "OUT_OF_MEMORY",
            "timeout": "EXECUTION_TIMEOUT",
            "hang": "EXECUTION_HANG",
            "infrastructure_error": "INFRASTRUCTURE_ERROR",
        }[failure_class]
        evidence = getattr(result, "error_summary", None) or str(getattr(result, "error_class", ""))
        # A coding adapter failure means Trae did not produce a candidate. It
        # must not be treated as an execution infrastructure retry, because
        # retrying the sealed commit would not address the failed coding call.
        if getattr(result, "failure_stage", None) == "coding":
            failure_class, reason = "code_error", "CODING_WORKER_FAILURE"
        elif not stage:
            owner = "execution_runner"
            trae_repairable = failure_class in {
                "code_error", "interface_error", "contract_error",
                "numerical_error",
            }
    else:
        accepted = getattr(result, "accepted", None)
        if accepted is not False:
            raise ValueError("accepted gate/output result is not a recovery input")
        failed = _failed_checks(result)
        violations = _violation_text(result)
        combined = " ".join(failed + [violations]).strip()
        lower = combined.lower()
        if any(token in lower for token in ("hidden", "secret", "credential", "network", "target label")):
            failure_class, reason = "contract_error", "INTEGRITY_VIOLATION"
        elif isinstance(getattr(result, "checks", None), dict):
            failure_class, reason = "output_contract", "OUTPUT_CONTRACT_ERROR"
        else:
            failure_class, reason = "contract_error", "PATCH_CONTRACT_ERROR"
        evidence = combined
        owner = "coding_worker"
        trae_repairable = True

    lower_evidence = str(evidence).lower()
    violation_codes = _violation_codes(result)
    all_integrity_codes = violation_codes | ({error_code} if error_code else set())
    candidate_integrity = bool(
        all_integrity_codes & CANDIDATE_SCOPED_INTEGRITY_CODES
    )
    deliberate = bool(all_integrity_codes & DELIBERATE_INTEGRITY_CODES) or any(
        token in lower_evidence
        for token in (
            "hidden label",
            "target label",
            "secret",
            "credential",
            "exfiltrat",
            "unauthorized network",
            "unapproved network",
        )
    )
    made_progress = any(
        token in lower_evidence for token in ("progress", "checkpoint", "epoch", "step ")
    )
    transient_coding = (
        getattr(result, "failure_stage", None) == "coding"
        and error_code in TRANSIENT_CODING_ERROR_CODES
    )
    if error_code in VERIFIER_PROTOCOL_ERROR_CODES:
        owner = "solution_verifier"
        # The verifier performs its own bounded protocol correction before
        # surfacing this code.  An outer retry would rerun Trae and discard the
        # staged candidate rather than retrying the failed owner safely.
        owner_retryable = False
        trae_repairable = False
        failure_class, reason = (
            "infrastructure_error",
            "SOLUTION_VERIFIER_RETRY_EXHAUSTED",
        )
    elif error_code in TRANSIENT_PROVIDER_ERROR_CODES:
        # Verifier failures carry a stable verifier marker in their redacted
        # summary.  Otherwise the adapter boundary itself remains the owner.
        if "solution verifier" in lower_evidence:
            owner = "solution_verifier"
            owner_retryable = False
            failure_class, reason = (
                "infrastructure_error",
                "SOLUTION_VERIFIER_RETRY_EXHAUSTED",
            )
        else:
            owner_retryable = True
        trae_repairable = False
    elif transient_coding or (
        stage == "coding" and error_code in CORRECTABLE_CODING_AGENT_ERROR_CODES
    ):
        owner_retryable = True
        trae_repairable = False

    disk_quota = error_code in DISK_QUOTA_ERROR_CODES or any(
        marker in lower_evidence
        for marker in (
            "no space left",
            "disk quota",
            "storage is full",
            "disk free floor",
            "enospc",
        )
    )
    if disk_quota and getattr(result, "failure_stage", None) != "coding":
        failure_class, reason = "infrastructure_error", "DISK_QUOTA_EXHAUSTED"
    invariant_failure = bool(violation_codes & CONTROL_PLANE_INVARIANT_CODES)
    if invariant_failure:
        owner = "recovery_control_plane"
        owner_retryable = False
        trae_repairable = False
        control_plane = True
    if deliberate and not candidate_integrity:
        owner = "operator"
        owner_retryable = False
        trae_repairable = False
    elif candidate_integrity:
        owner = "coding_worker"
        owner_retryable = False
        trae_repairable = False
    elif disk_quota:
        owner = "operator"
        owner_retryable = False
        trae_repairable = False
    elif failure_class in {"infrastructure_error", "hang", "timeout"} and stage == "execution":
        owner = "execution_runner"
        owner_retryable = True
    elif stage and stage != "coding" and failure_class in {
        "code_error", "interface_error", "contract_error", "numerical_error",
    }:
        # Reaching this branch means an AdapterFailureResult escaped the
        # boundary (typed gate rejections do not have ``failure_stage``).
        # The exception is not validated evidence of a candidate defect.
        trae_repairable = False
    if (
        stage in {"patch_gate", "output_gate", "evaluation"}
        and not owner_retryable
    ):
        control_plane = True
        trae_repairable = False
    return FailureClassification(
        failure_class=failure_class,
        reason_code=reason,
        fingerprint=fingerprint_result(result),
        evidence=normalize_text(str(evidence))[:800],
        deliberate_integrity_violation=deliberate,
        made_progress=made_progress,
        transient_coding_failure=transient_coding,
        disk_quota_failure=disk_quota,
        owner=owner,
        owner_retryable=owner_retryable,
        trae_repairable=trae_repairable,
        control_plane_failure=control_plane,
        candidate_integrity_violation=candidate_integrity,
    )
