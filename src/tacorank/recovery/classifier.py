"""Deterministic classification of completed operational failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fingerprints import fingerprint_result, normalize_text


@dataclass(frozen=True)
class FailureClassification:
    failure_class: str
    reason_code: str
    fingerprint: str
    evidence: str
    deliberate_integrity_violation: bool = False
    made_progress: bool = False


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _failed_checks(result: Any) -> list[str]:
    checks = getattr(result, "checks", None)
    if not isinstance(checks, dict):
        return []
    return [str(name) for name, status in checks.items() if str(_value(status)).lower() == "fail"]


def _violation_text(result: Any) -> str:
    return " ".join(
        f"{getattr(v, 'code', '')} {getattr(v, 'message', v)}"
        for v in (getattr(result, "violations", ()) or ())
    )


def classify_failure(result: Any) -> FailureClassification:
    """Classify a supported result without consulting hidden metrics or an LLM."""
    trust = getattr(result, "trust", None)
    if trust is not None:
        verdict = str(_value(getattr(trust, "verdict", ""))).lower()
        if verdict != "no_op":
            raise ValueError("only an evaluation verdict of no_op is a recovery input")
        failure_class, reason = "no_op", "NO_OP_WIRING"
        evidence = " ".join(getattr(trust, "flags", ()) or ()) or "verified prediction no-op"
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

    lower_evidence = str(evidence).lower()
    deliberate = failure_class == "contract_error" and any(
        token in lower_evidence for token in ("hidden", "secret", "credential", "exfiltrat")
    )
    made_progress = any(
        token in lower_evidence for token in ("progress", "checkpoint", "epoch", "step ")
    )
    return FailureClassification(
        failure_class=failure_class,
        reason_code=reason,
        fingerprint=fingerprint_result(result),
        evidence=normalize_text(str(evidence))[:800],
        deliberate_integrity_violation=deliberate,
        made_progress=made_progress,
    )
