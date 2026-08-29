"""Deterministic recovery classification and bounded policy."""

from .classifier import FailureClassification, classify_failure
from .fingerprints import fingerprint_failure, fingerprint_result, normalize_text
from .policy import MAX_REPAIR_ATTEMPTS, RecoveryManager, decide_recovery

__all__ = [
    "FailureClassification",
    "MAX_REPAIR_ATTEMPTS",
    "RecoveryManager",
    "classify_failure",
    "decide_recovery",
    "fingerprint_failure",
    "fingerprint_result",
    "normalize_text",
]
