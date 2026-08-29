"""Sparse operational lesson candidates emitted only from reusable evidence."""

from __future__ import annotations

from typing import Any

from .classifier import FailureClassification


def _construct(model: type, **values: Any) -> Any:
    return model(**values)


def build_operational_lesson(
    classification: FailureClassification,
    context: Any,
    *,
    exhausted: bool,
) -> Any | None:
    """Return a typed lesson only for exhausted/reusable operational constraints."""
    if not exhausted and not classification.deliberate_integrity_violation:
        return None
    failure = classification.failure_class
    repeated = list(getattr(context, "prior_error_fingerprints", ()) or ()).count(
        classification.fingerprint
    ) >= 1
    if failure == "oom" and repeated:
        category, tags = "resource_constraint", ["memory", "oom"]
        summary = "The configured execution profile exhausted memory on two attempts."
        applicability = "The current data, patch, and resource profile."
        avoid_when = "Avoid the same or larger memory footprint without an approved lower-memory setting."
        confidence = 0.95
    elif failure == "no_op" and exhausted:
        category, tags = "implementation_constraint", ["wiring", "no_op"]
        summary = "The accepted change produced unchanged predictions after focused wiring recovery."
        applicability = "This implementation path and configuration-consumption boundary."
        avoid_when = "Do not treat the underlying research hypothesis as falsified until wiring is verified."
        confidence = 0.9
    elif failure == "hang" and exhausted:
        category, tags = "process_rule", ["hang", "heartbeat"]
        summary = "The same execution profile repeatedly stopped making observable progress."
        applicability = "The current command and heartbeat profile."
        avoid_when = "Do not reuse this profile until its progress signal or process behavior is corrected."
        confidence = 0.9
    elif classification.deliberate_integrity_violation:
        category, tags = "integrity_warning", ["contract", "protected_boundary"]
        summary = "Execution attempted access forbidden by the protected contract."
        applicability = "All candidate patches under the frozen competition contract."
        avoid_when = "Never access hidden labels, secrets, credentials, or protected evaluator behavior."
        confidence = 0.99
    else:
        return None

    from tacorank.schemas import LessonCandidate

    return _construct(
        LessonCandidate,
        origin="operational",
        category=category,
        tags=tags,
        summary=summary,
        applicability=applicability,
        avoid_when=avoid_when,
        confidence=confidence,
        source_event_ids=[getattr(context, "failure_event_id", None)]
        if getattr(context, "failure_event_id", None)
        else [],
        source_commit_shas=[getattr(context, "current_patch_commit_sha", None)]
        if getattr(context, "current_patch_commit_sha", None)
        else [],
    )
