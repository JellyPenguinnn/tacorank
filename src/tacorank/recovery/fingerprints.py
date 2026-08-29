"""Stable, privacy-preserving fingerprints for recovery failures."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_LINE_NUMBER = re.compile(r"(line\s+)\d+", re.IGNORECASE)
_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_TEMP_PATH = re.compile(
    r"(?i)(?:[A-Z]:\\|/)(?:[^\s:'\"]*[\\/])?(?:tmp|temp)[\\/][^\s:'\"]+"
)
_PY_FRAME = re.compile(
    r'File\s+["\'](?P<path>[^"\']+)["\'],\s+line\s+\d+,\s+in\s+(?P<func>[^\s]+)'
)
_EXCEPTION = re.compile(r"(?m)^([A-Za-z_][\w.]*(?:Error|Exception))\s*:")


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def normalize_text(text: str | None) -> str:
    """Remove volatile details while retaining diagnostic structure."""
    if not text:
        return ""
    text = _TIMESTAMP.sub("<timestamp>", str(text))
    text = _ADDRESS.sub("<address>", text)
    text = _TEMP_PATH.sub("<temp-path>", text)
    text = _LINE_NUMBER.sub(r"\1<line>", text)
    return " ".join(text.split())


def _frames(text: str, candidate_markers: Iterable[str]) -> list[str]:
    markers = tuple(marker.lower() for marker in candidate_markers)
    result: list[str] = []
    for match in _PY_FRAME.finditer(text):
        path = match.group("path").replace("\\", "/")
        if markers and not any(marker in path.lower() for marker in markers):
            continue
        result.append(f"{path.rsplit('/', 1)[-1]}:{match.group('func')}")
        if len(result) == 3:
            break
    return result


def fingerprint_failure(
    error_class: Any,
    evidence: str | None = None,
    *,
    violation_codes: Iterable[Any] = (),
    candidate_markers: Iterable[str] = ("solution/", "candidate", "src/"),
) -> str:
    """Return SHA-256 over a compact canonical representation of a failure."""
    text = evidence or ""
    exception = _EXCEPTION.search(text)
    payload = {
        "error_class": str(_value(error_class) or "unknown").lower(),
        "exception_type": exception.group(1) if exception else None,
        "candidate_frames": _frames(text, candidate_markers),
        "violation_codes": sorted(
            {str(_value(code)).strip().upper() for code in violation_codes if code}
        ),
        # A short normalized summary distinguishes errors with no structured data.
        "summary": normalize_text(text)[:300] if not exception else None,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_result(result: Any) -> str:
    """Fingerprint any supported run, gate, output, or no-op result."""
    supplied = getattr(result, "error_fingerprint", None)
    if supplied and re.fullmatch(r"[0-9a-fA-F]{64}", str(supplied)):
        return str(supplied).lower()
    violations = getattr(result, "violations", ()) or ()
    codes = [getattr(item, "code", item) for item in violations]
    checks = getattr(result, "checks", None)
    if isinstance(checks, dict):
        codes.extend(key for key, status in checks.items() if str(_value(status)).lower() == "fail")
    evidence = (
        getattr(result, "error_summary", None)
        or getattr(result, "trace_tail", None)
        or " ".join(str(getattr(item, "message", item)) for item in violations)
    )
    return fingerprint_failure(
        getattr(result, "error_class", None) or getattr(result, "outcome", None) or "gate_failure",
        evidence,
        violation_codes=codes,
    )
