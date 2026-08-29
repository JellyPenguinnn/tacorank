"""Redaction applied before context persistence or provider calls."""

from __future__ import annotations

import re
from typing import Iterable, Tuple


_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"(?im)\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*([^\s,;]+)"
    ),
)


def redact(text: str) -> Tuple[str, int]:
    redacted = text
    total = 0
    for pattern in _PATTERNS:
        redacted, count = pattern.subn("[REDACTED]", redacted)
        total += count
    return redacted, total
