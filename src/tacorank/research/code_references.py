"""Shared detection and redaction of code-specific planner references."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


_PATH_SEGMENT = r"[A-Za-z0-9_.-]*[A-Za-z0-9_-]"
_REFERENCE_PATTERNS = (
    (
        "source_path",
        re.compile(
            rf"(?<![A-Za-z0-9_.:<>/-])(?:"
            rf"(?:(?:\.{{1,2}}|~)?/)(?:{_PATH_SEGMENT}/)*{_PATH_SEGMENT}|"
            rf"(?:src|solution|tests?|scripts?|lib|app|configs?|docker|worktrees?)/"
            rf"(?:{_PATH_SEGMENT}/)*{_PATH_SEGMENT}"
            rf")",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "source_file",
        re.compile(
            r"\b[A-Za-z0-9_.-]+\.(?:py|pyi|js|ts|tsx|java|go|rs|cpp|cc|c|h)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "implementation_name",
        re.compile(
            r"\b(?:entrypoint|function name|class name|line number|source file)\b",
            flags=re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class CodeReference:
    category: str
    text: str


def find_code_reference(value: str) -> Optional[CodeReference]:
    """Return the earliest explicit code reference in text, if one exists."""

    matches = []
    for category, pattern in _REFERENCE_PATTERNS:
        match = pattern.search(value)
        if match is not None:
            matches.append((match.start(), category, match.group(0)))
    if not matches:
        return None
    _, category, text = min(matches, key=lambda item: item[0])
    return CodeReference(category=category, text=text)


def redact_code_references(
    value: str,
    replacement: str = "[implementation detail withheld]",
) -> str:
    """Redact explicit code references without removing scientific slash notation."""

    result = value
    for _, pattern in _REFERENCE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result
