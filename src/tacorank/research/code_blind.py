"""Shared detection and redaction for code-blind research narratives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

_CODE_EXTENSION = r"(?:py|pyi|js|ts|tsx|java|go|rs|cpp|cc|c|h)"
_PATH_SEGMENT = r"[A-Za-z0-9_.-]*[A-Za-z0-9_-]"

# A single slash between plain research terms (for example ``user/item`` or
# ``positive/negative``) is not a repository path.  Real implementation
# references still match when they are explicit/absolute paths, contain at
# least two path separators, start with a common code root, or name a source
# file by extension.
_REFERENCE_PATTERNS = (
    (
        "source_path",
        re.compile(
            rf"(?<![A-Za-z0-9_.:<>/\\-])(?:"
            rf"(?:(?:\.{{1,2}}|~)?[/\\]|[A-Za-z]:\\)"
            rf"(?:{_PATH_SEGMENT}[/\\])*{_PATH_SEGMENT}|"
            rf"(?:{_PATH_SEGMENT}[/\\]){{2,}}{_PATH_SEGMENT}|"
            rf"(?:src|source|solution|app|lib|tests?|scripts?|configs?|contract|docs?)"
            rf"[/\\](?:{_PATH_SEGMENT}[/\\])*{_PATH_SEGMENT}"
            rf")",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "source_file",
        re.compile(
            r"\b[A-Za-z0-9_.-]+\." + _CODE_EXTENSION + r"\b",
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

IMPLEMENTATION_REFERENCE_RE = re.compile(
    "|".join("(?:%s)" % pattern.pattern for _, pattern in _REFERENCE_PATTERNS),
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ImplementationReference:
    category: str
    text: str


def find_implementation_reference(value: object) -> Optional[ImplementationReference]:
    """Return the earliest concrete implementation reference, if present."""

    text = str(value)
    matches = []
    for category, pattern in _REFERENCE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            matches.append((match.start(), category, match.group(0)))
    if not matches:
        return None
    _, category, match_text = min(matches, key=lambda item: item[0])
    return ImplementationReference(category=category, text=match_text)


def contains_implementation_reference(value: object) -> bool:
    """Return whether narrative text contains a concrete code reference."""

    return find_implementation_reference(value) is not None


def redact_implementation_references(value: Any) -> Any:
    """Recursively remove concrete code references from model-visible data."""

    if isinstance(value, str):
        result = value
        for _, pattern in _REFERENCE_PATTERNS:
            result = pattern.sub("[implementation detail withheld]", result)
        return result
    if isinstance(value, Mapping):
        return {
            str(key): redact_implementation_references(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_implementation_references(item) for item in value]
    return value
