"""Shared detection and redaction for code-blind research narratives."""

from __future__ import annotations

import re
from typing import Any, Mapping

_CODE_EXTENSION = r"(?:py|pyi|js|ts|tsx|java|go|rs|cpp|cc|c|h)"

# A single slash between plain research terms (for example ``user/item`` or
# ``positive/negative``) is not a repository path.  Real implementation
# references still match when they are explicit/absolute paths, contain at
# least two path separators, start with a common code root, or name a source
# file by extension.
IMPLEMENTATION_REFERENCE_RE = re.compile(
    r"(?:^|\s)(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.-]+\."
    + _CODE_EXTENSION
    + r"\b|"
    r"(?:^|\s)(?:\.\.?[/\\]|[/\\]|[A-Za-z]:\\)[A-Za-z0-9_.\\/-]+|"
    r"(?:^|\s)(?:[A-Za-z0-9_.-]+[/\\]){2,}[A-Za-z0-9_.-]+|"
    r"(?:^|\s)(?:src|source|solution|app|lib|tests?|scripts?|config|contract|docs?)"
    r"[/\\][A-Za-z0-9_.-]+|"
    r"\b[A-Za-z0-9_.-]+\."
    + _CODE_EXTENSION
    + r"\b|"
    r"\b(?:entrypoint|function name|class name|line number|source file)\b",
    flags=re.IGNORECASE,
)


def contains_implementation_reference(value: object) -> bool:
    """Return whether narrative text contains a concrete code reference."""

    return bool(IMPLEMENTATION_REFERENCE_RE.search(str(value)))


def redact_implementation_references(value: Any) -> Any:
    """Recursively remove concrete code references from model-visible data."""

    if isinstance(value, str):
        return IMPLEMENTATION_REFERENCE_RE.sub(
            "[implementation detail withheld]", value
        )
    if isinstance(value, Mapping):
        return {
            str(key): redact_implementation_references(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_implementation_references(item) for item in value]
    return value
