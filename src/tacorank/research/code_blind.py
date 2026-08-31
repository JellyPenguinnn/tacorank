"""Shared detection and redaction for code-blind research narratives."""

from __future__ import annotations

import re
from typing import Any, Mapping

_CODE_EXTENSION = r"(?:py|pyi|js|ts|tsx|java|go|rs|cpp|cc|c|h)"

# Slash-separated research terms are not repository paths. This is true of one
# slash (``user/item``, ``positive/negative``) and equally of several
# (``train/valid/test``, ``smoke/proxy/full``, ``user/item/date``) -- the last
# two are this harness's own vocabulary for its splits and fidelities, and the
# reviewed model_compact_ranker card is written with the third. Counting
# separators therefore cannot distinguish prose from a path, and rejecting a
# plan for saying "the train/valid/test split" stops a run over nothing.
#
# A bare forward-slash chain is only matched when it also carries a path
# signal, which the remaining branches already supply: a source-file
# extension, an explicit relative/absolute prefix, or a known code root.
# Backslash chains keep matching on their own, since a Windows path separator
# does not occur in research prose.
IMPLEMENTATION_REFERENCE_RE = re.compile(
    r"(?:^|\s)(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.-]+\."
    + _CODE_EXTENSION
    + r"\b|"
    r"(?:^|\s)(?:\.\.?[/\\]|[/\\]|[A-Za-z]:\\)[A-Za-z0-9_.\\/-]+|"
    r"(?:^|\s)(?:[A-Za-z0-9_.-]+\\){2,}[A-Za-z0-9_.-]+|"
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
