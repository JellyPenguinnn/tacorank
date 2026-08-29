"""Stable duplicate keys for planner proposals."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .graph_view import get_value


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted((_normalize(item) for item in value), key=repr)
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.strip().lower())
        return re.sub(r"[^a-z0-9_.:/ -]", "", text)
    return value


def duplicate_payload(spec: Any) -> dict[str, Any]:
    """Return the schema-v1 identity fields for a materially same plan.

    The memory contract defines duplicate identity as normalized
    ``parent + family + change``.  Target-file and fidelity differences are
    implementation details and must not silently create a second experiment
    with the same research change.
    """

    return {
        "parent_commit_sha": get_value(spec, "parent_commit_sha", ""),
        "parent_experiment_id": get_value(spec, "parent_experiment_id", ""),
        "family": get_value(spec, "family", ""),
        "change_summary": get_value(spec, "change_summary", ""),
    }


def compute_duplicate_key(spec: Any) -> str:
    payload = _normalize(duplicate_payload(spec))
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class DuplicateDetector:
    def __init__(self, summaries: Sequence[Any] = ()):
        self._keys: set[str] = set()
        for summary in summaries:
            key = get_value(summary, "duplicate_key", None)
            if key:
                self._keys.add(str(key))

    def contains(self, spec: Any) -> bool:
        supplied = get_value(spec, "duplicate_key", None)
        key = str(supplied) if supplied else compute_duplicate_key(spec)
        return key in self._keys

    def add(self, spec: Any) -> str:
        key = compute_duplicate_key(spec)
        self._keys.add(key)
        return key

    def validate(self, spec: Any) -> bool:
        supplied = get_value(spec, "duplicate_key", None)
        return not supplied or str(supplied) == compute_duplicate_key(spec)
