"""Canonical JSON encoding used by the event hash chain."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from pydantic import BaseModel


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    return value


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def event_hash_input(envelope: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(envelope)
    copied.pop("event_hash", None)
    return copied
