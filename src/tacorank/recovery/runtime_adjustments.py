"""Selection of named, contract-approved runtime adjustments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

APPROVED_ADJUSTMENTS = (
    "batch_size",
    "num_workers",
    "mixed_precision",
    "timeout_profile",
)


@dataclass(frozen=True)
class RuntimeAdjustment:
    name: str
    value: Any

    def instruction(self) -> str:
        return f"Apply approved runtime adjustment {self.name}={self.value!r}; do not alter command text."


def _allowed(context: Any) -> Mapping[str, Any]:
    raw = getattr(context, "allowed_runtime_adjustments", None) or {}
    if isinstance(raw, Mapping):
        return raw
    result = {}
    for item in raw:
        if isinstance(item, str):
            result[item] = True
        elif isinstance(item, Mapping) and item.get("name"):
            result[str(item["name"])] = item.get("next_value", item.get("value", True))
        elif getattr(item, "name", None):
            result[str(item.name)] = getattr(item, "next_value", getattr(item, "value", True))
    return result


def _next_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("next_value", value.get("value"))
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _valid_value(name: str, value: Any) -> bool:
    if value is None:
        return False
    if name == "batch_size":
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if name == "num_workers":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if name == "mixed_precision":
        return isinstance(value, bool)
    if name == "timeout_profile":
        return isinstance(value, str) and bool(value.strip())
    return False


def select_runtime_adjustment(failure_class: str, context: Any) -> RuntimeAdjustment | None:
    """Choose only a named allowlisted next value; arbitrary shell fragments are impossible."""
    allowed = _allowed(context)
    preference = {
        "oom": ("batch_size", "num_workers", "mixed_precision"),
        "numerical_error": ("mixed_precision",),
        "timeout": ("timeout_profile",),
    }.get(failure_class, ())
    for name in preference:
        if name not in allowed or name not in APPROVED_ADJUSTMENTS:
            continue
        value = _next_value(allowed[name])
        if not _valid_value(name, value):
            continue
        current = getattr(context, "current_runtime_settings", {}).get(name)
        if value == current:
            continue
        history = getattr(context, "attempt_history", ()) or ()
        if any(
            item.get("action") == "adjust_approved_runtime_setting"
            and item.get("runtime_adjustments", {}).get(name) == value
            for item in history
            if isinstance(item, Mapping)
        ):
            continue
        return RuntimeAdjustment(name, value)
    return None
