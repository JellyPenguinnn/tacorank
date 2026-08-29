"""Validated machine-readable control block embedded in the research playbook."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


REQUIRED_RULE_ORDER = (
    "output_rejected",
    "suspicious_or_compromised",
    "no_op",
    "unstable",
    "promotion_required",
    "non_public_or_incomplete",
    "pairwise_gauc_up_ndcg_down",
    "pairwise_gauc_down_ndcg_up",
    "pairwise_both_up",
    "meaningful_no_gain",
    "trusted_improvement",
    "trusted_regression",
)
SUPPORTED_RULES = frozenset(REQUIRED_RULE_ORDER)


class PlaybookError(ValueError):
    """Raised when the human-reviewed playbook control block is malformed."""


@dataclass(frozen=True)
class ImprovementPlaybook:
    schema_version: str
    source_path: str
    source_sha256: str
    rule_order: tuple[str, ...]
    family_order: tuple[str, ...]
    method_order: Mapping[str, tuple[str, ...]]

    def methods_for(self, family: str) -> tuple[str, ...]:
        return self.method_order.get(family, ())


def load_improvement_playbook(path: str | Path, *, source_path: str | None = None) -> ImprovementPlaybook:
    """Load the first JSON fence and fail closed on unsupported control data."""

    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    match = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.DOTALL)
    if match is None:
        raise PlaybookError("improvement playbook is missing its JSON control block")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise PlaybookError("improvement playbook JSON is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise PlaybookError("improvement playbook schema_version must be 1.0")

    def unique_strings(name: str) -> tuple[str, ...]:
        value = payload.get(name)
        if not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise PlaybookError("%s must be a non-empty string list" % name)
        normalized = tuple(item.strip() for item in value)
        if len(normalized) != len(set(normalized)):
            raise PlaybookError("%s must not contain duplicates" % name)
        return normalized

    rule_order = unique_strings("rule_order")
    unsupported = set(rule_order) - SUPPORTED_RULES
    missing = SUPPORTED_RULES - set(rule_order)
    if unsupported:
        raise PlaybookError(
            "unsupported playbook rules: %s" % ", ".join(sorted(unsupported))
        )
    if missing:
        raise PlaybookError(
            "missing mandatory playbook rules: %s" % ", ".join(sorted(missing))
        )
    if rule_order != REQUIRED_RULE_ORDER:
        raise PlaybookError("playbook rules must preserve the mandatory safety order")
    family_order = unique_strings("family_order")
    raw_methods = payload.get("method_order")
    if not isinstance(raw_methods, dict):
        raise PlaybookError("method_order must be an object")
    method_order = {}
    for family, methods in raw_methods.items():
        if family not in family_order or not isinstance(methods, list) or not methods:
            raise PlaybookError("method_order contains an invalid family or method list")
        normalized = tuple(str(item).strip() for item in methods)
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise PlaybookError("method_order entries must be unique non-empty strings")
        method_order[str(family)] = normalized
    if method_order.get("objective", (None,))[0] != "objective_pairwise_bpr":
        raise PlaybookError("objective_pairwise_bpr must be the first objective method")
    return ImprovementPlaybook(
        schema_version="1.0",
        source_path=source_path or path.as_posix(),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        rule_order=rule_order,
        family_order=family_order,
        method_order=method_order,
    )
