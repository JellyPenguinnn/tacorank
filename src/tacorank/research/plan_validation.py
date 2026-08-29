"""Pure validation for Person 1 planner outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Any

from .duplicate_detection import DuplicateDetector, compute_duplicate_key
from .graph_view import GraphView, as_list, get_value
from .portfolio import ALL_FAMILIES


HIDDEN_PATTERNS = (
    "hidden test",
    "hidden_test",
    "test label",
    "private label",
    "secret label",
    "ground truth test",
)

RUN_ID_PATTERN = re.compile(r"run_\d{8}_[a-z0-9][a-z0-9_-]*$")
EVENT_ID_PATTERN = re.compile(r"evt_\d{6}$")
EXPERIMENT_ID_PATTERN = re.compile(r"exp_\d{4}$")
CONTEXT_ID_PATTERN = re.compile(r"ctx_(planner|coder|recovery|evaluator)_\d{6}$")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?$")


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _nonempty(value: Any) -> bool:
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return bool(value)
    return value is not None and bool(str(value).strip())


def _valid_commit(value: Any) -> bool:
    return bool(value and COMMIT_PATTERN.fullmatch(str(value)))


def _normalized_path(path: Any) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return None
    path = path.replace("\\", "/").strip()
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\x00" in path:
        return None
    return str(pure)


def _budget_value(budget: Any, *names: str) -> float | None:
    for name in names:
        value = get_value(budget, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


class PlanValidator:
    """Validate a proposed ExperimentSpec against verified context only."""

    def validate(
        self,
        spec: Any,
        context: Any,
        choice: Any | None = None,
        duplicate_detector: DuplicateDetector | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        context_id = get_value(context, "context_id", None)
        contract_hash = get_value(context, "contract_sha256", None)
        contract = get_value(context, "contract_summary", None)
        if (
            not _nonempty(contract_hash)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", str(contract_hash))
            or get_value(contract, "resolved", True) is False
        ):
            errors.append("CONTRACT_UNRESOLVED")

        if str(get_value(context, "schema_version", "")) != "1.0":
            errors.append("CONTEXT_SCHEMA_VERSION_MISMATCH")

        for field in (
            "schema_version",
            "run_id",
            "experiment_id",
            "parent_experiment_id",
            "parent_commit_sha",
            "context_id",
            "hypothesis",
            "family",
            "change_summary",
            "target_stage",
            "expected_mechanism",
            "falsification_condition",
            "success_criteria",
            "estimated_cost",
            "duplicate_key",
        ):
            if not _nonempty(get_value(spec, field, None)):
                errors.append(f"MISSING_{field.upper()}")

        if str(get_value(spec, "schema_version", "")) != "1.0":
            errors.append("SCHEMA_VERSION_MISMATCH")
        if context_id and get_value(spec, "context_id", None) != context_id:
            errors.append("CONTEXT_ID_MISMATCH")
        run_id = get_value(context, "run_id", None)
        if run_id and get_value(spec, "run_id", None) != run_id:
            errors.append("RUN_ID_MISMATCH")

        if run_id and not RUN_ID_PATTERN.fullmatch(str(run_id)):
            errors.append("INVALID_RUN_ID")
        if not EXPERIMENT_ID_PATTERN.fullmatch(str(get_value(spec, "experiment_id", ""))):
            errors.append("INVALID_EXPERIMENT_ID")
        if not EXPERIMENT_ID_PATTERN.fullmatch(str(get_value(spec, "parent_experiment_id", ""))):
            errors.append("INVALID_PARENT_EXPERIMENT_ID")
        if context_id and not CONTEXT_ID_PATTERN.fullmatch(str(context_id)):
            errors.append("INVALID_CONTEXT_ID")
        if not _valid_commit(get_value(spec, "parent_commit_sha", None)):
            errors.append("INVALID_PARENT_COMMIT_SHA")

        family = str(get_value(spec, "family", ""))
        allowed = get_value(contract, "allowed_families", None) or get_value(
            contract, "experiment_families", None
        )
        legal_families = set(map(str, as_list(allowed))) if allowed is not None else set(ALL_FAMILIES)
        if family not in legal_families:
            errors.append("ILLEGAL_EXPERIMENT_FAMILY")
        if choice is not None:
            if get_value(choice, "parent", None) is not None and get_value(
                spec, "parent_experiment_id", None
            ) != get_value(get_value(choice, "parent"), "experiment_id", None):
                errors.append("PARENT_POLICY_MISMATCH")
            if get_value(choice, "family", None) and family != get_value(choice, "family"):
                errors.append("FAMILY_POLICY_MISMATCH")

        graph = GraphView.from_context(context)
        parent_id = get_value(spec, "parent_experiment_id", None)
        parent = graph.get(str(parent_id)) if parent_id else None
        if parent is None:
            errors.append("UNKNOWN_PARENT")
        elif not parent.is_parent_eligible:
            errors.append("INELIGIBLE_PARENT")
        elif get_value(spec, "parent_commit_sha", None) and parent.parent_commit_sha:
            # The summary's commit_sha represents the code state being branched from.
            if get_value(spec, "parent_commit_sha") != parent.parent_commit_sha:
                errors.append("PARENT_COMMIT_MISMATCH")

        target_files = as_list(get_value(spec, "target_files", None))
        normalized_files = [_normalized_path(path) for path in target_files]
        if not target_files:
            errors.append("NO_TARGET_FILES")
        if any(path is None for path in normalized_files):
            errors.append("INVALID_TARGET_PATH")
        if len(set(normalized_files)) != len(normalized_files):
            errors.append("DUPLICATE_TARGET_PATH")

        protected = set(map(str, get_value(contract, "protected_paths", []) or []))
        editable = get_value(contract, "editable_paths", None)
        for path in normalized_files:
            if path is None:
                continue
            if path in protected:
                errors.append("PROTECTED_TARGET_PATH")
            if editable and not any(path == item or path.startswith(f"{item.rstrip('/')}/") for item in editable):
                errors.append("TARGET_OUTSIDE_EDITABLE_PATHS")

        fidelity_plan = [str(item) for item in as_list(get_value(spec, "fidelity_plan", None))]
        allowed_fidelity = {"smoke", "proxy", "full"}
        if not fidelity_plan or any(item not in allowed_fidelity for item in fidelity_plan):
            errors.append("INVALID_FIDELITY_PLAN")
        order = {"smoke": 0, "proxy": 1, "full": 2}
        if any(order[fidelity_plan[index]] > order[fidelity_plan[index + 1]] for index in range(len(fidelity_plan) - 1)):
            errors.append("NON_MONOTONIC_FIDELITY_PLAN")
        if "full" not in fidelity_plan and str(family) == "ensemble":
            warnings.append("ENSEMBLE_WITHOUT_FULL_FIDELITY")

        method_ids = set(map(str, as_list(get_value(spec, "method_card_ids", None))))
        known_method_ids = {
            str(get_value(card, "method_id", ""))
            for card in as_list(get_value(context, "method_cards", None))
        }
        if method_ids and known_method_ids and not method_ids.issubset(known_method_ids):
            errors.append("UNKNOWN_METHOD_CARD")

        source_events = set(map(str, as_list(get_value(context, "source_event_ids", None))))
        if any(not EVENT_ID_PATTERN.fullmatch(event_id) for event_id in source_events):
            errors.append("INVALID_CONTEXT_EVENT_ID")
        evidence_events = set(map(str, as_list(get_value(spec, "evidence_event_ids", None))))
        if any(not EVENT_ID_PATTERN.fullmatch(event_id) for event_id in evidence_events):
            errors.append("INVALID_EVIDENCE_EVENT_ID")
        if not evidence_events.issubset(source_events):
            errors.append("EVIDENCE_OUTSIDE_CONTEXT")

        text = " ".join(
            str(get_value(spec, field, ""))
            for field in ("hypothesis", "change_summary", "expected_mechanism", "falsification_condition")
        ).lower()
        if any(pattern in text for pattern in HIDDEN_PATTERNS):
            errors.append("HIDDEN_TEST_REFERENCE")

        supplied_duplicate_key = get_value(spec, "duplicate_key", None)
        if duplicate_detector is not None and not duplicate_detector.validate(spec):
            errors.append("DUPLICATE_KEY_MISMATCH")
        elif supplied_duplicate_key != compute_duplicate_key(spec):
            errors.append("DUPLICATE_KEY_MISMATCH")

        known_summaries: list[Any] = []
        for field in ("baseline", "current_best", "eligible_frontier", "family_history"):
            known_summaries.extend(as_list(get_value(context, field, None)))
        seen_detector = duplicate_detector or DuplicateDetector(known_summaries)
        if seen_detector.contains(spec):
            errors.append("DUPLICATE_EXPERIMENT")

        cost = get_value(spec, "estimated_cost", None)
        cost_tier = str(get_value(cost, "cost_tier", "")).lower()
        if cost_tier not in {"low", "medium", "high"}:
            errors.append("INVALID_COST_TIER")
        budget = get_value(context, "remaining_budget", None) or get_value(
            context, "remaining_budgets", None
        )
        for estimate_names, budget_names, code in (
            (("llm_tokens_upper_bound", "llm_tokens_upper_bound"), ("remaining_llm_tokens", "llm_tokens"), "TOKEN_BUDGET_EXCEEDED"),
            (("wall_time_seconds_upper_bound", "wall_time_seconds"), ("remaining_wall_time_seconds", "wall_time_seconds"), "WALL_TIME_BUDGET_EXCEEDED"),
            (("gpu_seconds_upper_bound", "gpu_seconds"), ("remaining_gpu_seconds", "gpu_seconds"), "GPU_BUDGET_EXCEEDED"),
        ):
            estimate = _budget_value(cost, *estimate_names)
            remaining = _budget_value(budget, *budget_names)
            if estimate is not None and estimate < 0:
                errors.append("NEGATIVE_COST_ESTIMATE")
            if estimate is not None and remaining is not None and estimate > remaining:
                errors.append(code)

        return ValidationResult(accepted=not errors, errors=tuple(dict.fromkeys(errors)), warnings=tuple(dict.fromkeys(warnings)))
