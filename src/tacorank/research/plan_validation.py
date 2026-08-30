"""Pure validation for Person 1 planner outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Any

from .duplicate_detection import DuplicateDetector, compute_duplicate_key
from .graph_view import (
    ExperimentNodeView,
    GraphView,
    as_list,
    enum_value,
    get_value,
    has_value,
)
from .method_eligibility import evaluate_method_card, method_card_map
from .search_eligibility import classify_search_eligibility


HIDDEN_PATTERNS = (
    "hidden test",
    "hidden_test",
    "test label",
    "private label",
    "secret label",
    "ground truth test",
)

SHARED_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
EVENT_ID_PATTERN = re.compile(r"evt_\d{6,}$")
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


def _normalized_enum(value: Any) -> str:
    return str(enum_value(value) or "").strip().lower()


def _normalized_path(path: Any) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return None
    path = path.replace("\\", "/").strip()
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\x00" in path:
        return None
    return str(pure)


def _path_is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


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

        if str(get_value(context, "schema_version", "1.0")) != "1.0":
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

        if run_id and not SHARED_ID_PATTERN.fullmatch(str(run_id)):
            errors.append("INVALID_RUN_ID")
        if not SHARED_ID_PATTERN.fullmatch(str(get_value(spec, "experiment_id", ""))):
            errors.append("INVALID_EXPERIMENT_ID")
        if not SHARED_ID_PATTERN.fullmatch(str(get_value(spec, "parent_experiment_id", ""))):
            errors.append("INVALID_PARENT_EXPERIMENT_ID")
        if context_id and not SHARED_ID_PATTERN.fullmatch(str(context_id)):
            errors.append("INVALID_CONTEXT_ID")
        if not _valid_commit(get_value(spec, "parent_commit_sha", None)):
            errors.append("INVALID_PARENT_COMMIT_SHA")

        family = str(get_value(spec, "family", ""))
        allowed = get_value(contract, "allowed_families", None)
        if allowed is None:
            allowed = get_value(contract, "experiment_families", None)
        legal_families = set(map(str, as_list(allowed)))
        if not legal_families:
            errors.append("CONTRACT_ALLOWED_FAMILIES_MISSING")
        if family not in legal_families:
            errors.append("ILLEGAL_EXPERIMENT_FAMILY")
        if not as_list(get_value(contract, "allowed_data", None)):
            errors.append("CONTRACT_ALLOWED_DATA_MISSING")
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
        historical_summaries = as_list(get_value(context, "family_history", None))
        historical_by_id = {
            str(get_value(summary, "experiment_id", "")): summary
            for summary in historical_summaries
            if str(get_value(summary, "experiment_id", ""))
        }
        refinement_ids = (
            {
                str(item)
                for item in as_list(
                    get_value(context, "refinement_frontier_ids", None)
                )
            }
            if has_value(context, "refinement_frontier_ids")
            else {
                experiment_id
                for experiment_id, summary in historical_by_id.items()
                if classify_search_eligibility(
                    summary, context
                ).refinement_eligible
            }
        )
        refinement_by_id = {
            experiment_id: historical_by_id[experiment_id]
            for experiment_id in refinement_ids
            if experiment_id in historical_by_id
        }
        ensemble_ids = (
            {
                str(item)
                for item in as_list(
                    get_value(context, "ensemble_candidate_ids", None)
                )
            }
            if has_value(context, "ensemble_candidate_ids")
            else {
                experiment_id
                for experiment_id, summary in historical_by_id.items()
                if classify_search_eligibility(summary, context).ensemble_eligible
            }
        )
        ensemble_by_id = {
            experiment_id: historical_by_id[experiment_id]
            for experiment_id in ensemble_ids
            if experiment_id in historical_by_id
        }
        choice_phase = str(get_value(choice, "phase", "")) if choice is not None else ""
        if parent is None and choice_phase == "refinement" and parent_id:
            summary = refinement_by_id.get(str(parent_id))
            if (
                summary is not None
                and classify_search_eligibility(summary, context).refinement_eligible
            ):
                parent = ExperimentNodeView.from_summary(summary)
        if parent is None:
            errors.append("UNKNOWN_PARENT")
        elif not parent.is_parent_eligible and choice_phase != "refinement":
            errors.append("INELIGIBLE_PARENT")
        elif not parent.is_parent_eligible:
            summary = refinement_by_id.get(parent.experiment_id)
            if (
                summary is None
                or not classify_search_eligibility(summary, context).refinement_eligible
            ):
                errors.append("INELIGIBLE_REFINEMENT_PARENT")
        if (
            parent is not None
            and get_value(spec, "parent_commit_sha", None)
            and parent.parent_commit_sha
            and get_value(spec, "parent_commit_sha") != parent.parent_commit_sha
        ):
            # The summary's commit_sha represents the code state being branched from.
            errors.append("PARENT_COMMIT_MISMATCH")

        component_ids = [
            str(item)
            for item in as_list(get_value(spec, "component_experiment_ids", None))
        ]
        if len(component_ids) != len(set(component_ids)):
            errors.append("DUPLICATE_COMPONENT_EXPERIMENT")
        if any(not SHARED_ID_PATTERN.fullmatch(item) for item in component_ids):
            errors.append("INVALID_COMPONENT_EXPERIMENT_ID")
        required_components = tuple(
            str(item)
            for item in as_list(
                get_value(choice, "component_experiment_ids", None)
                if choice is not None
                else None
            )
        )
        if required_components and tuple(component_ids) != required_components:
            errors.append("COMPONENT_POLICY_MISMATCH")
        if family != "ensemble" and component_ids:
            errors.append("COMPONENTS_REQUIRE_ENSEMBLE_FAMILY")
        if family == "ensemble" and not component_ids:
            errors.append("ENSEMBLE_COMPONENT_REQUIRED")
        component_method_ids = {
            str(item)
            for item in as_list(get_value(spec, "method_card_ids", None))
        }
        residual_ensemble = (
            "ensemble_diverse_residual_candidate" in component_method_ids
        )
        for component_id in component_ids:
            if component_id == str(parent_id):
                errors.append("ENSEMBLE_COMPONENT_DUPLICATES_PARENT")
                continue
            component = historical_by_id.get(component_id)
            if component is None:
                errors.append("UNKNOWN_COMPONENT_EXPERIMENT")
                continue
            search = classify_search_eligibility(component, context)
            soft_authorized = component_id in ensemble_by_id
            if residual_ensemble and not (
                soft_authorized and search.ensemble_eligible
            ):
                errors.append("INELIGIBLE_ENSEMBLE_COMPONENT")
            elif not residual_ensemble and not (
                (soft_authorized and search.ensemble_eligible)
                or search.branch_eligible
            ):
                errors.append("INELIGIBLE_ENSEMBLE_COMPONENT")

        target_files = as_list(get_value(spec, "target_files", None))
        normalized_files = [_normalized_path(path) for path in target_files]
        if not target_files:
            errors.append("NO_TARGET_FILES")
        if any(path is None for path in normalized_files):
            errors.append("INVALID_TARGET_PATH")
        if len(set(normalized_files)) != len(normalized_files):
            errors.append("DUPLICATE_TARGET_PATH")

        protected = {
            str(item).rstrip("/")
            for item in (get_value(contract, "protected_paths", []) or [])
        }
        raw_editable = as_list(get_value(contract, "editable_paths", None))
        editable = [
            _normalized_path(str(item).rstrip("/")) for item in raw_editable
        ]
        if not raw_editable:
            errors.append("CONTRACT_EDITABLE_PATHS_MISSING")
        elif any(path is None for path in editable):
            errors.append("INVALID_EDITABLE_PATH")
        editable_roots = [path for path in editable if path is not None]
        for path in normalized_files:
            if path is None:
                continue
            if any(_path_is_within(path, item) for item in protected):
                errors.append("PROTECTED_TARGET_PATH")
            if not any(_path_is_within(path, root) for root in editable_roots):
                errors.append("TARGET_OUTSIDE_EDITABLE_PATHS")

        raw_interfaces = get_value(context, "target_interface_excerpts", None)
        try:
            interface_items = list(dict(raw_interfaces or {}).items())
        except (TypeError, ValueError):
            interface_items = []
            errors.append("INVALID_TARGET_INTERFACES")
        interface_paths: set[str] = set()
        if not interface_items:
            errors.append("TARGET_INTERFACES_MISSING")
        for raw_path, excerpt in interface_items:
            path = _normalized_path(raw_path)
            if path is None or not _nonempty(excerpt):
                errors.append("INVALID_TARGET_INTERFACE")
                continue
            interface_paths.add(path)
        normalized_target_set = {
            path for path in normalized_files if path is not None
        }
        if (
            normalized_target_set
            and interface_paths
            and normalized_target_set.isdisjoint(interface_paths)
        ):
            errors.append("TARGET_INTERFACE_NOT_TOUCHED")

        fidelity_plan = [
            _normalized_enum(item)
            for item in as_list(get_value(spec, "fidelity_plan", None))
        ]
        allowed_fidelity = {"smoke", "proxy", "full"}
        fidelity_is_valid = bool(fidelity_plan) and all(
            item in allowed_fidelity for item in fidelity_plan
        )
        if not fidelity_is_valid:
            errors.append("INVALID_FIDELITY_PLAN")
        else:
            if len(fidelity_plan) != len(set(fidelity_plan)):
                errors.append("DUPLICATE_FIDELITY")
            order = {"smoke": 0, "proxy": 1, "full": 2}
            if any(
                order[fidelity_plan[index]] >= order[fidelity_plan[index + 1]]
                for index in range(len(fidelity_plan) - 1)
            ):
                errors.append("NON_MONOTONIC_FIDELITY_PLAN")
        if "full" not in fidelity_plan and str(family) == "ensemble":
            warnings.append("ENSEMBLE_WITHOUT_FULL_FIDELITY")

        raw_method_ids = list(
            map(str, as_list(get_value(spec, "method_card_ids", None)))
        )
        method_ids = set(raw_method_ids)
        if not raw_method_ids:
            errors.append("METHOD_CARD_REQUIRED")
        if len(raw_method_ids) != len(method_ids):
            errors.append("DUPLICATE_METHOD_CARD")
        required_method_id = get_value(choice, "method_card_id", None)
        if required_method_id and method_ids != {str(required_method_id)}:
            errors.append("METHOD_POLICY_MISMATCH")
        cards = method_card_map(context)
        if not cards:
            errors.append("CONTEXT_METHOD_CARDS_MISSING")
        for method_id in sorted(method_ids):
            card = cards.get(method_id)
            if card is None:
                errors.append("UNKNOWN_METHOD_CARD")
                continue
            eligibility = evaluate_method_card(card, context, family=family)
            errors.extend(eligibility.reasons)
            implementation_targets = [
                _normalized_path(item)
                for item in as_list(
                    get_value(card, "implementation_targets", None)
                )
            ]
            if any(item is None for item in implementation_targets):
                errors.append("METHOD_IMPLEMENTATION_TARGET_INVALID")
                continue
            required_targets = {
                item for item in implementation_targets if item is not None
            }
            if required_targets and not required_targets.issubset(interface_paths):
                errors.append("METHOD_IMPLEMENTATION_TARGET_UNAUTHORIZED")
            if required_targets and normalized_target_set.isdisjoint(required_targets):
                errors.append("METHOD_IMPLEMENTATION_TARGET_NOT_TOUCHED")

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
        if any(component_id.lower() not in text for component_id in component_ids):
            errors.append("ENSEMBLE_COMPONENT_NOT_DESCRIBED")

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
        cost_tier = _normalized_enum(get_value(cost, "cost_tier", ""))
        if cost_tier not in {"low", "medium", "high"}:
            errors.append("INVALID_COST_TIER")
        else:
            cost_order = {"low": 0, "medium": 1, "high": 2}
            for method_id in method_ids:
                card = cards.get(method_id)
                if card is None:
                    continue
                method_cost = _normalized_enum(get_value(card, "cost_tier", ""))
                if method_cost in cost_order and cost_order[cost_tier] < cost_order[method_cost]:
                    errors.append("METHOD_COST_UNDERESTIMATED")
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
