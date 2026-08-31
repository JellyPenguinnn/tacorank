"""Pure validation for Person 1 planner outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from ..schemas import LiteratureEvidence

from .code_blind import contains_implementation_reference
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
from .portfolio import (
    COMPOSITION_DISTILLATION_OBJECTIVES,
    COMPOSITION_FEATURE_METHODS,
    COMPOSITION_HIDDEN_UNIT_ADAPTERS,
    COMPOSITION_INTEREST_METHODS,
    COMPOSITION_LOSS_ALIGNED_REFINEMENTS,
    COMPOSITION_MULTITASK_BACKBONES,
    COMPOSITION_OPTIONAL_ADDONS,
    COMPOSITION_PRIMARY_OBJECTIVES,
    COMPOSITION_SINGLE_TASK_BACKBONES,
)
from .search_eligibility import classify_search_eligibility

HIDDEN_PATTERNS = (
    "hidden test",
    "hidden_test",
    "test label",
    "private label",
    "secret label",
    "ground truth test",
)


def _is_no_op_summary(summary: Any, context: Any) -> bool:
    if str(enum_value(get_value(summary, "trust_verdict", ""))).lower() == "no_op":
        return True
    change = get_value(summary, "prediction_change", None)
    if change is not None and not isinstance(change, (int, float)):
        change = get_value(change, "changed_row_fraction", None)
    try:
        numeric_change = None if change is None else float(change)
    except (TypeError, ValueError):
        return False
    threshold = get_value(
        get_value(context, "contract_summary", None),
        "prediction_change_no_op_threshold",
        0.001,
    )
    try:
        numeric_threshold = float(threshold)
    except (TypeError, ValueError):
        numeric_threshold = 0.001
    return numeric_change is not None and numeric_change <= numeric_threshold


def _authorized_no_op_reimplementation(
    spec: Any, context: Any, choice: Any
) -> bool:
    """Allow one policy-selected duplicate mechanism after its first no-op."""

    if (
        str(get_value(choice, "phase", "")) != "no_op_reimplementation"
        or str(get_value(choice, "reason_code", ""))
        != "NO_OP_REIMPLEMENT_MECHANISM"
    ):
        return False
    history = as_list(get_value(context, "family_history", None))
    if not history:
        return False
    latest = history[-1]
    parent = get_value(choice, "parent", None)
    parent_id = str(get_value(latest, "parent_experiment_id", ""))
    method_ids = {
        str(item) for item in as_list(get_value(latest, "method_card_ids", None))
    }
    if (
        not _is_no_op_summary(latest, context)
        or str(get_value(latest, "status", "")).lower() != "no_op"
        or str(get_value(parent, "experiment_id", "")) != parent_id
        or str(get_value(spec, "parent_experiment_id", "")) != parent_id
        or str(get_value(spec, "family", ""))
        != str(get_value(latest, "family", ""))
        or set(map(str, as_list(get_value(spec, "method_card_ids", None))))
        != method_ids
        or len(method_ids) != 1
    ):
        return False
    matching_no_ops = 0
    for summary in history:
        if (
            _is_no_op_summary(summary, context)
            and str(get_value(summary, "parent_experiment_id", "")) == parent_id
            and str(get_value(summary, "family", ""))
            == str(get_value(latest, "family", ""))
            and method_ids.intersection(
                map(str, as_list(get_value(summary, "method_card_ids", None)))
            )
        ):
            matching_no_ops += 1
    return matching_no_ops == 1

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


def _budget_value(budget: Any, *names: str) -> float | None:
    for name in names:
        value = get_value(budget, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _literature_snapshot(value: Any) -> dict[str, Any] | None:
    try:
        return LiteratureEvidence.model_validate(value).model_dump(
            mode="json", exclude_none=False
        )
    except (TypeError, ValueError):
        return None


class PlanValidator:
    """Validate a code-blind research proposal against verified context only."""

    def validate(
        self,
        spec: Any,
        context: Any,
        choice: Any | None = None,
        duplicate_detector: DuplicateDetector | None = None,
        literature_evidence: Sequence[Any] = (),
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

        if any(
            has_value(spec, field)
            for field in ("target_stage", "target_files", "fidelity_plan")
        ):
            errors.append("PLANNER_IMPLEMENTATION_DETAIL_FORBIDDEN")

        raw_method_ids = list(
            map(str, as_list(get_value(spec, "method_card_ids", None)))
        )
        method_ids = set(raw_method_ids)
        if not raw_method_ids:
            errors.append("METHOD_CARD_REQUIRED")
        if len(raw_method_ids) != len(method_ids):
            errors.append("DUPLICATE_METHOD_CARD")
        required_method_ids = tuple(
            str(item)
            for item in as_list(
                get_value(choice, "method_card_ids", None)
                if choice is not None
                else None
            )
            if str(item)
        )
        if not required_method_ids:
            required_method_id = get_value(choice, "method_card_id", None)
            if required_method_id:
                required_method_ids = (str(required_method_id),)
        if required_method_ids and tuple(raw_method_ids) != required_method_ids:
            errors.append("METHOD_POLICY_MISMATCH")

        is_composition = family == "composition"
        if is_composition:
            enabled = get_value(contract, "aggressive_composition_enabled", False)
            if enabled is not True and str(enabled).strip().lower() != "true":
                errors.append("COMPOSITION_DISABLED")
            raw_limit = get_value(contract, "max_composed_methods", 12)
            try:
                max_methods = max(2, min(12, int(raw_limit)))
            except (TypeError, ValueError):
                max_methods = 12
            if len(raw_method_ids) < 3:
                errors.append("COMPOSITION_REQUIRES_OBJECTIVE_FEATURE_MODEL")
            if len(raw_method_ids) > max_methods:
                errors.append("COMPOSITION_TOO_LARGE")
        elif len(raw_method_ids) > 1:
            errors.append("COMPOSITION_FAMILY_REQUIRED")

        cards = method_card_map(context)
        if not cards:
            errors.append("CONTEXT_METHOD_CARDS_MISSING")
        known_composition_ids = {
            method_id
            for group in (
                COMPOSITION_PRIMARY_OBJECTIVES,
                COMPOSITION_DISTILLATION_OBJECTIVES,
                COMPOSITION_LOSS_ALIGNED_REFINEMENTS,
                COMPOSITION_FEATURE_METHODS,
                COMPOSITION_INTEREST_METHODS,
                COMPOSITION_SINGLE_TASK_BACKBONES,
                COMPOSITION_HIDDEN_UNIT_ADAPTERS,
                COMPOSITION_MULTITASK_BACKBONES,
                COMPOSITION_OPTIONAL_ADDONS,
            )
            for method_id in group
        }
        if is_composition:
            if len(method_ids.intersection(COMPOSITION_PRIMARY_OBJECTIVES)) != 1:
                errors.append("COMPOSITION_PRIMARY_OBJECTIVE_REQUIRED")
            if not method_ids.intersection(COMPOSITION_FEATURE_METHODS):
                errors.append("COMPOSITION_FEATURE_RESIDUAL_REQUIRED")
            single_backbones = method_ids.intersection(
                COMPOSITION_SINGLE_TASK_BACKBONES
            )
            multitask_backbones = method_ids.intersection(
                COMPOSITION_MULTITASK_BACKBONES
            )
            if len(single_backbones) > 1 or len(multitask_backbones) > 1:
                errors.append("COMPOSITION_BACKBONE_ALTERNATIVES_CONFLICT")
            if single_backbones and multitask_backbones:
                errors.append("COMPOSITION_SINGLE_AND_MULTITASK_CONFLICT")
            if not single_backbones and not multitask_backbones:
                errors.append("COMPOSITION_BACKBONE_REQUIRED")
            if len(method_ids.intersection(COMPOSITION_INTEREST_METHODS)) > 1:
                errors.append("COMPOSITION_INTEREST_ALTERNATIVES_CONFLICT")
            if method_ids.intersection(COMPOSITION_HIDDEN_UNIT_ADAPTERS) and (
                not single_backbones
                or "model_field_aware_fm" in single_backbones
            ):
                errors.append("COMPOSITION_LHUC_REQUIRES_NEURAL_BACKBONE")
            if (
                method_ids.intersection(COMPOSITION_LOSS_ALIGNED_REFINEMENTS)
                and method_ids.intersection(COMPOSITION_DISTILLATION_OBJECTIVES)
            ):
                errors.append("COMPOSITION_LOSS_REFINEMENT_CONFLICT")
            if method_ids - known_composition_ids:
                errors.append("COMPOSITION_METHOD_UNCLASSIFIED")
        for method_id in sorted(method_ids):
            card = cards.get(method_id)
            if card is None:
                errors.append("UNKNOWN_METHOD_CARD")
                continue
            card_family = str(get_value(card, "family", ""))
            if is_composition and card_family not in legal_families:
                errors.append("COMPOSITION_FAMILY_NOT_ALLOWED")
            eligibility = evaluate_method_card(
                card,
                context,
                family=card_family if is_composition else family,
            )
            errors.extend(eligibility.reasons)
        source_events = set(map(str, as_list(get_value(context, "source_event_ids", None))))
        if any(not EVENT_ID_PATTERN.fullmatch(event_id) for event_id in source_events):
            errors.append("INVALID_CONTEXT_EVENT_ID")
        evidence_events = set(map(str, as_list(get_value(spec, "evidence_event_ids", None))))
        if any(not EVENT_ID_PATTERN.fullmatch(event_id) for event_id in evidence_events):
            errors.append("INVALID_EVIDENCE_EVENT_ID")
        if not evidence_events.issubset(source_events):
            errors.append("EVIDENCE_OUTSIDE_CONTEXT")

        available_literature: dict[str, dict[str, Any]] = {}
        for item in literature_evidence:
            snapshot = _literature_snapshot(item)
            if snapshot is None:
                errors.append("LITERATURE_SKILL_EVIDENCE_INVALID")
                continue
            evidence_id = str(snapshot["evidence_id"])
            if evidence_id in available_literature:
                errors.append("DUPLICATE_LITERATURE_SKILL_EVIDENCE")
            available_literature[evidence_id] = snapshot

        proposed_literature = as_list(
            get_value(spec, "literature_evidence", None)
        )
        if available_literature and not proposed_literature:
            errors.append("LITERATURE_EVIDENCE_REQUIRED")
        proposed_literature_ids = [
            str(get_value(item, "evidence_id", ""))
            for item in proposed_literature
        ]
        if len(proposed_literature_ids) != len(set(proposed_literature_ids)):
            errors.append("DUPLICATE_LITERATURE_EVIDENCE")
        for item in proposed_literature:
            snapshot = _literature_snapshot(item)
            if snapshot is None:
                errors.append("INVALID_LITERATURE_EVIDENCE")
                continue
            source = available_literature.get(str(snapshot["evidence_id"]))
            if source is None:
                errors.append("LITERATURE_EVIDENCE_OUTSIDE_SKILL")
            elif snapshot != source:
                errors.append("LITERATURE_EVIDENCE_TAMPERED")

        text = " ".join(
            str(get_value(spec, field, ""))
            for field in ("hypothesis", "change_summary", "expected_mechanism", "falsification_condition")
        ).lower()
        if any(pattern in text for pattern in HIDDEN_PATTERNS):
            errors.append("HIDDEN_TEST_REFERENCE")
        if contains_implementation_reference(text):
            errors.append("CODE_SPECIFIC_PLAN_FORBIDDEN")
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
        if seen_detector.contains(spec) and not _authorized_no_op_reimplementation(
            spec, context, choice
        ):
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
