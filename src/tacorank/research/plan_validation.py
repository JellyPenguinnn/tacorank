"""Pure validation for Person 1 planner outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .code_blind import find_implementation_reference
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

_VARIANT_BOUNDS = {
    "embedding_dim": (2.0, 32.0, True),
    "learning_rate": (1e-5, 0.2, False),
    "epochs": (1.0, 8.0, True),
    "negative_count": (1.0, 16.0, True),
    "l2": (0.0, 0.1, False),
    "residual_scale": (0.0, 0.5, False),
    "max_train_rows": (1000.0, 250_000.0, True),
    "history_decay_days": (1.0, 180.0, False),
    "history_shrinkage": (0.0, 1000.0, False),
}
_VARIANT_LITERALS = {
    "listwise_strategy": {"full_observed"},
}


def _variant_parameter_errors(family: str, values: Any) -> list[str]:
    if not isinstance(values, dict):
        return ["CAMPAIGN_VARIANT_PARAMETERS_REQUIRED"]
    formulation = values.get("formulation")
    allowed_formulations = {
        "objective": {"pointwise", "bpr", "listwise"},
        "temporal_history": {"temporal_history"},
    }.get(family, set())
    errors = []
    if formulation not in allowed_formulations:
        errors.append("VARIANT_FORMULATION_MISMATCH")
    unknown = set(values) - {"formulation", *_VARIANT_BOUNDS, *_VARIANT_LITERALS}
    if unknown:
        errors.append("UNKNOWN_VARIANT_PARAMETER")
    for key, value in values.items():
        if key in _VARIANT_LITERALS:
            if value not in _VARIANT_LITERALS[key]:
                errors.append("VARIANT_PARAMETER_OUT_OF_RANGE")
            continue
        if key not in _VARIANT_BOUNDS:
            continue
        lower, upper, integer = _VARIANT_BOUNDS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append("INVALID_VARIANT_PARAMETER_TYPE")
            continue
        if integer and not isinstance(value, int):
            errors.append("INVALID_VARIANT_PARAMETER_TYPE")
        elif not lower <= float(value) <= upper:
            errors.append("VARIANT_PARAMETER_OUT_OF_RANGE")
    return errors


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


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


class PlanValidator:
    """Validate a code-blind research proposal against verified context only."""

    def validate(
        self,
        spec: Any,
        context: Any,
        choice: Any | None = None,
        duplicate_detector: DuplicateDetector | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        diagnostics: list[str] = []
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
        effective_implementation_parent_id = get_value(
            spec, "implementation_parent_experiment_id", None
        ) or get_value(spec, "parent_experiment_id", None)
        if not SHARED_ID_PATTERN.fullmatch(str(effective_implementation_parent_id or "")):
            errors.append("INVALID_IMPLEMENTATION_PARENT_EXPERIMENT_ID")
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
            expected_implementation_parent = (
                get_value(choice, "implementation_parent", None)
                or get_value(choice, "parent", None)
            )
            if expected_implementation_parent is not None and (
                get_value(spec, "implementation_parent_experiment_id", None)
                or get_value(spec, "parent_experiment_id", None)
            ) != get_value(expected_implementation_parent, "experiment_id", None):
                errors.append("IMPLEMENTATION_PARENT_POLICY_MISMATCH")
            if get_value(choice, "family", None) and family != get_value(choice, "family"):
                errors.append("FAMILY_POLICY_MISMATCH")
            for field, code in (
                ("campaign_id", "CAMPAIGN_POLICY_MISMATCH"),
                ("variant_id", "VARIANT_POLICY_MISMATCH"),
            ):
                if get_value(spec, field, None) != get_value(choice, field, None):
                    errors.append(code)

        campaign = get_value(context, "research_campaign", None)
        campaign_id = get_value(spec, "campaign_id", None)
        variant_id = get_value(spec, "variant_id", None)
        if campaign is None:
            if campaign_id is not None or variant_id is not None:
                errors.append("UNCONFIGURED_CAMPAIGN_VARIANT")
        else:
            if campaign_id != get_value(campaign, "campaign_id", None):
                errors.append("CAMPAIGN_CONTEXT_MISMATCH")
            if not _nonempty(get_value(spec, "variant_instruction", None)):
                errors.append("CAMPAIGN_VARIANT_INSTRUCTION_REQUIRED")
            variant_parameters = get_value(spec, "variant_parameters", None)
            if (
                not isinstance(variant_parameters, dict)
                or not _nonempty(variant_parameters.get("formulation"))
            ):
                errors.append("CAMPAIGN_VARIANT_PARAMETERS_REQUIRED")
            else:
                errors.extend(_variant_parameter_errors(family, variant_parameters))
            campaign_methods = get_value(
                campaign, "family_method_card_ids", None
            ) or {}
            allowed_campaign_methods = {
                str(item) for item in as_list(campaign_methods.get(family, ()))
            }
            proposed_campaign_methods = {
                str(item)
                for item in as_list(get_value(spec, "method_card_ids", None))
            }
            if not proposed_campaign_methods.issubset(allowed_campaign_methods):
                errors.append("VARIANT_METHOD_MISMATCH")

        graph = GraphView.from_context(context)
        parent_id = get_value(spec, "parent_experiment_id", None)
        parent = graph.get(str(parent_id)) if parent_id else None
        historical_summaries = as_list(get_value(context, "family_history", None))
        historical_by_id = {
            str(get_value(summary, "experiment_id", "")): summary
            for summary in historical_summaries
            if str(get_value(summary, "experiment_id", ""))
        }
        implementation_parent_id = get_value(
            spec, "implementation_parent_experiment_id", None
        ) or parent_id
        implementation_parent = graph.get(str(implementation_parent_id))
        if implementation_parent is None and implementation_parent_id:
            implementation_parent = ExperimentNodeView.from_summary(
                historical_by_id.get(str(implementation_parent_id))
            )
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
        if implementation_parent is None:
            errors.append("UNKNOWN_IMPLEMENTATION_PARENT")
        elif (
            get_value(spec, "parent_commit_sha", None)
            and implementation_parent.parent_commit_sha
            and get_value(spec, "parent_commit_sha")
            != implementation_parent.parent_commit_sha
        ):
            errors.append("IMPLEMENTATION_PARENT_COMMIT_MISMATCH")

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
        required_method_id = get_value(choice, "method_card_id", None)
        if required_method_id and method_ids != {str(required_method_id)}:
            errors.append("METHOD_POLICY_MISMATCH")
        allowed_method_ids = {
            str(item)
            for item in as_list(
                get_value(choice, "allowed_method_card_ids", None)
                if choice is not None
                else None
            )
        }
        if allowed_method_ids and (
            len(raw_method_ids) != 1 or not method_ids.issubset(allowed_method_ids)
        ):
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
            active_parameters = {
                str(item)
                for item in as_list(get_value(card, "active_parameters", None))
            }
            if campaign is not None and active_parameters:
                supplied_parameters = set(
                    (get_value(spec, "variant_parameters", None) or {}).keys()
                )
                if supplied_parameters != active_parameters:
                    errors.append("VARIANT_ACTIVE_PARAMETER_MISMATCH")
                expected_formulation = {
                    "objective_pairwise_bpr": "bpr",
                    "objective_listwise_user_softmax": "listwise",
                    "temporal_history_compact": "temporal_history",
                }.get(method_id)
                if expected_formulation is not None and (
                    (get_value(spec, "variant_parameters", None) or {}).get(
                        "formulation"
                    )
                    != expected_formulation
                ):
                    errors.append("VARIANT_METHOD_FORMULATION_MISMATCH")
        source_events = set(map(str, as_list(get_value(context, "source_event_ids", None))))
        if any(not EVENT_ID_PATTERN.fullmatch(event_id) for event_id in source_events):
            errors.append("INVALID_CONTEXT_EVENT_ID")
        evidence_events = set(map(str, as_list(get_value(spec, "evidence_event_ids", None))))
        if any(not EVENT_ID_PATTERN.fullmatch(event_id) for event_id in evidence_events):
            errors.append("INVALID_EVIDENCE_EVENT_ID")
        if not evidence_events.issubset(source_events):
            errors.append("EVIDENCE_OUTSIDE_CONTEXT")
        history = as_list(get_value(context, "family_history", None))
        hypothesis_evidence = get_value(spec, "hypothesis_evidence", None)
        if history and hypothesis_evidence is None:
            errors.append("HYPOTHESIS_EVIDENCE_REQUIRED_AFTER_FIRST_EXPERIMENT")
        if hypothesis_evidence is not None:
            cited_evaluations = set(
                map(
                    str,
                    as_list(
                        get_value(
                            hypothesis_evidence,
                            "source_evaluation_event_ids",
                            None,
                        )
                    ),
                )
            )
            prior_evaluations = {
                str(get_value(summary, "evaluation_event_id", ""))
                for summary in history
                if get_value(summary, "evaluation_event_id", None)
            }
            if not cited_evaluations:
                errors.append("HYPOTHESIS_EVALUATION_EVIDENCE_REQUIRED")
            if not cited_evaluations.issubset(prior_evaluations):
                errors.append("HYPOTHESIS_EVIDENCE_NOT_PRIOR_EVALUATION")
            if not cited_evaluations.issubset(evidence_events):
                errors.append("HYPOTHESIS_EVIDENCE_NOT_DECLARED")
            changed = set(
                map(
                    str,
                    as_list(get_value(hypothesis_evidence, "changed_factors", None)),
                )
            )
            held = set(
                map(
                    str,
                    as_list(get_value(hypothesis_evidence, "held_constant", None)),
                )
            )
            if not changed or not held:
                errors.append("HYPOTHESIS_TREATMENT_BOUNDARY_REQUIRED")
            if changed.intersection(held):
                errors.append("HYPOTHESIS_TREATMENT_BOUNDARY_OVERLAP")
            if get_value(spec, "campaign_id", None):
                parameter_names = set(
                    map(
                        str,
                        (get_value(spec, "variant_parameters", None) or {}).keys(),
                    )
                )
                if changed != parameter_names:
                    errors.append("CAMPAIGN_CHANGED_FACTORS_MISMATCH")
            effects = get_value(
                hypothesis_evidence, "expected_metric_effects", None
            ) or {}
            baseline_metric_set = get_value(
                get_value(context, "baseline", None), "metric_set", None
            )
            allowed_metrics = set(
                map(
                    str,
                    (
                        get_value(baseline_metric_set, "metrics", None) or {}
                    ).keys(),
                )
            )
            if not effects:
                errors.append("EXPECTED_METRIC_EFFECT_REQUIRED")
            elif allowed_metrics and not set(map(str, effects)).issubset(allowed_metrics):
                errors.append("EXPECTED_METRIC_EFFECT_UNKNOWN")

        narrative_fields = (
            "hypothesis",
            "change_summary",
            "expected_mechanism",
            "falsification_condition",
            "variant_instruction",
        )
        narrative = {
            field: str(get_value(spec, field, "")) for field in narrative_fields
        }
        parameters = get_value(spec, "variant_parameters", "")
        narrative["variant_parameters"] = (
            json.dumps(parameters, ensure_ascii=True, sort_keys=True)
            if isinstance(parameters, dict)
            else str(parameters)
        )
        text = " ".join(narrative.values()).lower()
        if any(pattern in text for pattern in HIDDEN_PATTERNS):
            errors.append("HIDDEN_TEST_REFERENCE")
        for field, value in narrative.items():
            reference = find_implementation_reference(value)
            if reference is None:
                continue
            errors.append("CODE_SPECIFIC_PLAN_FORBIDDEN")
            diagnostics.append(
                "code_reference field=%s category=%s token=%s"
                % (field, reference.category, json.dumps(reference.text))
            )
            break
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

        return ValidationResult(
            accepted=not errors,
            errors=tuple(dict.fromkeys(errors)),
            warnings=tuple(dict.fromkeys(warnings)),
            diagnostics=tuple(dict.fromkeys(diagnostics)),
        )
