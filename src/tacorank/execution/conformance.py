"""Plan-to-execution conformance checks for research candidates."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from ..schemas import ExperimentSpec, TrialType


class ExecutionConformanceError(ValueError):
    """The executable candidate did not implement its approved specification."""


def verify_execution_receipt(
    spec: ExperimentSpec, document: Mapping[str, Any]
) -> Dict[str, float]:
    if spec.implementation_id is None and not spec.active_parameter_names:
        # Legacy and first implementation trials do not yet have a frozen
        # capability identity. Gate A remains authoritative for that phase.
        return {}
    implementation_id = document.get("implementation_id")
    implementation_sha256 = document.get("implementation_sha256")
    effective = document.get("effective_parameters")
    semantics = document.get("training_semantics", {})
    if implementation_id != spec.implementation_id:
        raise ExecutionConformanceError("implementation_id does not match the plan")
    if spec.trial_type == TrialType.CONFIGURATION and (
        implementation_sha256 != spec.implementation_sha256
    ):
        raise ExecutionConformanceError(
            "configuration trial changed the hash-bound implementation"
        )
    if not isinstance(effective, dict):
        raise ExecutionConformanceError("effective_parameters receipt is missing")
    if set(effective) != set(spec.active_parameter_names):
        raise ExecutionConformanceError(
            "effective parameter names do not match the capability contract"
        )
    for name in spec.active_parameter_names:
        if name not in spec.variant_parameters:
            raise ExecutionConformanceError(
                "approved plan omitted active parameter %s" % name
            )
        if effective[name] != spec.variant_parameters[name]:
            raise ExecutionConformanceError(
                "effective parameter %s differs from the approved plan" % name
            )
    if not isinstance(semantics, dict):
        raise ExecutionConformanceError("training_semantics receipt is invalid")

    formulation = str(effective.get("formulation", ""))
    evidence: Dict[str, float] = {"implementation_conformant": 1.0}
    if formulation == "bpr":
        negative_count = _positive_int(semantics, "negative_count")
        positives = _positive_int(semantics, "positive_count")
        pairs = _positive_int(semantics, "pair_count")
        per_positive = _finite_number(semantics, "negatives_per_positive")
        expected = int(effective["negative_count"])
        if negative_count != expected or pairs != positives * expected:
            raise ExecutionConformanceError(
                "BPR receipt does not prove the declared negative count"
            )
        if abs(per_positive - expected) > 1e-12:
            raise ExecutionConformanceError(
                "BPR negatives_per_positive does not match the plan"
            )
        evidence.update(
            effective_negative_count=float(negative_count),
            training_pair_count=float(pairs),
        )
    elif formulation == "listwise":
        if semantics.get("listwise_strategy") != "full_observed":
            raise ExecutionConformanceError(
                "listwise execution did not use complete observed lists"
            )
        list_count = _positive_int(semantics, "list_count")
        informative_users = _positive_int(semantics, "informative_user_count")
        list_rows = _positive_int(semantics, "list_row_count")
        target_mass = _finite_number(
            semantics, "normalized_positive_target_mass"
        )
        if list_count != informative_users or abs(target_mass - 1.0) > 1e-12:
            raise ExecutionConformanceError(
                "listwise receipt does not prove normalized complete user lists"
            )
        evidence.update(
            training_list_count=float(list_count),
            training_list_row_count=float(list_rows),
        )
    return evidence


def _positive_int(values: Mapping[str, Any], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExecutionConformanceError("%s must be a positive integer" % name)
    return value


def _finite_number(values: Mapping[str, Any], name: str) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionConformanceError("%s must be numeric" % name)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExecutionConformanceError("%s must be finite" % name)
    return parsed
