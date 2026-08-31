"""Controller-owned normalization for typed campaign configurations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .graph_view import as_list, get_value


VARIANT_PARAMETER_DEFAULTS: Dict[str, Any] = {
    "formulation": "official_fm",
    "embedding_dim": 16,
    "learning_rate": 0.001,
    "epochs": 40,
    "negative_count": 2,
    "l2": 0.000001,
    "residual_scale": 0.05,
    "max_train_rows": 1141112,
    "history_decay_days": 14.0,
    "history_shrinkage": 20.0,
    "listwise_strategy": "full_observed",
}

METHOD_FORMULATIONS = {
    "objective_pairwise_bpr": "bpr",
    "objective_listwise_user_softmax": "listwise",
    "temporal_history_compact": "temporal_history",
    "features_history_affinity": "history_affinity",
}

METHOD_ACTIVE_PARAMETERS = {
    "objective_pairwise_bpr": (
        "formulation",
        "embedding_dim",
        "learning_rate",
        "epochs",
        "negative_count",
        "l2",
        "residual_scale",
        "max_train_rows",
    ),
    "objective_listwise_user_softmax": (
        "formulation",
        "embedding_dim",
        "learning_rate",
        "epochs",
        "l2",
        "residual_scale",
        "max_train_rows",
        "listwise_strategy",
    ),
    "temporal_history_compact": (
        "formulation",
        "residual_scale",
        "max_train_rows",
        "history_decay_days",
        "history_shrinkage",
    ),
}

METHOD_IMPLEMENTATION_IDS = {
    "objective_pairwise_bpr": "objective_bpr_v2",
    "objective_listwise_user_softmax": "objective_listwise_full_v2",
    "temporal_history_compact": "temporal_history_compact_v1",
}

FORMULATION_PARAMETER_OVERRIDES = {
    "history_affinity": {"epochs": 5},
}


def variant_parameter_defaults(
    active_parameters: Iterable[str], *, formulation: Optional[str] = None
) -> Dict[str, Any]:
    defaults = dict(VARIANT_PARAMETER_DEFAULTS)
    defaults.update(FORMULATION_PARAMETER_OVERRIDES.get(formulation or "", {}))
    return {
        name: defaults[name]
        for name in dict.fromkeys(map(str, active_parameters))
        if name in defaults
    }

CONTROL_PARAMETER_PRIORITY = (
    "max_train_rows",
    "residual_scale",
    "l2",
    "epochs",
    "learning_rate",
    "embedding_dim",
    "negative_count",
    "history_shrinkage",
    "history_decay_days",
)


def reference_variant_parameters(
    context: Any,
    implementation_parent_id: Optional[str],
    active_parameters: Iterable[str],
    *,
    formulation: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the inherited/default values used to define a treatment delta."""

    names = tuple(dict.fromkeys(map(str, active_parameters)))
    reference = variant_parameter_defaults(names, formulation=formulation)
    if not implementation_parent_id:
        return reference
    for summary in reversed(as_list(get_value(context, "family_history", None))):
        if str(get_value(summary, "experiment_id", "")) != implementation_parent_id:
            continue
        inherited = get_value(summary, "variant_parameters", None)
        if isinstance(inherited, Mapping):
            reference.update(
                {
                    name: inherited[name]
                    for name in names
                    if name in inherited
                }
            )
        break
    return reference


def resolve_variant_parameters(
    raw_parameters: Any,
    *,
    active_parameters: Iterable[str],
    formulation: str,
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    """Expand model-owned overrides into one complete effective configuration."""

    names = tuple(dict.fromkeys(map(str, active_parameters)))
    raw = raw_parameters if isinstance(raw_parameters, Mapping) else {}
    resolved = {
        name: reference.get(name, VARIANT_PARAMETER_DEFAULTS.get(name))
        for name in names
    }
    resolved.update({name: raw[name] for name in names if name in raw})
    if "formulation" in resolved:
        resolved["formulation"] = formulation
    return resolved


def treatment_partition(
    parameters: Mapping[str, Any],
    reference: Mapping[str, Any],
    active_parameters: Iterable[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Partition active parameters by their actual value delta from the parent."""

    names = tuple(dict.fromkeys(map(str, active_parameters)))
    changed = tuple(
        name for name in names if parameters.get(name) != reference.get(name)
    )
    held = tuple(name for name in names if name not in set(changed))
    return changed, held


def enforce_controlled_treatment(
    parameters: Mapping[str, Any],
    reference: Mapping[str, Any],
    active_parameters: Iterable[str],
) -> Dict[str, Any]:
    """Keep one matched control when a proposal changes every active parameter."""

    resolved = dict(parameters)
    changed, held = treatment_partition(resolved, reference, active_parameters)
    if held or not changed:
        return resolved
    changed_set = set(changed)
    anchor = next(
        (
            name
            for name in CONTROL_PARAMETER_PRIORITY
            if name in changed_set and name in reference
        ),
        None,
    )
    if anchor is None:
        anchor = next((name for name in changed if name in reference), None)
    if anchor is not None:
        resolved[anchor] = reference[anchor]
    return resolved
