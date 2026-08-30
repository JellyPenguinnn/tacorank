"""Controller-owned normalization for typed campaign configurations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

from .graph_view import as_list, get_value


VARIANT_PARAMETER_DEFAULTS: Dict[str, Any] = {
    "formulation": "passthrough",
    "embedding_dim": 8,
    "learning_rate": 0.01,
    "epochs": 2,
    "negative_count": 2,
    "l2": 0.0001,
    "residual_scale": 0.05,
    "max_train_rows": 100000,
    "history_decay_days": 14.0,
    "history_shrinkage": 20.0,
    "listwise_strategy": "full_observed",
}

METHOD_FORMULATIONS = {
    "objective_pairwise_bpr": "bpr",
    "objective_listwise_user_softmax": "listwise",
    "temporal_history_compact": "temporal_history",
}


def reference_variant_parameters(
    context: Any,
    implementation_parent_id: str | None,
    active_parameters: Iterable[str],
) -> Dict[str, Any]:
    """Return the inherited/default values used to define a treatment delta."""

    names = tuple(dict.fromkeys(map(str, active_parameters)))
    reference = {
        name: VARIANT_PARAMETER_DEFAULTS[name]
        for name in names
        if name in VARIANT_PARAMETER_DEFAULTS
    }
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
