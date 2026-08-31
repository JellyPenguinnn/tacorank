from solution.experiment_config import CONFIG

from tacorank.research.variant_configuration import (
    VARIANT_PARAMETER_DEFAULTS,
    enforce_controlled_treatment,
    resolve_variant_parameters,
    treatment_partition,
)


def test_controller_defaults_match_executable_scaffold_configuration():
    assert VARIANT_PARAMETER_DEFAULTS == {
        key: value for key, value in CONFIG.items() if key != "family"
    }


def test_partial_override_expands_and_partitions_against_parent():
    active = (
        "formulation",
        "embedding_dim",
        "learning_rate",
        "epochs",
        "negative_count",
        "l2",
        "residual_scale",
        "max_train_rows",
    )
    reference = {
        name: VARIANT_PARAMETER_DEFAULTS[name] for name in active
    }
    reference["formulation"] = "bpr"

    resolved = resolve_variant_parameters(
        {"negative_count": 4, "inactive_parameter": 99},
        active_parameters=active,
        formulation="bpr",
        reference=reference,
    )
    changed, held = treatment_partition(resolved, reference, active)

    assert set(resolved) == set(active)
    assert resolved["negative_count"] == 4
    assert "inactive_parameter" not in resolved
    assert changed == ("negative_count",)
    assert set(held) == set(active) - {"negative_count"}


def test_all_changed_parameters_are_reduced_to_a_controlled_treatment():
    active = (
        "formulation",
        "embedding_dim",
        "learning_rate",
        "epochs",
        "negative_count",
        "l2",
        "residual_scale",
        "max_train_rows",
    )
    reference = {
        name: VARIANT_PARAMETER_DEFAULTS[name] for name in active
    }
    reference["formulation"] = "bpr"
    proposal = dict(reference)
    proposal.update(
        embedding_dim=32,
        learning_rate=0.02,
        epochs=8,
        negative_count=4,
        l2=0.001,
        residual_scale=0.2,
        max_train_rows=200000,
    )
    proposal["formulation"] = "listwise"

    controlled = enforce_controlled_treatment(proposal, reference, active)
    changed, held = treatment_partition(controlled, reference, active)

    assert changed
    assert held == ("max_train_rows",)
    assert controlled["max_train_rows"] == reference["max_train_rows"]
