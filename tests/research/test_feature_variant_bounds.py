from tacorank.research.plan_validation import _variant_parameter_errors
from tacorank.research.variant_configuration import VARIANT_PARAMETER_DEFAULTS


def _feature_parameters(**overrides):
    values = {
        key: VARIANT_PARAMETER_DEFAULTS[key]
        for key in (
            "formulation",
            "learning_rate",
            "epochs",
            "l2",
            "residual_scale",
            "max_train_rows",
            "history_shrinkage",
        )
    }
    values["formulation"] = "history_affinity"
    values.update(overrides)
    return values


def test_history_affinity_variant_accepts_regularized_defaults() -> None:
    assert _variant_parameter_errors("features", _feature_parameters()) == []


def test_history_affinity_variant_rejects_overfit_configuration() -> None:
    errors = _variant_parameter_errors(
        "features",
        _feature_parameters(
            history_shrinkage=0.0,
            l2=0.0,
            epochs=8,
            residual_scale=0.5,
        ),
    )

    assert errors.count("FEATURE_REGULARIZATION_REQUIRED") == 2
    assert errors.count("FEATURE_COMPLEXITY_LIMIT_EXCEEDED") == 2
