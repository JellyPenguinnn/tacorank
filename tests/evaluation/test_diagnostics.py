from tacorank.evaluation.diagnostics import (
    DiagnosticFeatures,
    compute_evaluation_diagnostics,
)
from tacorank.evaluation.types import MetricDelta


def test_diagnostics_expose_generalization_temporal_and_slice_failures() -> None:
    users = ("u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4")
    labels = (1, 0, 1, 0, 1, 0, 1, 0)
    parent = (0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1)
    candidate = (0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9)
    features = DiagnosticFeatures(
        dates=(20220422, 20220422, 20220423, 20220423, 20220427, 20220427, 20220428, 20220428),
        duration_ms=(5_000, 5_000, 10_000, 10_000, 20_000, 20_000, 30_000, 30_000),
        item_popularity=(1, 1, 10, 10, 100, 100, 1_000, 1_000),
        user_history_count=(2, 2, 5, 5, 20, 20, 100, 100),
        validation_arms=("val_a", "val_a", "val_a", "val_a", "val_b", "val_b", "val_b", "val_b"),
    )

    diagnostics = compute_evaluation_diagnostics(
        user_ids=users,
        labels=labels,
        candidate_scores=candidate,
        parent_scores=parent,
        parent_delta=MetricDelta(
            -0.01, {"GAUC": -0.02, "nDCG@5": -0.01}
        ),
        features=features,
        proxy_parent_delta=0.02,
    )

    assert diagnostics.proxy_full_delta_gap == 0.03
    assert diagnostics.validation_arm_deltas["val_a"] > 0
    assert diagnostics.validation_arm_deltas["val_b"] < 0
    assert diagnostics.validation_arm_gap is not None
    assert diagnostics.temporal_delta_slope is not None
    assert diagnostics.temporal_delta_slope < 0
    assert diagnostics.gain_concentration_top10pct is not None
    assert "user_history.cold" in diagnostics.slice_deltas
    assert "duration_rank.short_lt7s" in diagnostics.slice_deltas
    assert "popularity_rank.cold" in diagnostics.slice_deltas
    assert diagnostics.best_slice in diagnostics.slice_deltas
    assert diagnostics.worst_slice in diagnostics.slice_deltas
    assert any("opposite signs" in item for item in diagnostics.failure_hypotheses)
    assert any("Temporal degradation" in item for item in diagnostics.failure_hypotheses)
    assert any("contract v1" in item for item in diagnostics.limitations)


def test_supplied_train_validation_gap_is_preserved_without_unavailable_warning() -> None:
    diagnostics = compute_evaluation_diagnostics(
        user_ids=("u1", "u1"),
        labels=(1, 0),
        candidate_scores=(0.9, 0.1),
        parent_scores=(0.8, 0.2),
        parent_delta=MetricDelta(0.01, {"GAUC": 0.01, "nDCG@5": 0.01}),
        train_validation_gap=0.12,
    )

    assert diagnostics.train_validation_gap == 0.12
    assert not any("contract v1" in item for item in diagnostics.limitations)
