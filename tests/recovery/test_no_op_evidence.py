"""No-op recovery evidence carries the measured distance from the parent."""

from __future__ import annotations

from types import SimpleNamespace

from tacorank.recovery.classifier import classify_failure


def _no_op(prediction_change):
    return SimpleNamespace(
        failure_stage="",
        error_class="",
        trust=SimpleNamespace(verdict="no_op", flags=("NO_PREDICTION_CHANGE",)),
        prediction_change=prediction_change,
    )


def test_evidence_reports_how_far_the_candidate_sat_from_the_parent():
    # The coding worker has no shell, so a bare flag name gives it nothing to
    # aim at. These are the values run_20260831T_v2 exp_002 actually produced.
    classification = classify_failure(
        _no_op(
            SimpleNamespace(
                within_user_rank_change=0.0000212,
                changed_row_fraction=0.0003182306376546402,
                spearman_vs_parent=0.9999989589395027,
            )
        )
    )

    assert classification.failure_class == "no_op"
    assert "NO_PREDICTION_CHANGE" in classification.evidence
    assert "within_user_rank_change=2.12e-05" in classification.evidence
    assert "spearman_vs_parent=0.999999" in classification.evidence
    assert "within-user ordering" in classification.evidence


def test_evidence_falls_back_cleanly_when_nothing_was_measured():
    classification = classify_failure(_no_op(None))

    assert classification.failure_class == "no_op"
    assert classification.evidence == "NO_PREDICTION_CHANGE"


def test_partial_measurements_are_reported_without_the_missing_fields():
    classification = classify_failure(
        _no_op(
            SimpleNamespace(
                within_user_rank_change=None,
                changed_row_fraction=0.5,
                spearman_vs_parent=None,
            )
        )
    )

    assert "changed_row_fraction=0.5" in classification.evidence
    assert "within_user_rank_change" not in classification.evidence
