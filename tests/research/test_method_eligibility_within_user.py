"""Within-user profile capabilities and the objective-card opening gate."""

from __future__ import annotations

from types import SimpleNamespace

from tacorank.research.method_eligibility import (
    WITHIN_USER_FEATURE_AXIS_MIN,
    available_capabilities,
)


def _context(dispersion, history=()):
    return SimpleNamespace(
        contract_summary=SimpleNamespace(
            allowed_data=[
                "train_interactions",
                "user_id",
                "long_view",
                "date",
                "duration_ms",
            ],
            research_capabilities=[],
            active_prohibitions=[],
            epsilon=0.002,
        ),
        family_history=list(history),
        data_profile=SimpleNamespace(
            train_long_view_by_date=[
                SimpleNamespace(positive_rate=0.30),
                SimpleNamespace(positive_rate=0.36),
            ],
            score_within_user_duration_dispersion=dispersion,
        ),
    )


def test_drift_capability_reads_the_profile_without_raising():
    """Regression: the profile branch referenced an undefined helper.

    Every real run carries a non-empty train_long_view_by_date, so a missing
    numeric coercion made available_capabilities raise NameError before any
    proposal could be scored.
    """

    capabilities = available_capabilities(_context(0.5))
    assert "drift_diagnostics_material" in capabilities


def test_within_list_axis_defers_the_objective_refit():
    capabilities = available_capabilities(
        _context(WITHIN_USER_FEATURE_AXIS_MIN + 0.1)
    )
    assert "within_user_feature_axis_material" in capabilities
    assert "objective_refit_justified" not in capabilities


def test_absent_within_list_axis_justifies_the_objective_refit():
    capabilities = available_capabilities(
        _context(WITHIN_USER_FEATURE_AXIS_MIN - 0.1)
    )
    assert "objective_refit_justified" in capabilities
    assert "within_user_feature_axis_material" not in capabilities


def test_missing_profile_leaves_the_objective_card_available():
    """No profile is not evidence against a loss re-fit."""

    assert "objective_refit_justified" in available_capabilities(_context(None))


def test_a_full_public_result_justifies_the_objective_refit():
    measured = SimpleNamespace(
        trust_verdict="accepted",
        integrity="clean",
        highest_completed_fidelity="full",
        population="public_validation",
        stability="confirmed",
        parent_eligible=True,
        method_card_ids=["temporal_history_compact"],
        output_accepted=True,
        metric_deltas={},
        experiment_id="exp_001",
    )
    capabilities = available_capabilities(
        _context(WITHIN_USER_FEATURE_AXIS_MIN + 0.1, history=[measured])
    )
    assert "objective_refit_justified" in capabilities
