from pathlib import Path
from types import SimpleNamespace

from tacorank.research.method_eligibility import available_capabilities
from tacorank.research.portfolio import default_portfolio, load_method_cards


def test_schema_v1_method_cards_load_with_markdown_sections():
    portfolio = load_method_cards(Path(__file__).parents[2] / "research" / "methods")

    assert len(portfolio.cards) == 26
    assert default_portfolio().ids() == portfolio.ids()
    assert {
        "objective_pairwise_hinge_margin",
        "objective_lambda_ndcg_surrogate",
        "temporal_recency_weighted_ranker",
        "temporal_hour_context",
        "multitask_watch_time_auxiliary",
        "multitask_negative_feedback_auxiliary",
        "features_frequency_crosses",
        "features_duration_context_interactions",
        "model_field_aware_ranker",
        "sampling_hard_negative_pairs",
    }.issubset(portfolio.ids())
    direct = next(
        card
        for card in portfolio.cards
        if card.method_id == "objective_direct_within_user_ranker"
    )
    assert "parent_replacement" in direct.tags
    assert "Do not add" in (
        Path(__file__).parents[2]
        / "research/methods/objective_direct_within_user_ranker.md"
    ).read_text(encoding="utf-8")
    card = next(card for card in portfolio.cards if card.method_id == "objective_pairwise_bpr")
    assert card.schema_version == "1.0"
    assert card.family == "objective"
    assert "within users" in card.mechanism
    assert card.falsifier.startswith("No stable")
    assert card.prerequisites == (
        "baseline_parity",
        "within_user_positive_negative_pairs",
    )
    assert set(card.allowed_data) == {
        "train_interactions",
        "user_id",
        "long_view",
        "verified_predictions",
    }
    assert card.prohibition_conditions == ("evaluator_or_split_change_required",)
    assert card.implementation_targets == (
        "solution/candidate.py",
        "solution/features.py",
        "solution/model.py",
        "solution/train.py",
        "solution/inference.py",
    )

    loss_aligned = next(
        card
        for card in portfolio.cards
        if card.method_id == "objective_loss_aligned_features"
    )
    assert loss_aligned.family == "objective"
    assert loss_aligned.prerequisites == ("pairwise_tested",)
    assert "user_id" in loss_aligned.allowed_data
    assert "simultaneous_loss_change" in loss_aligned.prohibition_conditions

    residual = next(
        card
        for card in portfolio.cards
        if card.method_id == "ensemble_diverse_residual_candidate"
    )
    assert residual.family == "ensemble"
    assert residual.cost_tier == "low"
    assert residual.prerequisites == (
        "verified_best_prediction",
        "diverse_clean_proxy_member",
    )
    assert residual.implementation_targets == (
        "solution/candidate.py",
        "solution/features.py",
        "solution/inference.py",
    )

    synthesis = next(
        card
        for card in portfolio.cards
        if card.method_id == "ensemble_parallel_round_synthesis"
    )
    assert synthesis.prerequisites == ("two_confirmed_clean_members",)
    assert "solution/candidate.py" in synthesis.implementation_targets

    history = next(
        card
        for card in portfolio.cards
        if card.method_id == "temporal_history_compact"
    )
    assert history.implementation_targets == (
        "solution/candidate.py",
        "solution/features.py",
        "solution/inference.py",
    )


def test_material_date_rate_shift_unlocks_temporal_drift(planner_context):
    planner_context.data_profile = SimpleNamespace(
        train_long_view_by_date=[
            SimpleNamespace(positive_rate=0.20),
            SimpleNamespace(positive_rate=0.24),
        ]
    )

    assert "drift_diagnostics_material" in available_capabilities(planner_context)


def test_verified_baseline_unlocks_public_evaluation(planner_context):
    assert "standard_public_evaluation_complete" in available_capabilities(
        planner_context
    )
