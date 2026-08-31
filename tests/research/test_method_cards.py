from pathlib import Path

from tacorank.research.portfolio import load_method_cards


def test_schema_v1_method_cards_load_with_markdown_sections():
    portfolio = load_method_cards(Path(__file__).parents[2] / "research" / "methods")

    assert len(portfolio.cards) == 17
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
    assert card.implementation_targets == ("solution/research_scaffold.py",)
    assert card.configuration_target == "solution/experiment_config.py"
    assert card.capability_status == "verified"
    assert card.implementation_id == "objective_bpr_v2"
    assert "negative_count" in card.active_parameters

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

    affinity = next(
        card
        for card in portfolio.cards
        if card.method_id == "features_history_affinity"
    )
    assert affinity.capability_status == "verified"
    assert affinity.implementation_id == "features_history_affinity_v1"
    assert "history_shrinkage" in affinity.active_parameters

    loss_aligned = next(
        card
        for card in portfolio.cards
        if card.method_id == "objective_loss_aligned_features"
    )
    assert loss_aligned.family == "objective"
    assert loss_aligned.prerequisites == ("pairwise_tested",)
    assert "simultaneous_loss_change" in loss_aligned.prohibition_conditions

    static = next(
        card
        for card in portfolio.cards
        if card.method_id == "static_feature_expansion_known_negative"
    )
    assert static.status == "known_negative"
