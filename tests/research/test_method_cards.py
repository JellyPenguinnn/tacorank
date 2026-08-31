from pathlib import Path

from tacorank.research.portfolio import load_method_cards


def test_schema_v1_method_cards_load_with_markdown_sections():
    portfolio = load_method_cards(Path(__file__).parents[2] / "research" / "methods")

    assert len(portfolio.cards) == 16
    card = next(card for card in portfolio.cards if card.method_id == "objective_pairwise_bpr")
    assert card.schema_version == "1.0"
    assert card.family == "objective"
    assert "within users" in card.mechanism
    assert card.falsifier.startswith("No stable")
    assert card.prerequisites == (
        "baseline_parity",
        "within_user_positive_negative_pairs",
        # A loss re-fit over the parent's own features cannot add information
        # as an opening move; it is gated until a full result exists or the
        # profile shows no within-list feature axis to try first.
        "objective_refit_justified",
    )
    assert set(card.allowed_data) == {
        "train_interactions",
        "user_id",
        "long_view",
        "verified_predictions",
    }
    assert card.prohibition_conditions == ("evaluator_or_split_change_required",)
    assert card.implementation_targets == ("solution/candidate.py",)

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
    assert residual.implementation_targets == ("solution/candidate.py",)
