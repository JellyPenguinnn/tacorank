from pathlib import Path

from tacorank.research.portfolio import load_method_cards


def test_schema_v1_method_cards_load_with_markdown_sections():
    portfolio = load_method_cards(Path(__file__).parents[2] / "research" / "methods")

    assert len(portfolio.cards) == 20
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
