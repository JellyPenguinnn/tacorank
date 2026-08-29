"""Method-card registry facade."""

from __future__ import annotations

from pathlib import Path

from .portfolio import ExperimentPortfolio, MethodCard, default_portfolio, load_method_cards


class MethodCardRegistry:
    def __init__(self, portfolio: ExperimentPortfolio | None = None):
        self.portfolio = portfolio or default_portfolio()

    @classmethod
    def from_directory(cls, directory: str | Path) -> "MethodCardRegistry":
        loaded = load_method_cards(directory)
        return cls(loaded if loaded.cards else default_portfolio())

    def get(self, method_id: str) -> MethodCard | None:
        return next((card for card in self.portfolio.cards if card.method_id == method_id), None)

    def for_family(self, family: str) -> tuple[MethodCard, ...]:
        return self.portfolio.for_family(family)

    def all(self) -> tuple[MethodCard, ...]:
        return tuple(self.portfolio.cards)
