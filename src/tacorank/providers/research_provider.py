"""Provider protocol for the bounded Person 1 research planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class ProviderRequest:
    context: Any
    policy_choice: Any
    input_token_limit: int | None = None
    output_token_limit: int | None = None


class ResearchProvider(Protocol):
    async def generate(self, request: ProviderRequest) -> Any:
        """Return one raw or validated code-blind ResearchProposal, never a list."""

    async def repair(self, request: ProviderRequest, errors: tuple[str, ...]) -> Any:
        """Return one corrected ResearchProposal after a bounded format repair."""


class ProviderError(RuntimeError):
    pass


class MockResearchProvider:
    """Deterministic provider for unit and integration tests."""

    def __init__(self, response: Any | Callable[[ProviderRequest], Any]):
        self.response = response
        self.requests: list[ProviderRequest] = []
        self.repair_requests: list[tuple[ProviderRequest, tuple[str, ...]]] = []

    async def generate(self, request: ProviderRequest) -> Any:
        self.requests.append(request)
        return self.response(request) if callable(self.response) else self.response

    async def repair(self, request: ProviderRequest, errors: tuple[str, ...]) -> Any:
        self.repair_requests.append((request, errors))
        return self.response(request) if callable(self.response) else self.response
