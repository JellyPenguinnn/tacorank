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
    literature_evidence: tuple[Any, ...] = ()
    literature_status: str = "disabled"
    observations: tuple[Any, ...] = ()
    legal_choices: tuple[Any, ...] = ()
    research_turn_index: int = 0
    research_turn_attempt: int = 1
    research_turn_error: str | None = None
    batch_role: str | None = None


class ResearchProvider(Protocol):
    async def generate(self, request: ProviderRequest) -> Any:
        """Return one raw or validated code-blind ResearchProposal, never a list."""

    async def repair(self, request: ProviderRequest, errors: tuple[str, ...]) -> Any:
        """Return one corrected ResearchProposal after a bounded format repair."""

    async def research_turn(self, request: ProviderRequest) -> Any:
        """Return one typed bounded-research turn, if the mode is enabled."""

    def begin_research_session(self) -> None:
        """Reset usage accounting before one bounded research session."""


class ProviderError(RuntimeError):
    pass


class ProviderProtocolError(ProviderError):
    """The provider returned an unusable response envelope or JSON payload."""


class TransientProviderError(ProviderError):
    """Provider failure that is safe to retry without changing the request."""


class ProviderTimeoutError(TransientProviderError):
    """Provider request exceeded its configured network timeout."""


class MockResearchProvider:
    """Deterministic provider for unit and integration tests."""

    def __init__(self, response: Any | Callable[[ProviderRequest], Any]):
        self.response = response
        self.requests: list[ProviderRequest] = []
        self.repair_requests: list[tuple[ProviderRequest, tuple[str, ...]]] = []

    def begin_research_session(self) -> None:
        """Match the production provider session boundary; usage is not tracked."""

        return None

    async def generate(self, request: ProviderRequest) -> Any:
        self.requests.append(request)
        return self.response(request) if callable(self.response) else self.response

    async def repair(self, request: ProviderRequest, errors: tuple[str, ...]) -> Any:
        self.repair_requests.append((request, errors))
        return self.response(request) if callable(self.response) else self.response

    async def research_turn(self, request: ProviderRequest) -> Any:
        self.requests.append(request)
        response = self.response(request) if callable(self.response) else self.response
        if isinstance(response, (list, tuple)):
            index = min(request.research_turn_index, len(response) - 1)
            return response[index]
        return response
