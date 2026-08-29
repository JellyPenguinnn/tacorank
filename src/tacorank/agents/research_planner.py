"""Person 1's strict, bounded research-planning adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from tacorank.providers.research_provider import ProviderRequest, ResearchProvider
from tacorank.research.convergence_advisor import ConvergenceAdvisor
from tacorank.research.plan_validation import PlanValidator, ValidationResult
from tacorank.research.search_policy import PolicyChoice, SearchPolicy


class PlannerSchemaUnavailable(RuntimeError):
    """Raised when the shared Person 2 schema has not been installed yet."""


@dataclass(frozen=True)
class PlannerFailure:
    reason_code: str
    reason: str
    validation: ValidationResult | None = None


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _default_output_factory(action: str, spec: Any, reason_code: str, reason: str, supporting_event_ids: list[str]) -> Any:
    try:
        from tacorank import schemas

        model = getattr(schemas, "PlannerOutput")
    except (ImportError, AttributeError) as exc:
        raise PlannerSchemaUnavailable(
            "Person 2 must provide tacorank.schemas.PlannerOutput before the planner can emit output."
        ) from exc
    payload = {
        "action": action,
        "spec": spec,
        "reason_code": reason_code,
        "reason": reason,
        "supporting_event_ids": supporting_event_ids,
    }
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model(**payload)


class ResearchPlanner:
    """Combine the deterministic policy with one bounded provider call."""

    def __init__(
        self,
        provider: ResearchProvider,
        policy: SearchPolicy | None = None,
        validator: PlanValidator | None = None,
        convergence_advisor: ConvergenceAdvisor | None = None,
        output_factory: Callable[[str, Any, str, str, list[str]], Any] | None = None,
        input_token_limit: int | None = None,
        output_token_limit: int | None = None,
    ):
        self.provider = provider
        self.policy = policy or SearchPolicy()
        self.validator = validator or PlanValidator()
        self.convergence_advisor = convergence_advisor or ConvergenceAdvisor()
        self.output_factory = output_factory or _default_output_factory
        self.input_token_limit = input_token_limit
        self.output_token_limit = output_token_limit

    async def propose(self, context: Any) -> Any:
        advice = self.convergence_advisor.advise(context)
        supporting = [str(item) for item in _get(context, "source_event_ids", []) or []]
        if advice.action == "recommend_stop":
            return self.output_factory(
                "recommend_stop",
                None,
                advice.reason_code,
                advice.reason,
                list(advice.supporting_event_ids) or supporting,
            )

        choice = self.policy.choose(context)
        if choice.action != "propose":
            return self.output_factory(
                "blocked",
                None,
                choice.reason_code,
                choice.reason,
                supporting,
            )

        request = ProviderRequest(
            context=context,
            policy_choice=choice,
            input_token_limit=self.input_token_limit,
            output_token_limit=self.output_token_limit,
        )
        raw_spec = await self.provider.generate(request)
        result = self.validator.validate(raw_spec, context, choice=choice)
        if not result.accepted:
            repair = getattr(self.provider, "repair", None)
            if repair is not None:
                raw_spec = await repair(request, result.errors)
                result = self.validator.validate(raw_spec, context, choice=choice)
        if not result.accepted:
            return self.output_factory(
                "blocked",
                None,
                "INVALID_PROVIDER_PLAN",
                "Provider proposal failed Person 1 plan validation: " + ", ".join(result.errors),
                supporting,
            )

        return self.output_factory(
            "propose",
            raw_spec,
            choice.reason_code,
            choice.reason,
            supporting,
        )
