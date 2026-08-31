"""Person 1's strict, bounded research-planning adapter."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Mapping

from tacorank.providers.research_provider import ProviderRequest, ResearchProvider
from tacorank.research.convergence_advisor import ConvergenceAdvisor
from tacorank.research.literature import LiteratureResearchSkill
from tacorank.research.plan_validation import PlanValidator, ValidationResult
from tacorank.research.search_policy import PolicyChoice, SearchPolicy


logger = logging.getLogger(__name__)


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
        literature_skill: LiteratureResearchSkill | None = None,
    ):
        self.provider = provider
        self.policy = policy or SearchPolicy()
        self.validator = validator or PlanValidator()
        self.convergence_advisor = convergence_advisor or ConvergenceAdvisor()
        self.output_factory = output_factory or _default_output_factory
        self.input_token_limit = input_token_limit
        self.output_token_limit = output_token_limit
        self.literature_skill = literature_skill

    def preflight(self) -> None:
        """Verify both the model provider and optional online research skill."""

        provider_preflight = getattr(self.provider, "preflight", None)
        if provider_preflight is not None:
            provider_preflight()
        if self.literature_skill is not None:
            self.literature_skill.preflight()

    def _attach_provider_usage(self, output: Any) -> Any:
        resource_delta = getattr(self.provider, "resource_delta", None)
        literature_delta = (
            getattr(self.literature_skill, "resource_delta", None)
            if self.literature_skill is not None
            else None
        )
        if literature_delta is not None:
            if resource_delta is None:
                resource_delta = literature_delta
            elif hasattr(resource_delta, "model_copy"):
                resource_delta = resource_delta.model_copy(
                    update={
                        "wall_time_ms": int(
                            getattr(resource_delta, "wall_time_ms", 0)
                        )
                        + int(getattr(literature_delta, "wall_time_ms", 0))
                    }
                )
        if resource_delta is None:
            return output
        if hasattr(output, "model_copy"):
            return output.model_copy(update={"resource_delta": resource_delta})
        if isinstance(output, Mapping):
            return {**output, "resource_delta": resource_delta}
        return output

    async def propose(self, context: Any) -> Any:
        return await self._propose(context)

    async def propose_parallel_direction(
        self, context: Any, direction_index: int, direction_count: int
    ) -> Any:
        choice = self.policy.choose_parallel_direction(
            context, direction_index, direction_count
        )
        return await self._propose(context, forced_choice=choice)

    def parallel_direction_capacity(self, context: Any) -> int:
        """Expose the policy's unique lane count to the deterministic router."""

        return self.policy.parallel_direction_capacity(context)

    async def propose_synthesis(
        self, context: Any, component_experiment_ids: list[str]
    ) -> Any:
        choice = self.policy.choose_synthesis(
            context, component_experiment_ids
        )
        return await self._propose(context, forced_choice=choice)

    async def _propose(self, context: Any, forced_choice: PolicyChoice | None = None) -> Any:
        logger.info(
            "research_planner_started context_id=%s run_id=%s",
            _get(context, "context_id", None),
            _get(context, "run_id", None),
        )
        advice = self.convergence_advisor.advise(context)
        supporting = [str(item) for item in _get(context, "source_event_ids", []) or []]
        if advice.action == "recommend_stop":
            logger.info(
                "research_planner_stop_recommended context_id=%s reason_code=%s",
                _get(context, "context_id", None),
                advice.reason_code,
            )
            return self.output_factory(
                "recommend_stop",
                None,
                advice.reason_code,
                advice.reason,
                list(advice.supporting_event_ids) or supporting,
            )

        choice = forced_choice or self.policy.choose(context)
        if choice.action != "propose":
            logger.info(
                "research_planner_blocked context_id=%s reason_code=%s",
                _get(context, "context_id", None),
                choice.reason_code,
            )
            return self.output_factory(
                "blocked",
                None,
                choice.reason_code,
                choice.reason,
                supporting,
            )

        literature_evidence = ()
        literature_required = None
        if self.literature_skill is not None:
            literature_required = bool(
                getattr(self.literature_skill, "requires_citation", True)
            )
            logger.info(
                "literature_research_started context_id=%s method_card_id=%s",
                _get(context, "context_id", None),
                choice.method_card_id,
            )
            literature_evidence = tuple(
                await self.literature_skill.research(context, choice)
            )
            logger.info(
                "literature_research_completed context_id=%s evidence_ids=%s",
                _get(context, "context_id", None),
                ",".join(item.evidence_id for item in literature_evidence),
            )

        request = ProviderRequest(
            context=context,
            policy_choice=choice,
            input_token_limit=self.input_token_limit,
            output_token_limit=self.output_token_limit,
            literature_evidence=literature_evidence,
            literature_required=literature_required,
        )
        logger.info(
            "research_provider_selected context_id=%s parent_id=%s family=%s "
            "method_card_id=%s phase=%s reason_code=%s",
            _get(context, "context_id", None),
            _get(choice.parent, "experiment_id", None),
            choice.family,
            choice.method_card_id,
            choice.phase,
            choice.reason_code,
        )
        raw_spec = await self.provider.generate(request)
        result = self.validator.validate(
            raw_spec,
            context,
            choice=choice,
            literature_evidence=literature_evidence,
            literature_required=literature_required,
        )
        if not result.accepted:
            logger.warning(
                "research_plan_validation_failed context_id=%s errors=%s diagnostics=%s",
                _get(context, "context_id", None),
                ",".join(result.errors),
                ";".join(result.diagnostics) or "none",
            )
            repair = getattr(self.provider, "repair", None)
            if repair is not None:
                logger.info(
                    "research_plan_repair_started context_id=%s errors=%s",
                    _get(context, "context_id", None),
                    ",".join(result.errors),
                )
                raw_spec = await repair(request, result.errors)
                result = self.validator.validate(
                    raw_spec,
                    context,
                    choice=choice,
                    literature_evidence=literature_evidence,
                    literature_required=literature_required,
                )
        if not result.accepted:
            logger.warning(
                "research_plan_blocked_after_repair context_id=%s errors=%s diagnostics=%s",
                _get(context, "context_id", None),
                ",".join(result.errors),
                ";".join(result.diagnostics) or "none",
            )
            diagnostic_suffix = (
                "; diagnostics: " + "; ".join(result.diagnostics)
                if result.diagnostics
                else ""
            )
            return self._attach_provider_usage(
                self.output_factory(
                    "blocked",
                    None,
                    "INVALID_PROVIDER_PLAN",
                    "Provider proposal failed Person 1 plan validation: "
                    + ", ".join(result.errors)
                    + diagnostic_suffix,
                    supporting,
                )
            )

        logger.info(
            "research_plan_accepted context_id=%s experiment_id=%s family=%s",
            _get(context, "context_id", None),
            _get(raw_spec, "experiment_id", None),
            choice.family,
        )
        return self._attach_provider_usage(
            self.output_factory(
                "propose",
                raw_spec,
                choice.reason_code,
                choice.reason,
                supporting,
            )
        )
