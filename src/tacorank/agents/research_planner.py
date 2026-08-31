"""Person 1's strict, bounded research-planning adapter."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Mapping

from tacorank.providers.research_provider import (
    ProviderProtocolError,
    ProviderRequest,
    ResearchProvider,
)
from tacorank.research.agent_tools import ResearchToolRegistry
from tacorank.research.convergence_advisor import ConvergenceAdvisor
from tacorank.research.literature import LiteratureResearchError, LiteratureResearchSkill
from tacorank.research.plan_validation import PlanValidator, ValidationResult
from tacorank.research.search_policy import PolicyChoice, SearchPolicy
from tacorank.schemas import ResearchTurn, ResearchTurnAction


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


def _current_run_evidence_ids(context: Any) -> set[str]:
    """Return event IDs exposed by the typed, current-run planner context.

    Planner contexts carry provenance in nested advisory aggregates (frontier,
    round summaries, experiment summaries, and observations).  Those IDs are
    still current-run evidence, even when they are not repeated in the
    document-level ``source_event_ids`` field.  Keep this allowlist structural
    and typed: never scrape IDs from rendered context text.
    """

    evidence: set[str] = set()

    def add(value: Any, *field_names: str) -> None:
        for field_name in field_names:
            values = _get(value, field_name, ()) or ()
            if isinstance(values, (str, bytes)):
                values = (values,)
            for item in values:
                if isinstance(item, str) and item.startswith("evt_"):
                    evidence.add(item)

    add(context, "source_event_ids")
    for field_name in (
        "baseline",
        "current_best",
        "eligible_frontier",
        "family_history",
        "active_lessons",
        "research_frontier",
        "research_observations",
    ):
        values = _get(context, field_name, None)
        if values is None:
            continue
        if isinstance(values, (list, tuple, set, frozenset)):
            values = values
        else:
            values = (values,)
        for item in values:
            add(item, "source_event_ids", "supporting_event_ids")

    add(_get(context, "round_summary", None), "source_event_ids")
    return evidence


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
        research_agent_mode: str = "legacy",
        research_tool_step_limit: int = 4,
        research_literature_max_queries: int = 2,
        research_planning_max_attempts: int = 2,
        literature_required: bool = False,
    ):
        self.provider = provider
        self.policy = policy or SearchPolicy()
        self.validator = validator or PlanValidator()
        self.convergence_advisor = convergence_advisor or ConvergenceAdvisor()
        self.output_factory = output_factory or _default_output_factory
        self.input_token_limit = input_token_limit
        self.output_token_limit = output_token_limit
        self.literature_skill = literature_skill
        self.research_agent_mode = research_agent_mode
        self.research_tool_step_limit = min(4, max(1, research_tool_step_limit))
        self.research_literature_max_queries = min(2, max(0, research_literature_max_queries))
        self.research_planning_max_attempts = min(
            3, max(1, research_planning_max_attempts)
        )
        self.literature_required = literature_required
        self._observation_sink: Callable[[Any], None] | None = None

    def set_observation_sink(self, sink: Callable[[Any], None] | None) -> None:
        self._observation_sink = sink

    def preflight(self) -> None:
        """Verify both the model provider and optional online research skill."""

        provider_preflight = getattr(self.provider, "preflight", None)
        if provider_preflight is not None:
            provider_preflight()
        if self.literature_required and self.literature_skill is None:
            raise LiteratureResearchError(
                "required OpenAlex literature skill is not configured",
                status="unavailable",
                retryable=False,
            )
        if self.literature_skill is not None:
            try:
                self.literature_skill.preflight(required=self.literature_required)
            except TypeError:
                self.literature_skill.preflight()
            except Exception:
                if self.literature_required:
                    raise
                logger.warning("optional literature preflight unavailable", exc_info=True)

    def _attach_provider_usage(self, output: Any, *, include_literature: bool = True) -> Any:
        resource_delta = getattr(self.provider, "resource_delta", None)
        literature_delta = (
            getattr(self.literature_skill, "resource_delta", None)
            if self.literature_skill is not None
            else None
        )
        if not include_literature:
            literature_delta = None
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
        return await self._propose(context, forced_choice=choice, legal_choices=(choice,))

    def parallel_direction_capacity(self, context: Any) -> int:
        """Expose the policy's unique lane count to the deterministic router."""

        return self.policy.parallel_direction_capacity(context)

    async def propose_synthesis(
        self, context: Any, component_experiment_ids: list[str]
    ) -> Any:
        choice = self.policy.choose_synthesis(
            context, component_experiment_ids
        )
        return await self._propose(context, forced_choice=choice, legal_choices=(choice,))

    async def _propose(
        self,
        context: Any,
        forced_choice: PolicyChoice | None = None,
        legal_choices: tuple[PolicyChoice, ...] | None = None,
    ) -> Any:
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

        if self.research_agent_mode == "bounded_react":
            if getattr(self.provider, "research_turn", None) is None:
                return self._output(
                    "blocked",
                    None,
                    "RESEARCH_AGENT_UNAVAILABLE",
                    "The bounded research mode requires a typed research-turn provider.",
                    supporting,
                )
            return await self._propose_bounded_react(
                context,
                choice,
                legal_choices or self.policy.legal_choices(context),
                supporting,
            )

        literature_evidence = ()
        if self.literature_skill is not None:
            logger.info(
                "literature_research_started context_id=%s method_card_ids=%s",
                _get(context, "context_id", None),
                ",".join(choice.selected_method_card_ids),
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
        )
        logger.info(
            "research_provider_selected context_id=%s parent_id=%s family=%s "
            "method_card_ids=%s phase=%s reason_code=%s",
            _get(context, "context_id", None),
            _get(choice.parent, "experiment_id", None),
            choice.family,
            ",".join(choice.selected_method_card_ids),
            choice.phase,
            choice.reason_code,
        )
        raw_spec = await self.provider.generate(request)
        result = self.validator.validate(
            raw_spec,
            context,
            choice=choice,
            literature_evidence=literature_evidence,
        )
        if not result.accepted:
            logger.warning(
                "research_plan_validation_failed context_id=%s errors=%s",
                _get(context, "context_id", None),
                ",".join(result.errors),
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
                )
        if not result.accepted:
            logger.warning(
                "research_plan_blocked_after_repair context_id=%s errors=%s",
                _get(context, "context_id", None),
                ",".join(result.errors),
            )
            return self._attach_provider_usage(
                self.output_factory(
                    "blocked",
                    None,
                    "INVALID_PROVIDER_PLAN",
                    "Provider proposal failed Person 1 plan validation: "
                    + ", ".join(result.errors),
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

    def _output(
        self,
        action: str,
        spec: Any,
        reason_code: str,
        reason: str,
        supporting: list[str],
        *,
        selected_action_id: str | None = None,
        confidence: float | None = None,
    ) -> Any:
        output = self.output_factory(action, spec, reason_code, reason, supporting)
        updates = {}
        if selected_action_id is not None:
            updates["selected_action_id"] = selected_action_id
        if confidence is not None:
            updates["confidence"] = confidence
        if updates and hasattr(output, "model_copy"):
            output = output.model_copy(update=updates)
        elif updates and isinstance(output, Mapping):
            output = {**output, **updates}
        return output

    @staticmethod
    def _normalize_research_turn_envelope(
        raw_turn: Any,
        *,
        context: Any,
        selected_choice: PolicyChoice,
        legal_choices: tuple[PolicyChoice, ...],
        observations: list[Any],
    ) -> Any:
        """Fill only safe final-envelope fields from the same proposed plan.

        DeepSeek sometimes returns the candidate object correctly but places its
        duplicated research claims inside ``spec``.  The controller still owns
        legality and evidence checks; this adapter only copies fields that are
        already present in the current-run response or context.
        """

        if not isinstance(raw_turn, Mapping):
            return raw_turn
        value = dict(raw_turn)
        raw_spec = value.get("spec") or value.get("experiment_spec")
        final_field_names = (
            "selected_action_id",
            "claim",
            "hypothesis",
            "expected_mechanism",
            "success_criterion",
            "falsification_condition",
            "evidence_event_ids",
            "conservative_parameter_guidance",
        )
        # Some provider responses contain the complete final envelope but omit
        # only the discriminator.  It is safe to classify that shape as a
        # final-plan candidate because ResearchTurn still enforces every final
        # field and the controller validates the selected action/spec below.
        if (
            value.get("action") is None
            and isinstance(raw_spec, Mapping)
            and any(value.get(name) is not None for name in final_field_names)
        ):
            value["action"] = ResearchTurnAction.FINALIZE_PLAN.value
        if value.get("action") not in {
            ResearchTurnAction.FINALIZE_PLAN,
            ResearchTurnAction.FINALIZE_PLAN.value,
        }:
            return value
        if not isinstance(raw_spec, Mapping):
            return value
        if "spec" not in value:
            value["spec"] = dict(raw_spec)

        def copy_if_missing(field: str, *spec_fields: str) -> None:
            if value.get(field) is not None:
                return
            for spec_field in spec_fields:
                candidate = raw_spec.get(spec_field)
                if candidate is not None:
                    value[field] = candidate
                    return

        copy_if_missing("claim", "claim", "change_summary", "hypothesis")
        copy_if_missing("hypothesis", "hypothesis")
        copy_if_missing("expected_mechanism", "expected_mechanism")
        copy_if_missing("success_criterion", "success_criterion", "success_criteria")
        copy_if_missing("falsification_condition", "falsification_condition")

        if value.get("selected_action_id") is None and len(legal_choices) == 1:
            value["selected_action_id"] = selected_choice.choice_id

        if not value.get("evidence_event_ids"):
            evidence = raw_spec.get("evidence_event_ids")
            if not isinstance(evidence, (list, tuple)) or not evidence:
                evidence = sorted(_current_run_evidence_ids(context))
            if not evidence:
                evidence = [
                    event_id
                    for observation in observations
                    for event_id in (_get(observation, "source_event_ids", []) or [])
                ]
            value["evidence_event_ids"] = list(dict.fromkeys(str(item) for item in evidence))

        if not value.get("conservative_parameter_guidance"):
            guidance = raw_spec.get("conservative_parameter_guidance")
            if isinstance(guidance, Mapping) and guidance:
                value["conservative_parameter_guidance"] = dict(guidance)
        return value

    async def _propose_bounded_react(
        self,
        context: Any,
        initial_choice: PolicyChoice,
        legal_choices: tuple[PolicyChoice, ...],
        supporting: list[str],
    ) -> Any:
        """Run a bounded controller-mediated read-only research session."""

        choices_by_id = {choice.choice_id: choice for choice in legal_choices}
        if initial_choice.choice_id not in choices_by_id:
            legal_choices = (initial_choice,) + tuple(legal_choices)
            choices_by_id = {choice.choice_id: choice for choice in legal_choices}
        observations = []
        literature = []
        literature_queries = 0
        registry = ResearchToolRegistry(context, self.literature_skill)
        turn_provider = getattr(self.provider, "research_turn")
        selected_choice = initial_choice

        begin_session = getattr(self.provider, "begin_research_session", None)
        if callable(begin_session):
            begin_session()

        def blocked(
            reason_code: str,
            reason: str,
            evidence: list[str] | None = None,
        ) -> Any:
            return self._attach_provider_usage(
                self._output(
                    "blocked",
                    None,
                    reason_code,
                    reason,
                    list(dict.fromkeys(evidence or supporting)),
                ),
                include_literature=False,
            )

        for turn_index in range(self.research_tool_step_limit + 1):
            repair_hint: str | None = None
            for turn_attempt in range(1, self.research_planning_max_attempts + 1):
                request = ProviderRequest(
                    context=context,
                    policy_choice=selected_choice,
                    input_token_limit=self.input_token_limit,
                    output_token_limit=self.output_token_limit,
                    literature_evidence=tuple(literature),
                    literature_status=(
                        "available" if literature
                        else (
                            "unavailable"
                            if self.literature_skill is None
                            else "not_searched"
                        )
                    ),
                    observations=tuple(observations),
                    legal_choices=tuple(legal_choices),
                    research_turn_index=turn_index,
                    research_turn_attempt=turn_attempt,
                    research_turn_error=repair_hint,
                    batch_role=selected_choice.batch_role,
                )
                try:
                    raw_turn = await turn_provider(request)
                except ProviderProtocolError:
                    if turn_attempt < self.research_planning_max_attempts:
                        repair_hint = "provider_response_invalid"
                        continue
                    return blocked(
                        "RESEARCH_PROVIDER_PROTOCOL",
                        "The bounded research provider did not return a usable JSON turn after its repair attempt.",
                    )
                raw_turn = self._normalize_research_turn_envelope(
                    raw_turn,
                    context=context,
                    selected_choice=selected_choice,
                    legal_choices=legal_choices,
                    observations=observations,
                )
                try:
                    turn = ResearchTurn.model_validate(raw_turn)
                except Exception as error:
                    if turn_attempt < self.research_planning_max_attempts:
                        repair_hint = "research_turn_schema_invalid"
                        continue
                    return blocked(
                        "MALFORMED_RESEARCH_TURN",
                        "Bounded research turn was malformed: %s" % str(error)[:300],
                    )

                if turn.action == ResearchTurnAction.FINALIZE_PLAN:
                    candidate_choice = choices_by_id.get(
                        str(turn.selected_action_id)
                    )
                    if (
                        candidate_choice is None
                        or candidate_choice.action != "propose"
                    ):
                        if turn_attempt < self.research_planning_max_attempts:
                            repair_hint = "selected_action_id_not_legal"
                            continue
                        return blocked(
                            "INVALID_RESEARCH_ACTION",
                            "The final research decision did not select a controller-approved action.",
                        )
                    selected_choice = candidate_choice
                    if self.literature_required and not literature:
                        return blocked(
                            "LITERATURE_REQUIRED_UNAVAILABLE",
                            "Required literature evidence was not available for the final plan.",
                        )
                    known_evidence = _current_run_evidence_ids(context)
                    known_evidence.update(
                        event_id
                        for observation in observations
                        for event_id in observation.source_event_ids
                    )
                    if any(
                        event_id not in known_evidence
                        for event_id in turn.evidence_event_ids
                    ):
                        if turn_attempt < self.research_planning_max_attempts:
                            repair_hint = "evidence_event_id_outside_current_run"
                            continue
                        return blocked(
                            "INVALID_RESEARCH_EVIDENCE",
                            "The final research decision cited an event outside the current-run evidence boundary.",
                        )
                    raw_spec = dict(turn.spec or {})
                    if not raw_spec.get("evidence_event_ids"):
                        raw_spec["evidence_event_ids"] = list(
                            turn.evidence_event_ids
                        )
                    validation_context = context
                    if hasattr(context, "model_copy"):
                        validation_context = context.model_copy(
                            update={"source_event_ids": sorted(known_evidence)}
                        )
                    elif isinstance(context, Mapping):
                        validation_context = dict(context)
                        validation_context["source_event_ids"] = sorted(
                            known_evidence
                        )
                    else:
                        # Unit/custom providers may use a lightweight object
                        # instead of the production Pydantic context model.
                        # Preserve that compatibility without mutating the
                        # caller's context in place.
                        try:
                            import copy

                            validation_context = copy.copy(context)
                            setattr(
                                validation_context,
                                "source_event_ids",
                                sorted(known_evidence),
                            )
                        except (TypeError, AttributeError):
                            validation_context = context
                    final_request = request
                    if selected_choice.choice_id != request.policy_choice.choice_id:
                        final_request = ProviderRequest(
                            **{
                                **request.__dict__,
                                "policy_choice": selected_choice,
                            }
                        )
                        normalize = getattr(
                            self.provider, "normalize_candidate", None
                        )
                        if normalize is not None:
                            raw_spec = await normalize(raw_spec, final_request)
                    if selected_choice.hypothesis_group_id is not None:
                        raw_spec.setdefault(
                            "hypothesis_group_id",
                            selected_choice.hypothesis_group_id,
                        )
                    if selected_choice.batch_role is not None:
                        raw_spec.setdefault(
                            "batch_role", selected_choice.batch_role
                        )
                    result = self.validator.validate(
                        raw_spec,
                        validation_context,
                        choice=selected_choice,
                        literature_evidence=tuple(literature),
                    )
                    if not result.accepted:
                        repair = getattr(self.provider, "repair", None)
                        if repair is not None:
                            raw_spec = await repair(final_request, result.errors)
                            result = self.validator.validate(
                                raw_spec,
                                validation_context,
                                choice=selected_choice,
                                literature_evidence=tuple(literature),
                            )
                    if not result.accepted:
                        if turn_attempt < self.research_planning_max_attempts:
                            repair_hint = "final_plan_validation_failed"
                            continue
                        return blocked(
                            "INVALID_PROVIDER_PLAN",
                            "Bounded research plan failed validation: "
                            + ", ".join(result.errors),
                            supporting + turn.evidence_event_ids,
                        )
                    spec = raw_spec
                    if hasattr(spec, "model_copy"):
                        spec = spec.model_copy(
                            update={
                                "hypothesis_group_id": selected_choice.hypothesis_group_id,
                                "batch_role": selected_choice.batch_role,
                            }
                        )
                    return self._attach_provider_usage(
                        self._output(
                            "propose",
                            spec,
                            selected_choice.reason_code,
                            turn.claim or selected_choice.reason,
                            list(
                                dict.fromkeys(
                                    supporting + turn.evidence_event_ids
                                )
                            ),
                            selected_action_id=selected_choice.choice_id,
                            confidence=turn.confidence,
                        ),
                        include_literature=False,
                    )

                if turn_index >= self.research_tool_step_limit:
                    return blocked(
                        "RESEARCH_TOOL_STEP_LIMIT",
                        "The bounded research loop exhausted its read-only tool budget before finalizing.",
                        supporting + turn.evidence_event_ids,
                    )
                if turn.action == ResearchTurnAction.SEARCH_LITERATURE:
                    literature_queries += 1
                    if literature_queries > self.research_literature_max_queries:
                        return blocked(
                            "LITERATURE_QUERY_LIMIT",
                            "The bounded research loop exceeded its literature-query budget.",
                            supporting + turn.evidence_event_ids,
                        )
                tool_result = await registry.execute(turn)
                observation = tool_result.observation
                observations.append(observation)
                literature.extend(observation.literature_evidence)
                if self._observation_sink is not None:
                    self._observation_sink(observation)
                break
            else:
                return blocked(
                    "RESEARCH_TOOL_STEP_LIMIT",
                    "The bounded research loop exhausted its read-only tool budget before finalizing.",
                )

        return blocked(
            "RESEARCH_TOOL_STEP_LIMIT",
            "The bounded research loop exhausted its read-only tool budget before finalizing.",
        )
