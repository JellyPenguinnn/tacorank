import asyncio

from tacorank.agents.research_planner import ResearchPlanner
from tacorank.providers.research_provider import MockResearchProvider
from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.schemas import ResourceDelta, TokenMeasurement


def _proposal(context, choice):
    values = {
        "schema_version": "1.0",
        "run_id": context.run_id,
        "experiment_id": "exp_0001",
        "parent_experiment_id": choice.parent.experiment_id,
        "parent_commit_sha": choice.parent.parent_commit_sha or "a" * 40,
        "context_id": context.context_id,
        "hypothesis": "A bounded pairwise mechanism improves within-user ordering.",
        "family": choice.family,
        "change_summary": "Test one atomic ranking mechanism.",
        "expected_mechanism": "It improves relative ordering without changing the score scale.",
        "success_criteria": "Stable full-fidelity primary delta exceeds the frozen threshold.",
        "falsification_condition": "The stable delta is non-positive or diagnostics collapse.",
        "estimated_cost": {
            "llm_tokens_upper_bound": 100,
            "wall_time_seconds_upper_bound": 60,
            "gpu_seconds_upper_bound": 0,
            "cost_tier": choice.cost_tier,
        },
        "method_card_ids": [choice.method_card_id],
        "evidence_event_ids": ["evt_000001"],
        "literature_evidence": [],
    }
    values["duplicate_key"] = compute_duplicate_key(values)
    return values


def test_bounded_research_turns_are_limited_and_select_legal_action(planner_context):
    seen = []

    class TurnProvider(MockResearchProvider):
        async def research_turn(self, request):
            seen.append(request)
            if request.research_turn_index == 0:
                return {"action": "inspect_frontier"}
            choice = request.legal_choices[0]
            return {
                "action": "finalize_plan",
                "selected_action_id": choice.choice_id,
                "claim": "Use the strongest legal mechanism after inspecting the frontier.",
                "hypothesis": "A bounded mechanism improves ranking.",
                "expected_mechanism": "It changes relative ordering conservatively.",
                "success_criterion": "The stable primary score improves.",
                "falsification_condition": "The stable score does not improve.",
                "confidence": 0.6,
                "evidence_event_ids": ["evt_000001"],
                "conservative_parameter_guidance": {"default": "low capacity"},
                "spec": _proposal(planner_context, choice),
            }

    observations = []
    planner = ResearchPlanner(
        TurnProvider(None),
        research_agent_mode="bounded_react",
        output_token_limit=1000,
    )
    planner.set_observation_sink(observations.append)
    result = asyncio.run(planner.propose(planner_context))

    assert result.action.value == "propose"
    assert len(seen) == 2
    assert len(observations) == 1
    assert observations[0].tool_name == "inspect_frontier"
    assert observations[0].source_event_ids == ["evt_000001"] or observations[0].source_event_ids == []
    assert result.selected_action_id == seen[0].legal_choices[0].choice_id


def test_bounded_research_rejects_invalid_final_action(planner_context):
    class InvalidProvider(MockResearchProvider):
        async def research_turn(self, request):
            return {
                "action": "finalize_plan",
                "selected_action_id": "choice_not_legal",
                "claim": "Select only a controller-approved action.",
                "hypothesis": "A bounded mechanism improves ranking.",
                "expected_mechanism": "It changes relative ordering.",
                "success_criterion": "The stable score improves.",
                "falsification_condition": "The stable score does not improve.",
                "confidence": 0.4,
                "conservative_parameter_guidance": {"default": "low capacity"},
                "evidence_event_ids": ["evt_000001"],
                "spec": {},
            }

    planner = ResearchPlanner(
        InvalidProvider(None), research_agent_mode="bounded_react"
    )
    result = asyncio.run(planner.propose(planner_context))

    assert result.action.value == "blocked"
    assert result.reason_code == "INVALID_RESEARCH_ACTION"


def test_bounded_research_stops_after_four_tool_actions(planner_context):
    seen = []

    class NonFinalizingProvider(MockResearchProvider):
        async def research_turn(self, request):
            seen.append(request)
            return {"action": "inspect_frontier"}

    observations = []
    planner = ResearchPlanner(
        NonFinalizingProvider(None),
        research_agent_mode="bounded_react",
    )
    planner.set_observation_sink(observations.append)
    result = asyncio.run(planner.propose(planner_context))

    assert result.action.value == "blocked"
    assert result.reason_code == "RESEARCH_TOOL_STEP_LIMIT"
    assert len(seen) == 5
    assert len(observations) == 4


def test_bounded_research_repairs_a_malformed_turn_without_losing_usage(
    planner_context,
):
    seen = []

    class RepairingProvider(MockResearchProvider):
        def __init__(self):
            super().__init__(None)
            self._resource_delta = ResourceDelta(
                llm_input_tokens=17,
                llm_output_tokens=5,
                token_measurement=TokenMeasurement.PROVIDER,
            )

        @property
        def resource_delta(self):
            return self._resource_delta

        async def research_turn(self, request):
            seen.append(request)
            if request.research_turn_attempt == 1:
                return {"action": "finalize_plan"}
            choice = request.legal_choices[0]
            return {
                "action": "finalize_plan",
                "selected_action_id": choice.choice_id,
                "claim": "Use one legal mechanism after repairing the turn format.",
                "hypothesis": "A bounded mechanism improves ranking.",
                "expected_mechanism": "It changes relative ordering conservatively.",
                "success_criterion": "The stable primary score improves.",
                "falsification_condition": "The stable score does not improve.",
                "confidence": 0.5,
                "evidence_event_ids": ["evt_000001"],
                "conservative_parameter_guidance": {"default": "low capacity"},
                "spec": _proposal(planner_context, choice),
            }

    provider = RepairingProvider()
    planner = ResearchPlanner(
        provider,
        research_agent_mode="bounded_react",
        research_planning_max_attempts=2,
    )
    result = asyncio.run(planner.propose(planner_context))

    assert result.action.value == "propose"
    assert [request.research_turn_index for request in seen] == [0, 0]
    assert [request.research_turn_attempt for request in seen] == [1, 2]
    assert seen[1].research_turn_error == "research_turn_schema_invalid"
    assert result.resource_delta.llm_input_tokens == 17
    assert result.resource_delta.llm_output_tokens == 5


def test_bounded_research_accounts_usage_on_exhaustion(planner_context):
    class UsageProvider(MockResearchProvider):
        @property
        def resource_delta(self):
            return ResourceDelta(
                llm_input_tokens=31,
                llm_output_tokens=7,
                token_measurement=TokenMeasurement.PROVIDER,
            )

        async def research_turn(self, request):
            return {"action": "finalize_plan"}

    planner = ResearchPlanner(
        UsageProvider(None),
        research_agent_mode="bounded_react",
        research_planning_max_attempts=2,
    )
    result = asyncio.run(planner.propose(planner_context))

    assert result.action.value == "blocked"
    assert result.reason_code == "MALFORMED_RESEARCH_TURN"
    assert result.resource_delta.llm_input_tokens == 31
    assert result.resource_delta.llm_output_tokens == 7


def test_bounded_research_normalizes_final_fields_from_current_plan(
    planner_context,
):
    class NestedEnvelopeProvider(MockResearchProvider):
        async def research_turn(self, request):
            choice = request.legal_choices[0]
            return {
                "action": "finalize_plan",
                "confidence": 0.5,
                "conservative_parameter_guidance": {
                    "default_setting": "one conservative setting",
                },
                "spec": _proposal(planner_context, choice),
            }

    planner = ResearchPlanner(
        NestedEnvelopeProvider(None),
        research_agent_mode="bounded_react",
    )
    result = asyncio.run(
        planner.propose_parallel_direction(planner_context, 0, 1)
    )

    assert result.action.value == "propose"
    assert result.selected_action_id is not None
