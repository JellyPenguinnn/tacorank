import asyncio

import pytest

from tacorank.agents.research_planner import ResearchPlanner
from tacorank.providers.research_provider import MockResearchProvider
from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.schemas import LiteratureEvidence, ResourceDelta

from .conftest import make_summary


def output_factory(action, spec, reason_code, reason, supporting_event_ids):
    return {
        "action": action,
        "spec": spec,
        "reason_code": reason_code,
        "reason": reason,
        "supporting_event_ids": supporting_event_ids,
    }


def make_spec(context, choice, literature_evidence=()):
    spec = type("Spec", (), {})()
    for name, value in {
        "schema_version": "1.0",
        "run_id": context.run_id,
        "experiment_id": "exp_0001",
        "parent_experiment_id": choice.parent.experiment_id,
        "parent_commit_sha": choice.parent.parent_commit_sha or ("a" * 40),
        "context_id": context.context_id,
        "hypothesis": "Pairwise loss aligns training with ranking metrics.",
        "family": choice.family,
        "change_summary": "Add the policy-selected bounded mechanism.",
        "expected_mechanism": "Improve within-user ordering.",
        "success_criteria": type("Criteria", (), {"full_parent_delta_min": 0.002})(),
        "falsification_condition": "No stable improvement.",
        "estimated_cost": type(
            "Cost",
            (),
            {
                "llm_tokens_upper_bound": 1000,
                "wall_time_seconds_upper_bound": 60,
                "gpu_seconds_upper_bound": 60,
                "cost_tier": "medium",
            },
        )(),
        "method_card_ids": [choice.method_card_id],
        "evidence_event_ids": ["evt_000001"],
        "literature_evidence": list(literature_evidence),
    }.items():
        setattr(spec, name, value)
    spec.duplicate_key = compute_duplicate_key(spec)
    return spec


def paper_evidence():
    return LiteratureEvidence(
        evidence_id="lit_paper_001",
        paper_id="W1234567890",
        title="Bayesian Personalized Ranking from Implicit Feedback",
        abstract="Pairwise ranking optimizes relative preference ordering.",
        year=2009,
        authors=["Steffen Rendle"],
        venue="UAI",
        citation_count=1000,
        influential_citation_count=100,
        url="https://openalex.org/W1234567890",
        query="Bayesian personalized ranking recommender systems",
    )


def test_planner_returns_one_valid_proposal(planner_context):
    provider = MockResearchProvider(lambda request: make_spec(planner_context, request.policy_choice))
    planner = ResearchPlanner(
        provider,
        output_factory=output_factory,
        input_token_limit=2000,
        output_token_limit=1000,
    )

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "propose"
    assert result["spec"].experiment_id == "exp_0001"
    assert len(provider.requests) == 1


def test_parallel_workers_receive_distinct_method_cards(planner_context):
    contract = type("Contract", (), vars(planner_context.contract_summary))()
    contract.allowed_families = [
        *contract.allowed_families,
        "features",
        "sampling",
    ]
    context_values = vars(planner_context).copy()
    context_values["contract_summary"] = contract
    context = type("Context", (), context_values)()
    provider = MockResearchProvider(lambda request: make_spec(context, request.policy_choice))
    planner = ResearchPlanner(
        provider,
        output_factory=output_factory,
        input_token_limit=2000,
        output_token_limit=1000,
    )

    async def propose_all():
        capacity = planner.parallel_direction_capacity(context)
        outputs = await asyncio.gather(
            *(
                planner.propose_parallel_direction(context, index, capacity)
                for index in range(capacity)
            )
        )
        return capacity, outputs

    capacity, outputs = asyncio.run(propose_all())
    method_ids = [
        request.policy_choice.method_card_id for request in provider.requests
    ]

    assert capacity >= 7
    assert len(outputs) == capacity
    assert len(method_ids) == len(set(method_ids))


def test_planner_researches_and_requires_immutable_literature(planner_context):
    evidence = paper_evidence()

    class Skill:
        def __init__(self):
            self.calls = []
            self.preflight_calls = 0
            self.resource_delta = ResourceDelta(wall_time_ms=25)

        def preflight(self):
            self.preflight_calls += 1

        async def research(self, context, choice):
            self.calls.append((context, choice))
            return [evidence]

    skill = Skill()
    provider = MockResearchProvider(
        lambda request: make_spec(
            planner_context,
            request.policy_choice,
            request.literature_evidence,
        )
    )
    planner = ResearchPlanner(
        provider,
        output_factory=output_factory,
        literature_skill=skill,
    )

    planner.preflight()
    result = asyncio.run(planner.propose(planner_context))

    assert skill.preflight_calls == 1
    assert len(skill.calls) == 1
    assert provider.requests[0].literature_evidence == (evidence,)
    assert result["action"] == "propose"
    assert result["spec"].literature_evidence == [evidence]
    assert result["resource_delta"].wall_time_ms == 25


def test_planner_treats_advisory_bank_evidence_as_optional(planner_context):
    evidence = paper_evidence()

    class Skill:
        requires_citation = False
        resource_delta = ResourceDelta()

        async def research(self, context, choice):
            del context, choice
            return [evidence]

    provider = MockResearchProvider(
        lambda request: make_spec(planner_context, request.policy_choice)
    )
    planner = ResearchPlanner(
        provider,
        output_factory=output_factory,
        literature_skill=Skill(),
    )

    result = asyncio.run(planner.propose(planner_context))

    assert provider.requests[0].literature_required is False
    assert result["action"] == "propose"
    assert result["spec"].literature_evidence == []


def test_planner_returns_blocked_when_no_parent(planner_context):
    planner_context.eligible_frontier = []
    planner = ResearchPlanner(
        MockResearchProvider(None),
        output_factory=output_factory,
    )

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "blocked"
    assert result["reason_code"] == "NO_ELIGIBLE_PARENT"


@pytest.mark.parametrize(
    ("overrides", "reason_code"),
    [
        ({"output_accepted": False}, "OUTPUT_CHECK_REJECTED"),
        (
            {"trust_verdict": "suspicious", "integrity": "compromised"},
            "SUSPICIOUS_RESULT_REQUIRES_QUARANTINE",
        ),
        ({"stability": "unstable"}, "UNSTABLE_RESULT_REQUIRES_CONFIRMATION"),
        (
            {"fidelity": "proxy", "population": "internal_proxy"},
            "FIDELITY_PROMOTION_REQUIRED",
        ),
        ({"output_accepted": None}, "RESULT_NOT_BRANCHABLE"),
        ({"population": "unbiased_audit"}, "RESULT_NOT_BRANCHABLE"),
    ],
)
def test_guardrail_blocks_never_call_provider(
    planner_context, overrides, reason_code
):
    planner_context.family_history = [
        make_summary(
            "exp_0001",
            parent_experiment_id="exp_0000",
            family="objective",
            parent_eligible=False,
            method_card_ids=["objective_pairwise_bpr"],
            **overrides,
        )
    ]
    provider = MockResearchProvider(None)
    planner = ResearchPlanner(provider, output_factory=output_factory)

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "blocked"
    assert result["reason_code"] == reason_code
    assert provider.requests == []


def test_suspicious_result_is_quarantined_without_stopping_planner(
    planner_context,
):
    planner_context.family_history = [
        make_summary(
            "exp_0001",
            parent_experiment_id="exp_0000",
            family="objective",
            parent_eligible=False,
            trust_verdict="suspicious",
            integrity="inconclusive",
            method_card_ids=["objective_pairwise_bpr"],
        )
    ]
    provider = MockResearchProvider(
        lambda request: make_spec(planner_context, request.policy_choice)
    )
    planner = ResearchPlanner(provider, output_factory=output_factory)

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "propose"
    assert result["reason_code"] == "SUSPICIOUS_RESULT_QUARANTINED"
    assert result["spec"].family == "temporal_history"
    assert result["spec"].parent_experiment_id == "exp_0000"
    assert len(provider.requests) == 1
