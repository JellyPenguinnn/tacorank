import asyncio

from tacorank.agents.research_planner import ResearchPlanner
from tacorank.providers.research_provider import MockResearchProvider
from tacorank.research.duplicate_detection import compute_duplicate_key


def output_factory(action, spec, reason_code, reason, supporting_event_ids):
    return {
        "action": action,
        "spec": spec,
        "reason_code": reason_code,
        "reason": reason,
        "supporting_event_ids": supporting_event_ids,
    }


def make_spec(context, choice):
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
        "change_summary": "Replace pointwise loss with pairwise BPR.",
        "target_stage": "objective",
        "target_files": ["solution/loss.py"],
        "fidelity_plan": ["smoke", "proxy", "full"],
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
        "method_card_ids": ["objective_pairwise_bpr"],
        "evidence_event_ids": ["evt_000001"],
    }.items():
        setattr(spec, name, value)
    spec.duplicate_key = compute_duplicate_key(spec)
    return spec


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


def test_planner_returns_blocked_when_no_parent(planner_context):
    planner_context.eligible_frontier = []
    planner = ResearchPlanner(
        MockResearchProvider(None),
        output_factory=output_factory,
    )

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "blocked"
    assert result["reason_code"] == "NO_ELIGIBLE_PARENT"
