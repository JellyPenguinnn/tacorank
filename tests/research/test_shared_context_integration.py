import asyncio
from types import SimpleNamespace

from tacorank.agents.research_planner import ResearchPlanner
from tacorank.providers.research_provider import MockResearchProvider
from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.schemas import CostEstimate, CostTier, ExperimentSpec, Fidelity, PlannerAction


def test_planner_consumes_context_builder_output(harness, baseline_evaluation):
    harness.bootstrap(baseline_evaluation)
    context = harness.context_builder.build_planner(harness.events())

    def make_spec(request):
        choice = request.policy_choice
        values = {
            "run_id": context.run_id,
            "experiment_id": "exp_0001",
            "parent_experiment_id": choice.parent.experiment_id,
            "parent_commit_sha": choice.parent.parent_commit_sha,
            "context_id": context.context_id,
            "hypothesis": "Pairwise ranking aligns training with the evaluator.",
            "family": choice.family,
            "change_summary": "Replace pointwise loss with bounded pairwise BPR.",
            "target_stage": "objective",
            "target_files": ["solution/model.py"],
            "fidelity_plan": [Fidelity.SMOKE, Fidelity.PROXY, Fidelity.FULL],
            "expected_mechanism": "Improve within-user relative ordering.",
            "success_criteria": "Trusted full score improves beyond epsilon.",
            "falsification_condition": "No stable improvement over the baseline.",
                "estimated_cost": CostEstimate(
                    llm_tokens_upper_bound=500,
                    wall_time_seconds_upper_bound=60,
                    gpu_seconds_upper_bound=0,
                    cost_tier=CostTier.MEDIUM,
                ),
                "method_card_ids": [choice.method_card_id],
            "evidence_event_ids": list(context.source_event_ids),
        }
        values["duplicate_key"] = compute_duplicate_key(SimpleNamespace(**values))
        return ExperimentSpec(**values)

    provider = MockResearchProvider(make_spec)
    output = asyncio.run(ResearchPlanner(provider).propose(context))

    assert context.context_id.startswith("ctx_")
    assert len(context.context_id) == 20
    assert [item.experiment_id for item in context.eligible_frontier] == ["baseline"]
    assert output.action == PlannerAction.PROPOSE
    assert output.spec.parent_experiment_id == "baseline"
    assert len(provider.requests) == 1
