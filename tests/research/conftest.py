from types import SimpleNamespace

import pytest


def make_summary(
    experiment_id: str,
    *,
    parent_experiment_id: str | None = None,
    commit_sha: str = "a" * 40,
    family: str | None = None,
    fidelity: str = "full",
    score: float | None = 0.5946,
    decision: str = "accept",
    parent_eligible: bool | None = None,
    child_count: int = 0,
    actual_cost: str = "low",
):
    return SimpleNamespace(
        experiment_id=experiment_id,
        parent_experiment_id=parent_experiment_id,
        commit_sha=commit_sha,
        family=family,
        hypothesis_summary=f"Hypothesis for {experiment_id}",
        trust_verdict="accepted",
        integrity="clean",
        decision=decision,
        highest_completed_fidelity=fidelity,
        primary_score=score,
        child_count=child_count,
        actual_cost=actual_cost,
        parent_eligible=parent_eligible,
        status="accepted",
    )


@pytest.fixture
def planner_context():
    root = make_summary("exp_0000", commit_sha="a" * 40, family=None, score=0.5946)
    return SimpleNamespace(
        schema_version="1.0",
        context_id="ctx_planner_000001",
        run_id="run_20260829_a",
        contract_sha256="a" * 64,
        contract_summary=SimpleNamespace(
            resolved=True,
            allowed_families=[
                "objective",
                "temporal_history",
                "model",
                "multitask",
                "duration_bias",
                "ensemble",
                "other",
            ],
            protected_paths=["evaluate.py", "contract/COMPETITION.md"],
            editable_paths=["solution", "research"],
        ),
        baseline=root,
        current_best=root,
        eligible_frontier=[root],
        family_history=[],
        active_lessons=[],
        method_cards=[SimpleNamespace(method_id="objective_pairwise_bpr")],
        remaining_budget=SimpleNamespace(
            remaining_llm_tokens=10_000,
            remaining_wall_time_seconds=10_000,
            remaining_gpu_seconds=10_000,
            remaining_experiments=10,
            remaining_public_queries=10,
        ),
        convergence=SimpleNamespace(
            patience=3,
            consecutive_non_improving_full_evaluations=0,
            full_evaluations_completed=0,
        ),
        public_validation_queries=0,
        source_event_ids=["evt_000001"],
    )
