from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tacorank.research.portfolio import load_method_cards
from tacorank.research.playbook import load_improvement_playbook


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
    stability: str = "confirmed",
    trust_verdict: str = "accepted",
    integrity: str = "clean",
    population: str = "public_validation",
    output_accepted: bool | None = True,
    metric_deltas=None,
    parent_delta: float | None = 0.003,
    prediction_change: float | None = 1.0,
    prediction_spearman_vs_parent: float | None = 0.5,
    method_card_ids=None,
    component_experiment_ids=None,
    best_eligible: bool = False,
):
    return SimpleNamespace(
        experiment_id=experiment_id,
        parent_experiment_id=parent_experiment_id,
        commit_sha=commit_sha,
        family=family,
        hypothesis_summary=f"Hypothesis for {experiment_id}",
        trust_verdict=trust_verdict,
        stability=stability,
        integrity=integrity,
        trust_flags=[],
        decision=decision,
        highest_completed_fidelity=fidelity,
        population=population,
        output_accepted=output_accepted,
        output_checks={},
        output_violations=[],
        primary_score=score,
        child_count=child_count,
        actual_cost=actual_cost,
        metric_deltas=metric_deltas or {},
        parent_delta=parent_delta,
        prediction_change=prediction_change,
        prediction_spearman_vs_parent=prediction_spearman_vs_parent,
        method_card_ids=method_card_ids or [],
        component_experiment_ids=component_experiment_ids or [],
        parent_eligible=parent_eligible,
        best_eligible=best_eligible,
        status="accepted",
    )


@pytest.fixture
def planner_context():
    root = make_summary("exp_0000", commit_sha="a" * 40, family=None, score=0.5946)
    return SimpleNamespace(
        schema_version="1.0",
        context_id="ctx_0123456789abcdef",
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
            protected_paths=[],
            editable_paths=[],
            allowed_data=[
                "train_interactions",
                "public_validation",
                "user_id",
                "video_id",
                "author_id",
                "tab",
                "date",
                "duration_ms",
                "long_view",
                "verified_predictions",
            ],
            research_capabilities=[
                "baseline_parity",
                "objective_data_frame_verified",
                "verified_best_prediction",
            ],
            active_prohibitions=[],
            data_manifest_sha256="b" * 64,
            evaluator_sha256="c" * 64,
            epsilon=0.002,
            prediction_change_no_op_threshold=0.001,
        ),
        baseline=root,
        current_best=root,
        eligible_frontier=[root],
        family_history=[],
        playbook=load_improvement_playbook(
            Path(__file__).parents[2] / "research/CURRENT_RUN_IMPROVEMENT_PLAN.md",
            source_path="research/CURRENT_RUN_IMPROVEMENT_PLAN.md",
        ),
        active_lessons=[],
        method_cards=load_method_cards(
            Path(__file__).parents[2] / "research" / "methods"
        ).cards,
        target_interface_excerpts={},
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
