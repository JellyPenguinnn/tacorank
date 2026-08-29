from types import SimpleNamespace

from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.research.plan_validation import PlanValidator


def make_spec(planner_context, **overrides):
    values = dict(
        schema_version="1.0",
        run_id=planner_context.run_id,
        experiment_id="exp_0001",
        parent_experiment_id="exp_0000",
        parent_commit_sha="a" * 40,
        context_id=planner_context.context_id,
        hypothesis="Pairwise loss aligns training with ranking metrics.",
        family="objective",
        change_summary="Replace pointwise loss with bounded pairwise loss.",
        target_stage="objective",
        target_files=["solution/loss.py"],
        fidelity_plan=["smoke", "proxy", "full"],
        expected_mechanism="Optimize within-user relative ordering.",
        success_criteria=SimpleNamespace(full_parent_delta_min=0.002),
        falsification_condition="No stable improvement over the parent.",
        estimated_cost=SimpleNamespace(
            llm_tokens_upper_bound=1000,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=60,
            cost_tier="medium",
        ),
        method_card_ids=["objective_pairwise_bpr"],
        evidence_event_ids=["evt_000001"],
    )
    values.update(overrides)
    spec = SimpleNamespace(**values)
    spec.duplicate_key = compute_duplicate_key(spec)
    return spec


def test_validator_accepts_contract_compatible_plan(planner_context):
    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert result.accepted
    assert result.errors == ()


def test_validator_rejects_protected_path_and_unknown_evidence(planner_context):
    spec = make_spec(
        planner_context,
        target_files=["evaluate.py"],
        evidence_event_ids=["evt_not_in_context"],
    )

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "PROTECTED_TARGET_PATH" in result.errors
    assert "EVIDENCE_OUTSIDE_CONTEXT" in result.errors


def test_validator_rejects_unresolved_contract(planner_context):
    planner_context.contract_summary.resolved = False

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "CONTRACT_UNRESOLVED" in result.errors


def test_validator_rejects_duplicate_experiment(planner_context):
    spec = make_spec(planner_context)
    planner_context.family_history = [SimpleNamespace(duplicate_key=spec.duplicate_key)]

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "DUPLICATE_EXPERIMENT" in result.errors


def test_validator_enforces_memory_schema_identifiers(planner_context):
    spec = make_spec(
        planner_context,
        experiment_id="exp_bad",
        parent_commit_sha="not-a-commit",
    )

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "INVALID_EXPERIMENT_ID" in result.errors
    assert "INVALID_PARENT_COMMIT_SHA" in result.errors
