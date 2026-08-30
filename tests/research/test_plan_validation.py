from dataclasses import replace
from types import SimpleNamespace

from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.research.plan_validation import PlanValidator
from tacorank.schemas import CostTier, Fidelity


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
        target_files=["solution/candidate.py"],
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
    if not hasattr(spec, "duplicate_key"):
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


def test_validator_rejects_target_outside_editable_paths(planner_context):
    spec = make_spec(planner_context, target_files=["src/tacorank/train.py"])

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "TARGET_OUTSIDE_EDITABLE_PATHS" in result.errors


def test_validator_rejects_plan_that_does_not_touch_real_entrypoint(
    planner_context,
):
    spec = make_spec(planner_context, target_files=["solution/train.py"])

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "TARGET_INTERFACE_NOT_TOUCHED" in result.errors
    assert "METHOD_IMPLEMENTATION_TARGET_NOT_TOUCHED" in result.errors


def test_validator_allows_helper_only_alongside_real_entrypoint(planner_context):
    spec = make_spec(
        planner_context,
        target_files=["solution/candidate.py", "solution/train.py"],
    )

    result = PlanValidator().validate(spec, planner_context)

    assert result.accepted


def test_validator_fails_closed_without_target_interfaces(planner_context):
    planner_context.target_interface_excerpts = {}

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "TARGET_INTERFACES_MISSING" in result.errors
    assert "METHOD_IMPLEMENTATION_TARGET_UNAUTHORIZED" in result.errors


def test_validator_fails_closed_without_editable_paths(planner_context):
    planner_context.contract_summary.editable_paths = []

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "CONTRACT_EDITABLE_PATHS_MISSING" in result.errors
    assert "TARGET_OUTSIDE_EDITABLE_PATHS" in result.errors


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
        experiment_id="bad/id",
        parent_commit_sha="not-a-commit",
    )

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "INVALID_EXPERIMENT_ID" in result.errors
    assert "INVALID_PARENT_COMMIT_SHA" in result.errors


def test_validator_normalizes_shared_schema_enums(planner_context):
    spec = make_spec(
        planner_context,
        fidelity_plan=[Fidelity.SMOKE, Fidelity.PROXY, Fidelity.FULL],
        estimated_cost=SimpleNamespace(
            llm_tokens_upper_bound=1000,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=60,
            cost_tier=CostTier.MEDIUM,
        ),
    )

    result = PlanValidator().validate(spec, planner_context)

    assert result.accepted


def test_validator_rejects_unknown_fidelity_without_crashing(planner_context):
    result = PlanValidator().validate(
        make_spec(planner_context, fidelity_plan=["smoke", "bogus"]),
        planner_context,
    )

    assert not result.accepted
    assert "INVALID_FIDELITY_PLAN" in result.errors


def test_validator_rejects_duplicate_fidelity(planner_context):
    result = PlanValidator().validate(
        make_spec(planner_context, fidelity_plan=["smoke", "smoke", "full"]),
        planner_context,
    )

    assert not result.accepted
    assert "DUPLICATE_FIDELITY" in result.errors
    assert "NON_MONOTONIC_FIDELITY_PLAN" in result.errors


def test_validator_requires_one_policy_selected_method_card(planner_context):
    result = PlanValidator().validate(
        make_spec(planner_context, method_card_ids=[]),
        planner_context,
        choice=SimpleNamespace(
            parent=SimpleNamespace(experiment_id="exp_0000"),
            family="objective",
            method_card_id="objective_pairwise_bpr",
        ),
    )

    assert not result.accepted
    assert "METHOD_CARD_REQUIRED" in result.errors
    assert "METHOD_POLICY_MISMATCH" in result.errors


def test_validator_enforces_method_status(planner_context):
    card = next(
        card
        for card in planner_context.method_cards
        if card.method_id == "objective_pairwise_bpr"
    )
    planner_context.method_cards = [replace(card, status="blocked")]

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "METHOD_STATUS_NOT_CANDIDATE" in result.errors


def test_validator_enforces_method_prerequisites(planner_context):
    card = next(
        card
        for card in planner_context.method_cards
        if card.method_id == "objective_pairwise_bpr"
    )
    planner_context.method_cards = [
        replace(card, prerequisites=("human_approval_not_recorded",))
    ]

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "METHOD_PREREQUISITES_UNSATISFIED" in result.errors


def test_validator_enforces_method_allowed_data(planner_context):
    card = next(
        card
        for card in planner_context.method_cards
        if card.method_id == "objective_pairwise_bpr"
    )
    planner_context.method_cards = [replace(card, allowed_data=("hidden_labels",))]

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "METHOD_DATA_NOT_ALLOWED" in result.errors


def test_validator_enforces_method_prohibitions(planner_context):
    card = next(
        card
        for card in planner_context.method_cards
        if card.method_id == "objective_pairwise_bpr"
    )
    planner_context.method_cards = [
        replace(card, prohibition_conditions=("leakage_detected",))
    ]
    planner_context.contract_summary.active_prohibitions = ["leakage_detected"]

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "METHOD_PROHIBITED" in result.errors


def test_validator_rejects_method_family_mismatch(planner_context):
    card = next(
        card
        for card in planner_context.method_cards
        if card.method_id == "objective_pairwise_bpr"
    )
    planner_context.method_cards = [replace(card, family="model")]

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "METHOD_FAMILY_MISMATCH" in result.errors


def test_validator_rejects_underestimated_method_cost(planner_context):
    result = PlanValidator().validate(
        make_spec(
            planner_context,
            estimated_cost=SimpleNamespace(
                llm_tokens_upper_bound=1000,
                wall_time_seconds_upper_bound=60,
                gpu_seconds_upper_bound=60,
                cost_tier="low",
            ),
        ),
        planner_context,
    )

    assert not result.accepted
    assert "METHOD_COST_UNDERESTIMATED" in result.errors


def test_validator_fails_closed_without_allowed_data(planner_context):
    planner_context.contract_summary.allowed_data = []

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert not result.accepted
    assert "CONTRACT_ALLOWED_DATA_MISSING" in result.errors
