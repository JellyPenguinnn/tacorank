from dataclasses import replace
from types import SimpleNamespace

from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.research.plan_validation import PlanValidator
from tacorank.schemas import CostTier


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
        component_experiment_ids=[],
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


def test_validator_allows_policy_authorized_soft_refinement(planner_context):
    from tacorank.research.search_policy import SearchPolicy

    latest = SimpleNamespace(
        experiment_id="exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        hypothesis_summary="Pairwise metric tradeoff",
        trust_verdict="negative",
        stability="not_applicable",
        integrity="clean",
        decision="prune",
        highest_completed_fidelity="proxy",
        population="internal_proxy",
        output_accepted=True,
        primary_score=0.593,
        metric_deltas={"GAUC": 0.006, "nDCG@5": -0.008},
        parent_delta=-0.001,
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
        child_count=0,
        actual_cost="medium",
        parent_eligible=False,
        best_eligible=False,
        status="pruned",
        duplicate_key="prior",
        method_card_ids=["objective_pairwise_bpr"],
        component_experiment_ids=[],
    )
    planner_context.family_history = [latest]
    choice = SearchPolicy().choose(planner_context)
    spec = make_spec(
        planner_context,
        experiment_id="exp_0002",
        parent_experiment_id="exp_0001",
        parent_commit_sha="b" * 40,
        family="objective",
        change_summary="Add one listwise refinement.",
        method_card_ids=["objective_listwise_user_softmax"],
    )

    result = PlanValidator().validate(spec, planner_context, choice=choice)

    assert choice.phase == "refinement"
    assert result.accepted, result.errors


def test_validator_accepts_soft_component_for_bounded_ensemble(planner_context):
    from tacorank.research.search_policy import SearchPolicy

    latest = SimpleNamespace(
        experiment_id="exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="temporal_history",
        hypothesis_summary="Diverse history residual",
        trust_verdict="negative",
        stability="not_applicable",
        integrity="clean",
        decision="prune",
        highest_completed_fidelity="proxy",
        population="internal_proxy",
        output_accepted=True,
        primary_score=0.5906,
        metric_deltas={"GAUC": -0.003, "nDCG@5": -0.005},
        parent_delta=-0.004,
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
        child_count=0,
        actual_cost="medium",
        parent_eligible=False,
        best_eligible=False,
        status="pruned",
        duplicate_key="prior",
        method_card_ids=["temporal_history_compact"],
        component_experiment_ids=[],
    )
    planner_context.family_history = [latest]
    choice = SearchPolicy().choose(planner_context)
    spec = make_spec(
        planner_context,
        experiment_id="exp_0002",
        family="ensemble",
        change_summary="Blend the trusted parent with exp_0001 at one fixed weight.",
        method_card_ids=["ensemble_diverse_residual_candidate"],
        component_experiment_ids=["exp_0001"],
        estimated_cost=SimpleNamespace(
            llm_tokens_upper_bound=1000,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=60,
            cost_tier="low",
        ),
    )

    result = PlanValidator().validate(spec, planner_context, choice=choice)

    assert choice.phase == "ensemble"
    assert result.accepted, result.errors


def test_validator_rejects_severely_regressed_ensemble_component(planner_context):
    component = SimpleNamespace(
        experiment_id="exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="model",
        trust_verdict="negative",
        stability="not_applicable",
        integrity="clean",
        decision="prune",
        highest_completed_fidelity="proxy",
        population="internal_proxy",
        output_accepted=True,
        parent_delta=-0.05,
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.4,
        child_count=0,
        parent_eligible=False,
        best_eligible=False,
        status="pruned",
        method_card_ids=["model_compact_ranker"],
    )
    planner_context.family_history = [component]
    spec = make_spec(
        planner_context,
        family="ensemble",
        method_card_ids=["ensemble_diverse_residual_candidate"],
        component_experiment_ids=["exp_0001"],
        estimated_cost=SimpleNamespace(
            llm_tokens_upper_bound=1000,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=60,
            cost_tier="low",
        ),
    )

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "INELIGIBLE_ENSEMBLE_COMPONENT" in result.errors


def test_validator_requires_ensemble_component_in_plan_narrative(planner_context):
    component = SimpleNamespace(
        experiment_id="exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="temporal_history",
        trust_verdict="negative",
        stability="not_applicable",
        integrity="clean",
        decision="prune",
        highest_completed_fidelity="proxy",
        population="internal_proxy",
        output_accepted=True,
        parent_delta=-0.004,
        metric_deltas={"GAUC": -0.003, "nDCG@5": -0.005},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
        child_count=0,
        parent_eligible=False,
        best_eligible=False,
        status="pruned",
        method_card_ids=["temporal_history_compact"],
    )
    planner_context.family_history = [component]
    spec = make_spec(
        planner_context,
        family="ensemble",
        method_card_ids=["ensemble_diverse_residual_candidate"],
        component_experiment_ids=["exp_0001"],
        change_summary="Blend one unspecified secondary scorer.",
        estimated_cost=SimpleNamespace(
            llm_tokens_upper_bound=1000,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=60,
            cost_tier="low",
        ),
    )

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "ENSEMBLE_COMPONENT_NOT_DESCRIBED" in result.errors


def test_validator_rejects_implementation_details_and_unknown_evidence(
    planner_context,
):
    spec = make_spec(
        planner_context,
        target_files=["evaluate.py"],
        evidence_event_ids=["evt_not_in_context"],
    )

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "PLANNER_IMPLEMENTATION_DETAIL_FORBIDDEN" in result.errors
    assert "EVIDENCE_OUTSIDE_CONTEXT" in result.errors


def test_validator_accepts_without_code_policy_in_context(planner_context):
    planner_context.target_interface_excerpts = {}
    planner_context.contract_summary.editable_paths = []
    planner_context.contract_summary.protected_paths = []

    result = PlanValidator().validate(make_spec(planner_context), planner_context)

    assert result.accepted, result.errors


def test_validator_rejects_code_specific_narrative(
    planner_context,
):
    spec = make_spec(
        planner_context,
        change_summary="Edit solution/candidate.py to change the objective.",
    )

    result = PlanValidator().validate(spec, planner_context)

    assert not result.accepted
    assert "CODE_SPECIFIC_PLAN_FORBIDDEN" in result.errors


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
        estimated_cost=SimpleNamespace(
            llm_tokens_upper_bound=1000,
            wall_time_seconds_upper_bound=60,
            gpu_seconds_upper_bound=60,
            cost_tier=CostTier.MEDIUM,
        ),
    )

    result = PlanValidator().validate(spec, planner_context)

    assert result.accepted


def test_validator_rejects_planner_owned_fidelity_plan(planner_context):
    result = PlanValidator().validate(
        make_spec(planner_context, fidelity_plan=["smoke", "bogus"]),
        planner_context,
    )

    assert not result.accepted
    assert "PLANNER_IMPLEMENTATION_DETAIL_FORBIDDEN" in result.errors


def test_validator_rejects_even_valid_planner_owned_fidelity_plan(planner_context):
    result = PlanValidator().validate(
        make_spec(planner_context, fidelity_plan=["smoke", "smoke", "full"]),
        planner_context,
    )

    assert not result.accepted
    assert "PLANNER_IMPLEMENTATION_DETAIL_FORBIDDEN" in result.errors


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
