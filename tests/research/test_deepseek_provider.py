from __future__ import annotations

import asyncio
import io
import json
from urllib.error import HTTPError

import pytest

from tacorank.agents.research_planner import ResearchPlanner
from tacorank.cli import _planner_for
from tacorank.providers import deepseek as deepseek_module
from tacorank.providers.deepseek import DeepSeekResearchProvider
from tacorank.providers.research_provider import ProviderError, ProviderRequest
from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.research.literature import OpenAlexLiteratureSkill
from tacorank.research.plan_validation import PlanValidator
from tacorank.research.search_policy import SearchPolicy
from tacorank.schemas import LiteratureEvidence, ResearchProposal, TokenMeasurement

from .conftest import make_summary


def candidate(**updates):
    value = {
        "hypothesis": "Pairwise BPR should improve within-user ranking.",
        "change_summary": "Replace pointwise loss with pairwise BPR.",
        "expected_mechanism": "Optimize relative positive-negative ordering.",
        "success_criteria": "Full primary delta is at least 0.002.",
        "falsification_condition": "No trusted full-fidelity gain.",
        "estimated_cost": {
            "llm_tokens_upper_bound": 800,
            "wall_time_seconds_upper_bound": 300,
            "gpu_seconds_upper_bound": 60,
            "cost_tier": "high",
        },
        "method_card_ids": ["objective_pairwise_bpr"],
        "evidence_event_ids": ["evt_000001", "evt_not_in_context"],
        # These are deliberately hostile: the adapter must replace policy-owned fields.
        "run_id": "wrong_run",
        "experiment_id": "wrong_experiment",
        "parent_experiment_id": "wrong_parent",
        "parent_commit_sha": "f" * 40,
        "context_id": "wrong_context",
        "family": "model",
    }
    value.update(updates)
    return value


def response(value, *, finish_reason="stop", prompt_tokens=101, completion_tokens=37):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": json.dumps(value)},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def empty_response(*, prompt_tokens=101, completion_tokens=0):
    value = response(
        {},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    value["choices"][0]["message"]["content"] = ""
    return value


def output_factory(action, spec, reason_code, reason, supporting_event_ids):
    return {
        "action": action,
        "spec": spec,
        "reason_code": reason_code,
        "reason": reason,
        "supporting_event_ids": supporting_event_ids,
    }


def literature_evidence():
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


def test_deepseek_provider_constrains_policy_fields_and_records_usage(planner_context):
    calls = []
    planner_context.data_profile = {
        "profile_sha256": "d" * 64,
        "train_rows": 4,
        "score_rows": 2,
    }
    planner_context.baseline.diagnostic_metrics = {
        "user_rankable_fraction": 1.0,
    }
    prior = make_summary(
        experiment_id="exp_0001",
        family="temporal_history",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        trust_verdict="negative",
        parent_delta=-0.00007,
        prediction_change=0.0126,
    )
    prior.failure_hypotheses = [
        "Concentrated movement: a small user cohort carries most score movement."
    ]
    prior.diagnostic_best_slice = "popularity_rank.cold"
    prior.diagnostic_worst_slice = "popularity_rank.hot"
    prior.diagnostic_metrics = {
        "gain_concentration_top10pct": 1.0,
        "best_slice_delta": 0.00025,
        "worst_slice_delta": -0.00021,
    }
    planner_context.family_history = [prior]
    planner_context.active_lessons = [
        {
            "lesson_id": "lesson_001",
            "origin": "research",
            "category": "research_result",
            "tags": ["objective", "confirmed"],
            "summary": "A confirmed objective change improved validation ranking.",
            "applicability": "Clean full public validation.",
            "avoid_when": "Only proxy evidence is available.",
            "confidence": 0.9,
            "source_event_ids": ["evt_000001"],
            "source_commit_shas": ["f" * 40],
        }
    ]

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return response(candidate())

    choice = SearchPolicy().choose(planner_context)
    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    result = asyncio.run(
        provider.generate(
            ProviderRequest(
                context=planner_context,
                policy_choice=choice,
                input_token_limit=2_000,
                output_token_limit=1_000,
            )
        )
    )

    assert result["run_id"] == planner_context.run_id
    assert result["context_id"] == planner_context.context_id
    assert result["experiment_id"] == "exp_0002"
    assert result["parent_experiment_id"] == choice.parent.experiment_id
    assert result["parent_commit_sha"] == choice.parent.parent_commit_sha
    assert result["family"] == choice.family
    assert result["estimated_cost"]["cost_tier"] == choice.cost_tier
    assert result["evidence_event_ids"] == ["evt_000001"]
    assert result["duplicate_key"] == compute_duplicate_key(result)

    url, headers, payload, timeout = calls[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert headers["Authorization"] == "Bearer secret-key"
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 1_000
    assert payload["thinking"] == {"type": "enabled"}
    prompt_document = json.loads(payload["messages"][1]["content"])
    serialized_prompt = json.dumps(prompt_document, sort_keys=True)
    assert "target_interfaces" not in prompt_document["context"]
    assert "editable_paths" not in prompt_document["context"]["contract"]
    assert "protected_paths" not in prompt_document["context"]["contract"]
    assert "commit_sha" not in prompt_document["context"]["baseline"]
    assert prompt_document["context"]["baseline"]["diagnostic_metrics"] == {
        "user_rankable_fraction": 1.0,
    }
    assert prompt_document["context"]["data_profile"] == {
        "profile_sha256": "d" * 64,
        "train_rows": 4,
        "score_rows": 2,
    }
    prior_prompt = prompt_document["context"]["family_history"][0]
    assert prior_prompt["failure_hypotheses"] == prior.failure_hypotheses
    assert prior_prompt["diagnostic_best_slice"] == "popularity_rank.cold"
    assert prior_prompt["diagnostic_worst_slice"] == "popularity_rank.hot"
    assert prior_prompt["diagnostic_metrics"]["gain_concentration_top10pct"] == 1.0
    assert prompt_document["context"]["active_lessons"] == [
        {
            "lesson_id": "lesson_001",
            "origin": "research",
            "category": "research_result",
            "tags": ["objective", "confirmed"],
            "summary": "A confirmed objective change improved validation ranking.",
            "applicability": "Clean full public validation.",
            "avoid_when": "Only proxy evidence is available.",
            "confidence": 0.9,
            "source_event_ids": ["evt_000001"],
        }
    ]
    assert "source_commit_shas" not in serialized_prompt
    assert "read-only aggregate" in payload["messages"][0]["content"]
    assert "do not merely increase residual magnitude" in payload["messages"][0]["content"]
    assert all(
        "implementation_targets" not in card
        for card in prompt_document["context"]["method_cards"]
    )
    assert "solution/candidate.py" not in serialized_prompt
    assert "implementation_targets" not in serialized_prompt
    assert "source_path" not in serialized_prompt
    assert "parent_commit_sha" not in serialized_prompt
    assert "target_files" not in payload["messages"][0]["content"]
    assert "pipeline stages" in payload["messages"][0]["content"]
    system_prompt = " ".join(payload["messages"][0]["content"].split())
    assert "bounded additive residual" in system_prompt
    assert "do not propose clipping" in system_prompt
    assert "secret-key" not in json.dumps(payload)
    assert timeout == 120
    assert provider.resource_delta.llm_input_tokens == 101
    assert provider.resource_delta.llm_output_tokens == 37
    assert provider.resource_delta.token_measurement == TokenMeasurement.PROVIDER


def test_deepseek_provider_binds_campaign_variant(planner_context):
    planner_context.contract_summary.allowed_families = ["objective"]
    planner_context.research_campaign = {
        "campaign_id": "depth_test",
        "family_order": ["objective"],
        "family_budgets": {"objective": 2},
        "family_method_card_ids": {
            "objective": ["objective_pairwise_bpr"],
        },
        "family_directives": {
            "objective": "Adapt the next objective from prior evidence.",
        },
    }
    objective_history = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        method_card_ids=["objective_pairwise_bpr"],
    )
    objective_history.campaign_id = "depth_test"
    objective_history.variant_id = "objective_00"
    objective_history.variant_instruction = "Use one negative per positive."
    temporal_history = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0000",
        family="temporal_history",
        method_card_ids=["temporal_history_compact"],
    )
    temporal_history.campaign_id = "other_campaign"
    temporal_history.variant_id = "temporal_history_01"
    temporal_history.variant_instruction = "Use a seven-day author history."
    planner_context.family_history = [temporal_history, objective_history]
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return response(
            candidate(
                variant_instruction="Use four uniform negatives per positive.",
                    variant_parameters={
                        "formulation": "bpr",
                        "negative_sampling": "uniform",
                        "negative_count": 4,
                    },
                    hypothesis_evidence={
                        "observation": "There is no prior experiment yet.",
                        "source_evaluation_event_ids": ["evt_999999"],
                        "changed_factors": ["negative_count"],
                        "held_constant": ["formulation"],
                        "expected_metric_effects": {"GAUC": 0.001},
                    },
                )
            )

    choice = SearchPolicy().choose(planner_context)
    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    result = asyncio.run(
        provider.generate(
            ProviderRequest(context=planner_context, policy_choice=choice)
        )
    )

    assert result["campaign_id"] == "depth_test"
    assert result["variant_id"] == "objective_02"
    assert result["variant_instruction"] == (
        "Use four uniform negatives per positive."
    )
    assert result["variant_parameters"] == {
        "formulation": "bpr",
            "embedding_dim": 16,
            "learning_rate": 0.001,
            "epochs": 40,
        "negative_count": 4,
            "l2": 0.000001,
        "residual_scale": 0.05,
            "max_train_rows": 1141112,
    }
    assert result["hypothesis_evidence"] is None
    prompt = json.loads(calls[0]["messages"][1]["content"])
    assert [item["family"] for item in prompt["context"]["family_history"]] == [
        "objective"
    ]
    assert prompt["policy"]["variant_id"] == "objective_02"
    assert prompt["policy"]["campaign_directive"] == (
        "Adapt the next objective from prior evidence."
    )
    bpr = next(
        item
        for item in prompt["context"]["method_cards"]
        if item["method_id"] == "objective_pairwise_bpr"
    )
    assert bpr["active_parameters"] == [
        "formulation",
        "embedding_dim",
        "learning_rate",
        "epochs",
        "negative_count",
        "l2",
        "residual_scale",
        "max_train_rows",
    ]
    assert bpr["parameter_defaults"]["negative_count"] == 2
    validation = PlanValidator().validate(
        result, planner_context, choice=choice
    )
    assert validation.accepted, validation.errors


def test_deepseek_provider_derives_later_campaign_treatment_boundary(
    planner_context,
):
    planner_context.contract_summary.allowed_families = ["objective"]
    planner_context.research_campaign = {
        "campaign_id": "depth_test",
        "family_order": ["objective"],
        "family_budgets": {"objective": 3},
        "family_method_card_ids": {"objective": ["objective_pairwise_bpr"]},
        "family_directives": {"objective": "Adapt from prior evidence."},
    }
    prior = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        method_card_ids=["objective_pairwise_bpr"],
    )
    prior.campaign_id = "depth_test"
    prior.variant_id = "objective_01"
    prior.evaluation_event_id = "evt_000010"
    prior.variant_parameters = {
        "formulation": "bpr",
        "embedding_dim": 8,
        "learning_rate": 0.01,
        "epochs": 2,
        "negative_count": 2,
        "l2": 0.0001,
        "residual_scale": 0.05,
        "max_train_rows": 100000,
    }
    planner_context.family_history = [prior]
    planner_context.source_event_ids.append("evt_000010")

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        return response(
            candidate(
                variant_instruction="Increase BPR negatives from two to four.",
                variant_parameters={"negative_count": 4},
                hypothesis_evidence={
                    "observation": "The prior BPR result remained within noise.",
                    "source_evaluation_event_ids": ["evt_000010"],
                    "changed_factors": ["epochs"],
                    "held_constant": ["negative_count"],
                    "expected_metric_effects": {"GAUC": 0.001},
                },
            )
        )

    choice = SearchPolicy().choose(planner_context)
    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    result = asyncio.run(
        provider.generate(ProviderRequest(planner_context, choice))
    )

    assert result["variant_parameters"]["negative_count"] == 4
    assert result["variant_parameters"]["learning_rate"] == 0.01
    assert result["hypothesis_evidence"]["changed_factors"] == [
        "negative_count"
    ]
    assert set(result["hypothesis_evidence"]["held_constant"]) == {
        "formulation",
        "embedding_dim",
        "learning_rate",
        "epochs",
        "l2",
        "residual_scale",
        "max_train_rows",
    }
    assert "evt_000010" in result["evidence_event_ids"]
    validation = PlanValidator().validate(
        result, planner_context, choice=choice
    )
    assert validation.accepted, validation.errors


def test_deepseek_provider_binds_latest_prior_evaluation_when_citation_is_invalid(
    planner_context,
):
    planner_context.contract_summary.allowed_families = ["objective"]
    planner_context.research_campaign = {
        "campaign_id": "depth_test",
        "family_order": ["objective"],
        "family_budgets": {"objective": 3},
        "family_method_card_ids": {"objective": ["objective_pairwise_bpr"]},
        "family_directives": {"objective": "Adapt from prior evidence."},
    }
    prior = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        method_card_ids=["objective_pairwise_bpr"],
    )
    prior.campaign_id = "depth_test"
    prior.variant_id = "objective_01"
    prior.evaluation_event_id = "evt_000010"
    prior.variant_parameters = {
        "formulation": "bpr",
        "embedding_dim": 8,
        "learning_rate": 0.01,
        "epochs": 2,
        "negative_count": 2,
        "l2": 0.0001,
        "residual_scale": 0.05,
        "max_train_rows": 100000,
    }
    planner_context.family_history = [prior]
    planner_context.source_event_ids.append("evt_000010")

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        return response(
            candidate(
                variant_instruction="Increase BPR negatives from two to four.",
                variant_parameters={"negative_count": 4},
                hypothesis_evidence={
                    "observation": "The prior BPR result remained within noise.",
                    "source_evaluation_event_ids": ["evt_999999"],
                    "expected_metric_effects": {"GAUC": 0.001},
                },
            )
        )

    choice = SearchPolicy().choose(planner_context)
    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    result = asyncio.run(
        provider.generate(ProviderRequest(planner_context, choice))
    )

    assert result["hypothesis_evidence"]["source_evaluation_event_ids"] == [
        "evt_000010"
    ]
    assert "evt_000010" in result["evidence_event_ids"]
    validation = PlanValidator().validate(
        result, planner_context, choice=choice
    )
    assert validation.accepted, validation.errors


def test_deepseek_provider_enforces_control_when_all_treatment_values_change(
    planner_context,
):
    planner_context.contract_summary.allowed_families = ["temporal_history"]
    planner_context.contract_summary.research_capabilities.append(
        "strict_temporal_cutoff"
    )
    planner_context.research_campaign = {
        "campaign_id": "temporal_depth_test",
        "family_order": ["temporal_history"],
        "family_budgets": {"temporal_history": 25},
        "family_method_card_ids": {
            "temporal_history": ["temporal_history_compact"]
        },
        "family_directives": {
            "temporal_history": "Adapt from prior matched evidence."
        },
        "minimum_family_full_evaluations": 25,
        "family_convergence_patience": 25,
    }
    prior = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        method_card_ids=["objective_pairwise_bpr"],
    )
    prior.evaluation_event_id = "evt_000010"
    planner_context.family_history = [prior]
    planner_context.source_event_ids.append("evt_000010")

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        return response(
            candidate(
                method_card_ids=["temporal_history_compact"],
                variant_instruction="Test longer, more strongly shrunk history.",
                variant_parameters={
                    "formulation": "temporal_history",
                    "residual_scale": 0.2,
                    "max_train_rows": 200000,
                    "history_decay_days": 30.0,
                    "history_shrinkage": 50.0,
                },
                hypothesis_evidence={
                    "observation": "The prior objective result remained within noise.",
                    "source_evaluation_event_ids": ["evt_000010"],
                    "changed_factors": ["history_decay_days"],
                    "held_constant": ["max_train_rows"],
                    "expected_metric_effects": {"GAUC": 0.001},
                },
            )
        )

    choice = SearchPolicy().choose(planner_context)
    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    result = asyncio.run(
        provider.generate(ProviderRequest(planner_context, choice))
    )

    assert result["variant_parameters"]["max_train_rows"] == 1141112
    assert "matched control(s)" in result["variant_instruction"]
    assert result["hypothesis_evidence"]["source_evaluation_event_ids"] == [
        "evt_000010"
    ]
    assert result["hypothesis_evidence"]["held_constant"] == [
        "max_train_rows"
    ]
    assert set(result["hypothesis_evidence"]["changed_factors"]) == {
        "formulation",
        "residual_scale",
        "history_decay_days",
        "history_shrinkage",
    }
    validation = PlanValidator().validate(
        result, planner_context, choice=choice
    )
    assert validation.accepted, validation.errors


def test_deepseek_provider_does_not_fabricate_missing_hypothesis_evidence(
    planner_context,
):
    planner_context.contract_summary.allowed_families = ["objective"]
    planner_context.research_campaign = {
        "campaign_id": "objective_depth_test",
        "family_order": ["objective"],
        "family_budgets": {"objective": 25},
        "family_method_card_ids": {"objective": ["objective_pairwise_bpr"]},
        "family_directives": {"objective": "Adapt from prior evidence."},
        "minimum_family_full_evaluations": 25,
        "family_convergence_patience": 25,
    }
    prior = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="objective",
        method_card_ids=["objective_pairwise_bpr"],
    )
    prior.campaign_id = "objective_depth_test"
    prior.evaluation_event_id = "evt_000010"
    prior.variant_parameters = {
        "formulation": "bpr",
        "embedding_dim": 8,
        "learning_rate": 0.01,
        "epochs": 2,
        "negative_count": 2,
        "l2": 0.0001,
        "residual_scale": 0.05,
        "max_train_rows": 100000,
    }
    planner_context.family_history = [prior]
    planner_context.source_event_ids.append("evt_000010")

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        return response(
            candidate(
                variant_instruction="Increase BPR negatives.",
                variant_parameters={"negative_count": 4},
                hypothesis_evidence=None,
            )
        )

    choice = SearchPolicy().choose(planner_context)
    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    result = asyncio.run(
        provider.generate(ProviderRequest(planner_context, choice))
    )

    assert result["hypothesis_evidence"] is None
    validation = PlanValidator().validate(
        result, planner_context, choice=choice
    )
    assert not validation.accepted
    assert "HYPOTHESIS_EVIDENCE_REQUIRED_AFTER_PRIOR_EVALUATION" in (
        validation.errors
    )


def test_deepseek_provider_preserves_policy_owned_ensemble_components(
    planner_context,
):
    component = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        family="temporal_history",
        fidelity="proxy",
        population="internal_proxy",
        decision="prune",
        parent_eligible=False,
        trust_verdict="negative",
        stability="not_applicable",
        parent_delta=-0.004,
        metric_deltas={"GAUC": -0.003, "nDCG@5": -0.005},
        prediction_change=0.8,
        prediction_spearman_vs_parent=0.6,
        method_card_ids=["temporal_history_compact"],
    )
    planner_context.family_history = [component]
    planner_context.refinement_frontier_ids = []
    planner_context.ensemble_candidate_ids = ["exp_0001"]
    choice = SearchPolicy().choose(planner_context)
    calls = []

    def transport(url, headers, payload, timeout):
        del url, headers, timeout
        calls.append(payload)
        return response(candidate(component_experiment_ids=["hostile_component"]))

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    result = asyncio.run(
        provider.generate(ProviderRequest(planner_context, choice))
    )

    assert choice.phase == "ensemble"
    assert result["family"] == "ensemble"
    assert result["method_card_ids"] == ["ensemble_diverse_residual_candidate"]
    assert result["component_experiment_ids"] == ["exp_0001"]
    prompt = json.loads(calls[0]["messages"][1]["content"])
    assert prompt["policy"]["component_experiment_ids"] == ["exp_0001"]
    assert prompt["context"]["ensemble_candidate_ids"] == ["exp_0001"]


def test_research_planner_requests_one_deepseek_repair(planner_context):
    responses = [
        response(candidate(hypothesis=""), prompt_tokens=100, completion_tokens=20),
        response(candidate(), prompt_tokens=90, completion_tokens=30),
    ]
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return responses.pop(0)

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    planner = ResearchPlanner(
        provider,
        output_factory=output_factory,
        input_token_limit=2_000,
        output_token_limit=1_000,
    )

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "propose"
    assert len(requests) == 2
    repair_prompt = json.loads(requests[1]["messages"][1]["content"])
    assert "MISSING_HYPOTHESIS" in repair_prompt["repair"]["validation_errors"]
    assert "parent_commit_sha" not in repair_prompt["repair"]["previous_candidate"]
    assert result["resource_delta"].llm_input_tokens == 190
    assert result["resource_delta"].llm_output_tokens == 50


def test_deepseek_provider_discards_unsolicited_implementation_details(
    planner_context,
):
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return response(
            candidate(
                target_stage="training_objective",
                target_files=["src/tacorank/train.py"],
                fidelity_plan=["full"],
            )
        )

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    choice = SearchPolicy().choose(planner_context)
    result = asyncio.run(
        provider.generate(ProviderRequest(planner_context, choice))
    )

    assert "target_stage" not in result
    assert "target_files" not in result
    assert "fidelity_plan" not in result
    assert len(requests) == 1


def test_research_planner_repairs_code_specific_narrative(planner_context):
    responses = [
        response(
            candidate(
                change_summary=(
                    "Compare pairwise/listwise objectives in solution/candidate.py."
                )
            )
        ),
        response(
            candidate(
                change_summary=(
                    "Compare positive/negative preference ordering for user/item "
                    "ranking."
                )
            )
        ),
    ]
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return responses.pop(0)

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    planner = ResearchPlanner(
        provider,
        output_factory=output_factory,
        input_token_limit=2_000,
        output_token_limit=1_000,
    )

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "propose"
    repair_prompt = json.loads(requests[1]["messages"][1]["content"])
    assert "CODE_SPECIFIC_PLAN_FORBIDDEN" in repair_prompt["repair"][
        "validation_errors"
    ]
    assert "solution/candidate.py" not in json.dumps(repair_prompt)
    assert "pairwise/listwise" in json.dumps(repair_prompt)
    assert "Remove repository paths" in repair_prompt["repair"]["instruction"]


def test_research_planner_persists_code_reference_diagnostic(planner_context):
    responses = [
        response(candidate(change_summary="Edit solution/candidate.py.")),
        response(candidate(change_summary="Edit src/tacorank/training.")),
    ]

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        return responses.pop(0)

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    planner = ResearchPlanner(
        provider,
        output_factory=output_factory,
        input_token_limit=2_000,
        output_token_limit=1_000,
    )

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "blocked"
    assert result["reason_code"] == "INVALID_PROVIDER_PLAN"
    assert "CODE_SPECIFIC_PLAN_FORBIDDEN" in result["reason"]
    assert "field=change_summary" in result["reason"]
    assert "category=source_path" in result["reason"]
    assert 'token="src/tacorank/training"' in result["reason"]


def test_deepseek_provider_rejects_truncated_completion(planner_context):
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return response(candidate(), finish_reason="length")

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    choice = SearchPolicy().choose(planner_context)

    with pytest.raises(ProviderError, match="did not finish cleanly"):
        asyncio.run(provider.generate(ProviderRequest(planner_context, choice)))

    assert len(calls) == 2
    assert calls[0]["thinking"] == {"type": "enabled"}
    assert calls[1]["thinking"] == {"type": "disabled"}
    assert "compactly" in calls[1]["messages"][0]["content"]


def test_deepseek_provider_retries_length_once_without_thinking(planner_context):
    responses = [
        response(candidate(), finish_reason="length", completion_tokens=1_000),
        response(candidate(), completion_tokens=100),
    ]
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return responses.pop(0)

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    choice = SearchPolicy().choose(planner_context)

    result = asyncio.run(provider.generate(ProviderRequest(planner_context, choice)))

    assert result["experiment_id"] == "exp_0001"
    assert len(calls) == 2
    assert calls[1]["thinking"] == {"type": "disabled"}
    assert provider.resource_delta.llm_output_tokens == 1_100


def test_cli_selects_deepseek_without_putting_secret_in_config(config, monkeypatch):
    configured = config.model_copy(update={"research_provider": "deepseek"})
    monkeypatch.setenv(configured.deepseek_api_key_env, "secret-key")

    planner = _planner_for(configured)

    assert isinstance(planner, ResearchPlanner)
    assert isinstance(planner.provider, DeepSeekResearchProvider)
    assert "secret-key" not in json.dumps(configured.canonical_dict(), sort_keys=True)


def test_cli_enables_keyless_openalex_skill(config, monkeypatch):
    configured = config.model_copy(
        update={"research_provider": "deepseek", "literature_research_enabled": True}
    )
    monkeypatch.setenv(configured.deepseek_api_key_env, "secret-key")
    planner = _planner_for(configured)

    assert isinstance(planner.literature_skill, OpenAlexLiteratureSkill)
    serialized_config = json.dumps(configured.canonical_dict(), sort_keys=True)
    assert "secret-key" not in serialized_config


def test_cli_fails_closed_when_deepseek_key_is_missing(config, monkeypatch):
    configured = config.model_copy(update={"research_provider": "deepseek"})
    monkeypatch.delenv(configured.deepseek_api_key_env, raising=False)

    with pytest.raises(ProviderError, match=configured.deepseek_api_key_env):
        _planner_for(configured)


def test_deepseek_preflight_authenticates_and_requires_configured_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @staticmethod
        def read(limit):
            assert limit == 1024 * 1024
            return json.dumps(
                {"object": "list", "data": [{"id": "deepseek-v4-flash"}]}
            ).encode("utf-8")

    requests = []

    def open_request(request, timeout, context):
        requests.append((request, timeout, context))
        return Response()

    monkeypatch.setattr("tacorank.providers.deepseek.urlopen", open_request)
    provider = DeepSeekResearchProvider(api_key="secret-key")

    provider.preflight()

    request, timeout, context = requests[0]
    assert request.full_url == "https://api.deepseek.com/models"
    assert request.headers["Authorization"] == "Bearer secret-key"
    assert timeout == 30
    assert context is deepseek_module._TLS_CONTEXT


def test_deepseek_preflight_redacts_http_failure_detail(monkeypatch):
    def reject(request, timeout, context):
        del request, timeout
        assert context is deepseek_module._TLS_CONTEXT
        raise HTTPError(
            "https://api.deepseek.com/models",
            401,
            "bad secret-key",
            {},
            io.BytesIO(b'{"error":"secret-key"}'),
        )

    monkeypatch.setattr("tacorank.providers.deepseek.urlopen", reject)
    provider = DeepSeekResearchProvider(api_key="secret-key")

    with pytest.raises(ProviderError, match="HTTP 401") as captured:
        provider.preflight()

    assert "secret-key" not in str(captured.value)
