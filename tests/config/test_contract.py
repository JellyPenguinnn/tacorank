from __future__ import annotations

import json
from pathlib import Path

import pytest

from tacorank.config import ContractError, RunConfig, verify_contract
from tacorank.schemas import ResearchCampaign


def test_objective_temporal_feature_campaign_has_exact_fifty_slot_budget():
    path = Path(__file__).parents[2] / "research/campaigns/objective_temporal_50.json"
    campaign = ResearchCampaign.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert campaign.campaign_id == "objective_temporal_features_50_v3"
    assert campaign.family_order == ["objective", "temporal_history", "features"]
    assert campaign.family_budgets == {
        "objective": 27,
        "temporal_history": 15,
        "features": 8,
    }
    assert campaign.experiment_budget == 50
    assert campaign.minimum_family_full_evaluations == 5
    assert campaign.family_convergence_patience == 5
    assert campaign.family_method_card_ids["objective"] == [
        "objective_pairwise_bpr",
        "objective_listwise_user_softmax",
    ]
    assert campaign.family_method_card_ids["features"] == [
        "features_history_affinity"
    ]


def test_run_config_requires_budget_but_campaign_owns_its_convergence(config):
    payload = config.model_dump(mode="python")
    payload.update(
        max_experiments=4,
        convergence_patience=4,
        research_campaign={
            "campaign_id": "test_campaign",
            "family_order": ["objective", "temporal_history"],
            "family_budgets": {"objective": 2, "temporal_history": 2},
            "family_method_card_ids": {
                "objective": ["objective_pairwise_bpr"],
                "temporal_history": ["temporal_history_compact"],
            },
            "family_directives": {
                "objective": "Adapt objective trials.",
                "temporal_history": "Adapt temporal trials.",
            },
        },
    )

    parsed = RunConfig.model_validate(payload)
    assert parsed.research_campaign is not None
    assert parsed.research_campaign.experiment_budget == 4

    payload["convergence_patience"] = 3
    parsed = RunConfig.model_validate(payload)
    assert parsed.convergence_patience == 3


def test_blank_contract_is_a_hard_stop(config, repository):
    (repository / config.contract_path).write_text("")
    with pytest.raises(ContractError, match="empty"):
        verify_contract(config)


def test_unresolved_contract_is_a_hard_stop(config, repository):
    (repository / config.contract_path).write_text(
        "Contract status: FROZEN\nMetrics: gauc ndcg@5 primary\nTODO resolve labels\n"
    )
    with pytest.raises(ContractError, match="unresolved"):
        verify_contract(config)


def test_negated_frozen_status_is_rejected(config, repository):
    path = repository / config.contract_path
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "Contract status: FROZEN", "NOT Contract status: FROZEN"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="exact line"):
        verify_contract(config)


def test_commands_and_artifact_roots_must_match_frozen_contract(config):
    config.command_ids = ["run_smoke", "unreviewed_command"]
    with pytest.raises(ContractError, match="command_ids"):
        verify_contract(config)

    config.command_ids = ["run_smoke", "run_proxy", "run_full"]
    config.artifact_roots = ["artifacts"]
    with pytest.raises(ContractError, match="artifact_roots"):
        verify_contract(config)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("deepseek_base_url", "http://api.deepseek.com", "HTTPS origin"),
        ("deepseek_base_url", "https://user:secret@api.deepseek.com", "credential-free"),
        ("deepseek_api_key_env", "deepseek-key", "uppercase environment variable"),
    ],
)
def test_deepseek_configuration_rejects_unsafe_values(config, field, value, match):
    payload = config.model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        RunConfig.model_validate(payload)


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "literature_base_url",
            "http://api.openalex.org",
            "OpenAlex HTTPS origin",
        ),
        (
            "literature_base_url",
            "https://example.com",
            "OpenAlex HTTPS origin",
        ),
        (
            "literature_base_url",
            "https://user:secret@api.openalex.org",
            "credential-free",
        ),
    ],
)
def test_literature_configuration_rejects_unsafe_values(
    config, field, value, match
):
    payload = config.model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        RunConfig.model_validate(payload)


def test_coding_token_limit_can_be_explicitly_unbounded(config):
    payload = config.model_dump(mode="python")
    payload["coding_token_limit"] = None

    parsed = RunConfig.model_validate(payload)

    assert parsed.coding_token_limit is None


def test_coding_token_limit_is_unbounded_by_default(config):
    payload = config.model_dump(mode="python")
    payload.pop("coding_token_limit")

    parsed = RunConfig.model_validate(payload)

    assert parsed.coding_token_limit is None


@pytest.mark.parametrize(
    "seeds,match",
    [
        ([11, 11, 33], "distinct seeds"),
        ([11, 22], "confirmation seed"),
    ],
)
def test_seed_schedule_covers_distinct_confirmation_evidence(config, seeds, match):
    payload = config.model_dump(mode="python")
    payload["seed_schedule"] = seeds

    with pytest.raises(ValueError, match=match):
        RunConfig.model_validate(payload)
