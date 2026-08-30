from __future__ import annotations

import pytest

from tacorank.config import ContractError, RunConfig, verify_contract


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
