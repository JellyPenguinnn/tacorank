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
