from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tacorank.research.literature import LiteratureResearchError
from tacorank.research.paper_bank import (
    EXPECTED_ORGANIZATION_COUNTS,
    METHOD_BANK_TOPICS,
    PaperBankLiteratureSkill,
)


BANK_PATH = Path(__file__).parents[2] / "research" / "paper_bank.json"


def _skill(*, max_papers: int = 6) -> PaperBankLiteratureSkill:
    return PaperBankLiteratureSkill(
        bank_path=BANK_PATH,
        expected_sha256=hashlib.sha256(BANK_PATH.read_bytes()).hexdigest(),
        max_papers=max_papers,
    )


def _research(skill: PaperBankLiteratureSkill, method_id: str):
    return asyncio.run(
        skill.research(None, SimpleNamespace(method_card_id=method_id))
    )


def test_bank_has_exact_reviewed_size_balance_and_advisory_semantics():
    skill = _skill()

    skill.preflight()

    assert skill.paper_count == 70
    assert skill.organization_counts == EXPECTED_ORGANIZATION_COUNTS
    assert skill.requires_citation is False


def test_bank_retrieval_changes_with_method_and_preserves_provenance():
    skill = _skill()

    duration = _research(skill, "duration_bias_censored_watch_time")
    temporal = _research(skill, "temporal_history_compact")

    assert len(duration) == len(temporal) == 6
    assert {item.evidence_id for item in duration} != {
        item.evidence_id for item in temporal
    }
    assert any("watch_time" in item.topics for item in duration)
    assert any("sequential" in item.topics for item in temporal)
    assert all(item.provider == "paper_bank" for item in duration + temporal)
    assert all(item.organization and item.relationship for item in duration + temporal)
    assert all(item.citation_count == 0 for item in duration + temporal)


def test_every_current_method_gets_a_distinct_reference_slice():
    skill = _skill()

    result_sets = {
        tuple(item.evidence_id for item in _research(skill, method_id))
        for method_id in METHOD_BANK_TOPICS
    }

    assert len(result_sets) == len(METHOD_BANK_TOPICS)


def test_bank_rejects_hash_mismatch():
    with pytest.raises(LiteratureResearchError, match="hash does not match"):
        PaperBankLiteratureSkill(
            bank_path=BANK_PATH,
            expected_sha256="0" * 64,
        )
