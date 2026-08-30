from pathlib import Path

import pytest
from pydantic import ValidationError

from tacorank.research.playbook import (
    REQUIRED_RULE_ORDER,
    SUPPORTED_RULES,
    PlaybookError,
    load_improvement_playbook,
)
from tacorank.schemas import PlannerPlaybookSummary


def control_block(*, rules=None, objective_methods=None):
    import json

    return "```json\n%s\n```\n" % json.dumps(
        {
            "schema_version": "1.0",
            "rule_order": list(rules or REQUIRED_RULE_ORDER),
            "family_order": ["objective"],
            "method_order": {
                "objective": objective_methods or ["objective_pairwise_bpr"]
            },
        }
    )


def test_markdown_playbook_control_block_is_executable():
    root = Path(__file__).parents[2]
    playbook = load_improvement_playbook(
        root / "research/CURRENT_RUN_IMPROVEMENT_PLAN.md",
        source_path="research/CURRENT_RUN_IMPROVEMENT_PLAN.md",
    )

    assert playbook.rule_order[0] == "output_rejected"
    assert playbook.family_order[0] == "objective"
    assert playbook.methods_for("objective")[0] == "objective_pairwise_bpr"
    assert len(playbook.source_sha256) == 64


def test_markdown_playbook_rejects_unknown_rule(tmp_path):
    path = tmp_path / "playbook.md"
    path.write_text(
        """```json
{"schema_version":"1.0","rule_order":["let_the_llm_override_safety"],"family_order":["objective"],"method_order":{"objective":["objective_pairwise_bpr"]}}
```
""",
        encoding="utf-8",
    )

    with pytest.raises(PlaybookError, match="unsupported playbook rules"):
        load_improvement_playbook(path)


def test_markdown_playbook_rejects_missing_mandatory_rule(tmp_path):
    path = tmp_path / "playbook.md"
    path.write_text(
        control_block(rules=sorted(SUPPORTED_RULES - {"output_rejected"})),
        encoding="utf-8",
    )

    with pytest.raises(PlaybookError, match="missing mandatory playbook rules"):
        load_improvement_playbook(path)


def test_markdown_playbook_forces_pairwise_bpr_first(tmp_path):
    path = tmp_path / "playbook.md"
    path.write_text(
        control_block(
            objective_methods=[
                "objective_listwise_user_softmax",
                "objective_pairwise_bpr",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlaybookError, match="must be the first objective method"):
        load_improvement_playbook(path)


def test_markdown_playbook_rejects_unsafe_rule_reordering(tmp_path):
    path = tmp_path / "playbook.md"
    reordered = list(REQUIRED_RULE_ORDER)
    reordered[0], reordered[-1] = reordered[-1], reordered[0]
    path.write_text(control_block(rules=reordered), encoding="utf-8")

    with pytest.raises(PlaybookError, match="mandatory safety order"):
        load_improvement_playbook(path)


def test_shared_playbook_schema_rejects_unsafe_rule_reordering():
    reordered = list(REQUIRED_RULE_ORDER)
    reordered[0], reordered[-1] = reordered[-1], reordered[0]

    with pytest.raises(ValidationError, match="mandatory planner rule order"):
        PlannerPlaybookSummary(
            source_path="research/CURRENT_RUN_IMPROVEMENT_PLAN.md",
            source_sha256="a" * 64,
            rule_order=reordered,
            family_order=["objective"],
            method_order={"objective": ["objective_pairwise_bpr"]},
        )
