"""Slash-separated research terms are prose, not repository paths.

run_20260831T005157Z rejected two consecutive plans with
CODE_SPECIFIC_PLAN_FORBIDDEN. The planner's own context contained no code
reference at all, so nothing was fed to it; the detector was matching ordinary
research phrasing on separator count alone.
"""

from __future__ import annotations

import pytest

from tacorank.research.code_blind import (
    contains_implementation_reference,
    redact_implementation_references,
)


@pytest.mark.parametrize(
    "text",
    [
        "the train/valid/test split is fixed",
        # smoke/proxy/full is this harness's own fidelity vocabulary.
        "evaluate at smoke/proxy/full fidelity",
        # The reviewed model_compact_ranker card is written with this phrase.
        "user/item/date interactions",
        "author/tab/duration features",
        "positive/negative pairs",
        "GAUC/nDCG@5 trade-off",
    ],
)
def test_research_prose_is_not_an_implementation_reference(text):
    assert not contains_implementation_reference(text)
    assert redact_implementation_references(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "patch solution/candidate.py",
        "solution/train.py",
        "rewrite candidate.py",
        "edit src/tacorank/foo",
        "modify tests/test_x.py",
        "lib/models/ranker",
        "see ./scripts/run.sh",
        r"C:\nus\techjam\solution",
        r"a\b\c\d",
        "the entrypoint must be preserved",
        "change the function name",
    ],
)
def test_real_references_are_still_detected(text):
    assert contains_implementation_reference(text)
    assert redact_implementation_references(text) != text
