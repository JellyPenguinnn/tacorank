from tacorank.research.code_blind import (
    contains_implementation_reference,
    redact_implementation_references,
)


def test_scientific_slash_notation_is_not_a_code_reference() -> None:
    for value in (
        "positive/negative pairs",
        "user/item interactions",
        "train/validation tradeoff",
        "accuracy/diversity",
        "and/or",
    ):
        assert not contains_implementation_reference(value), value
        assert redact_implementation_references(value) == value


def test_concrete_implementation_references_remain_forbidden() -> None:
    for value in (
        "solution/candidate.py",
        "src/tacorank/train",
        "/tmp/candidate",
        "../solution/candidate",
        "candidate.py",
        "the entrypoint",
        "a function name",
        "a class name",
        "a line number",
    ):
        assert contains_implementation_reference(value), value
        assert value not in redact_implementation_references(value)
