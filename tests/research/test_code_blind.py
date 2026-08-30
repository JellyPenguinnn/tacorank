from tacorank.research.code_blind import (
    contains_implementation_reference,
    find_implementation_reference,
    redact_implementation_references,
)


def test_scientific_slash_notation_is_not_a_code_reference() -> None:
    for value in (
        "positive/negative pairs",
        "user/item interactions",
        "train/validation tradeoff",
        "accuracy/diversity",
        "and/or",
        "pairwise/listwise objectives",
        "GAUC/nDCG tradeoff",
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


def test_markup_and_urls_are_not_source_paths() -> None:
    value = "</evidence> See https://example.com/research for context."

    assert find_implementation_reference(value) is None
    assert redact_implementation_references(value) == value


def test_finder_reports_the_earliest_reference_and_category() -> None:
    value = "Edit solution/candidate.py after reviewing candidate.py."

    reference = find_implementation_reference(value)

    assert reference is not None
    assert reference.category == "source_path"
    assert reference.text == "solution/candidate.py"
