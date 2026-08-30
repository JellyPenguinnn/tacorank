from tacorank.research.code_references import (
    find_code_reference,
    redact_code_references,
)


def test_scientific_slash_notation_is_not_a_code_reference():
    text = (
        "Compare pairwise/listwise objectives, positive/negative sampling, "
        "and the GAUC/nDCG tradeoff."
    )

    assert find_code_reference(text) is None
    assert redact_code_references(text) == text


def test_markup_and_urls_are_not_source_paths():
    text = "</evidence> See https://example.com/research for context."

    assert find_code_reference(text) is None
    assert redact_code_references(text) == text


def test_explicit_paths_and_source_files_are_code_references():
    samples = {
        "Edit solution/candidate.py.": "solution/candidate.py",
        "Inspect src/tacorank/training.": "src/tacorank/training",
        "Read ./candidate before proposing.": "./candidate",
        "Use /tmp/candidate/output.": "/tmp/candidate/output",
        "Change candidate.py directly.": "candidate.py",
    }

    for text, expected in samples.items():
        reference = find_code_reference(text)
        assert reference is not None
        assert reference.text == expected
        assert expected not in redact_code_references(text)
