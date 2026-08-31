from __future__ import annotations

import asyncio
import io
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from tacorank.research.literature import (
    LiteratureResearchError,
    OpenAlexLiteratureSkill,
)
from tacorank.research import literature as literature_module
from tacorank.research.search_policy import SearchPolicy


def _paper(
    paper_id: str,
    *,
    citations: int = 20,
    abstract: str = "Pairwise ranking optimizes relative preference ordering.",
    title: str = "A paper-backed recommender method",
):
    inverted = {
        token: [index]
        for index, token in enumerate(abstract.split())
    }
    return {
        "id": "https://openalex.org/" + paper_id,
        "doi": "https://doi.org/10.0000/" + paper_id.lower(),
        "title": title,
        "abstract_inverted_index": inverted,
        "publication_year": 2024,
        "authorships": [
            {"author": {"display_name": "A. Researcher"}},
            {"author": {"display_name": "B. Scientist"}},
        ],
        "primary_location": {"source": {"display_name": "RecSys"}},
        "cited_by_count": citations,
    }


def test_openalex_skill_queries_selected_method_and_bounds_evidence(
    planner_context,
):
    calls = []

    def transport(url, headers, timeout):
        calls.append((url, headers, timeout))
        return {
            "results": [
                _paper("W100", citations=2),
                _paper("W200", citations=40),
                _paper("W300", citations=30),
                _paper("W400", citations=25),
            ]
        }

    skill = OpenAlexLiteratureSkill(
        max_papers=2,
        min_citation_count=5,
        transport=transport,
    )
    choice = SearchPolicy().choose(planner_context)

    evidence = asyncio.run(skill.research(planner_context, choice))

    assert [item.paper_id for item in evidence] == ["W200", "W300"]
    assert all(item.query.startswith("within user pairwise") for item in evidence)
    assert all(item.provider == "openalex" for item in evidence)
    assert evidence[0].abstract == (
        "Pairwise ranking optimizes relative preference ordering."
    )
    assert all(item.evidence_id.startswith("lit_") for item in evidence)
    url, headers, timeout = calls[0]
    query = parse_qs(urlparse(url).query)
    assert urlparse(url).path == "/works"
    assert query["search"] == [
        "within user pairwise listwise learning to rank recommender systems BPR"
    ]
    assert query["per-page"] == ["20"]
    assert "abstract_inverted_index" in query["select"][0]
    assert "x-api-key" not in headers
    assert timeout == 20
    assert skill.resource_delta.wall_time_ms >= 0


def test_openalex_skill_fails_closed_without_usable_paper(planner_context):
    skill = OpenAlexLiteratureSkill(
        min_citation_count=5,
        transport=lambda url, headers, timeout: {
            "results": [_paper("W100", citations=0, abstract="")]
        },
    )

    with pytest.raises(
        LiteratureResearchError,
        match="no usable relevant cited papers",
    ):
        asyncio.run(
            skill.research(planner_context, SearchPolicy().choose(planner_context))
        )


def test_openalex_skill_rejects_highly_cited_irrelevant_paper(planner_context):
    skill = OpenAlexLiteratureSkill(
        max_papers=1,
        transport=lambda url, headers, timeout: {
            "results": [
                _paper(
                    "W100",
                    citations=10_000,
                    title="A survey of biodiversity metrics",
                    abstract="Ecological diversity is measured across forest habitats.",
                ),
                _paper("W200", citations=20),
            ]
        },
    )

    evidence = asyncio.run(
        skill.research(planner_context, SearchPolicy().choose(planner_context))
    )

    assert [item.paper_id for item in evidence] == ["W200"]


def test_openalex_preflight_requires_search_collection():
    calls = []

    def transport(url, headers, timeout):
        calls.append((url, headers, timeout))
        return {"results": []}

    skill = OpenAlexLiteratureSkill(transport=transport)

    skill.preflight()

    assert len(calls) == 1
    assert parse_qs(urlparse(calls[0][0]).query)["per-page"] == ["1"]


def test_openalex_preflight_rejects_invalid_envelope():
    skill = OpenAlexLiteratureSkill(
        transport=lambda url, headers, timeout: {"message": "not searchable"}
    )

    with pytest.raises(LiteratureResearchError, match="preflight"):
        skill.preflight()


def test_openalex_rate_limit_error_is_redacted(monkeypatch):
    def reject(request, timeout, context):
        del request, timeout, context
        raise HTTPError(
            "https://api.openalex.org/works",
            429,
            "rate limited",
            {},
            io.BytesIO(b'{"message":"untrusted detail"}'),
        )

    monkeypatch.setattr(literature_module, "urlopen", reject)
    skill = OpenAlexLiteratureSkill()

    with pytest.raises(LiteratureResearchError, match="service limit") as captured:
        skill.preflight()

    assert "untrusted detail" not in str(captured.value)
