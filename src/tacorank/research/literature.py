"""Bounded online scholarly research for the code-blind planner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import ssl
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from ..schemas import LiteratureEvidence, ResourceDelta
from .graph_view import get_value


_TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_ABSTRACT_CHARACTERS = 2_000
_MAX_ABSTRACT_TOKENS = 2_048
_MAX_AUTHORS = 8
_SEARCH_FIELDS = (
    "id,doi,title,publication_year,authorships,primary_location,"
    "cited_by_count,abstract_inverted_index"
)

LiteratureTransport = Callable[
    [str, Mapping[str, str], int], Mapping[str, Any]
]


METHOD_QUERIES = {
    "objective_direct_within_user_ranker": (
        "within user pairwise listwise learning to rank recommender systems BPR"
    ),
    "objective_pairwise_bpr": (
        "Bayesian Personalized Ranking BPR pairwise recommender implicit feedback"
    ),
    "objective_listwise_user_softmax": (
        "listwise learning to rank ListNet softmax recommender systems nDCG"
    ),
    "objective_loss_aligned_features": (
        "pairwise learning to rank recommender feature interactions user item context"
    ),
    "temporal_history_compact": (
        "sequential recommendation temporal user behavior history attention"
    ),
    "multitask_single_auxiliary": (
        "multi task learning recommender systems auxiliary engagement ranking"
    ),
    "duration_bias_censored_watch_time": (
        "Counteracting Duration Bias Video Recommendation Counterfactual Watch Time"
    ),
    "features_author_affinity_past_only": (
        "author creator affinity context aware recommender temporal user preference"
    ),
    "features_tab_context_residual": (
        "context aware recommender feature interaction residual ranking"
    ),
    "temporal_drift_past_only": (
        "temporal distribution shift recommender systems recency ranking"
    ),
    "model_compact_ranker": (
        "DeepFM deep cross network recommender ranking feature interaction"
    ),
    "sampling_deterministic_coverage": (
        "negative sampling exposure bias recommender systems implicit feedback"
    ),
    "ensemble_diverse_residual_candidate": (
        "rank ensemble diverse recommender systems score fusion"
    ),
    "ensemble_confirmed_members": (
        "rank averaging ensemble recommender systems complementary models"
    ),
    "evaluation_random_exposure_robustness": (
        "unbiased evaluation recommender systems random exposure data"
    ),
}

# Every tuple is an AND group; a paper must contain at least one phrase from
# each group in its title or abstract. These static gates keep a highly cited
# but topically unrelated paper from grounding an implementation proposal.
METHOD_RELEVANCE_GROUPS = {
    "objective_direct_within_user_ranker": (
        ("recommend", "ranking"),
        (
            "bayesian personalized ranking",
            "bpr",
            "pairwise",
            "listwise",
            "learning to rank",
        ),
    ),
    "objective_pairwise_bpr": (
        ("recommend", "ranking"),
        (
            "bayesian personalized ranking",
            "bpr",
            "pairwise",
            "preference ordering",
        ),
    ),
    "objective_listwise_user_softmax": (
        ("recommend", "ranking"),
        ("listwise", "listnet", "listmle", "softmax"),
    ),
    "objective_loss_aligned_features": (
        ("recommend", "ranking"),
        ("pairwise", "listwise", "learning to rank", "preference"),
        ("feature", "representation", "interaction", "embedding"),
    ),
    "temporal_history_compact": (
        ("recommend",),
        ("temporal", "sequential", "history", "behavior sequence"),
    ),
    "multitask_single_auxiliary": (
        ("recommend", "ranking"),
        ("multi task", "multitask", "auxiliary task"),
    ),
    "duration_bias_censored_watch_time": (
        ("recommend", "video ranking"),
        ("watch time", "watch-time", "duration bias", "view duration"),
    ),
    "features_author_affinity_past_only": (
        ("recommend",),
        ("author", "creator", "affinity", "context aware"),
    ),
    "features_tab_context_residual": (
        ("recommend", "ranking"),
        ("context aware", "contextual feature", "feature interaction"),
    ),
    "temporal_drift_past_only": (
        ("recommend",),
        ("temporal", "distribution shift", "drift", "recency"),
    ),
    "model_compact_ranker": (
        ("recommend", "click through rate", "ctr"),
        (
            "deepfm",
            "factorization machine",
            "cross network",
            "feature interaction",
        ),
    ),
    "sampling_deterministic_coverage": (
        ("recommend",),
        ("negative sampling", "exposure sampling", "implicit feedback sampling"),
    ),
    "ensemble_diverse_residual_candidate": (
        ("recommend", "ranking"),
        ("ensemble", "rank fusion", "model combination"),
    ),
    "ensemble_confirmed_members": (
        ("recommend", "ranking"),
        ("ensemble", "rank averaging", "model combination"),
    ),
    "evaluation_random_exposure_robustness": (
        ("recommend",),
        ("unbiased evaluation", "random exposure", "off policy evaluation"),
    ),
}


class LiteratureResearchError(RuntimeError):
    """Raised when required online literature evidence cannot be established."""


class LiteratureResearchSkill(Protocol):
    @property
    def requires_citation(self) -> bool:
        """Return whether a proposal must cite at least one returned paper."""

    @property
    def resource_delta(self) -> ResourceDelta:
        """Return resources consumed by the most recent research call."""

    def preflight(self) -> None:
        """Verify that the configured scholarly source is reachable."""

    async def research(
        self, context: Any, policy_choice: Any
    ) -> Sequence[LiteratureEvidence]:
        """Retrieve bounded evidence for one controller-selected method."""


def _clean_text(value: Any, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip()


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _abstract_from_inverted_index(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    by_position: dict[int, str] = {}
    for raw_token, raw_positions in value.items():
        token = _clean_text(raw_token, limit=100)
        if not token or not isinstance(raw_positions, list):
            continue
        for position in raw_positions:
            if (
                isinstance(position, int)
                and not isinstance(position, bool)
                and 0 <= position < _MAX_ABSTRACT_TOKENS
            ):
                by_position.setdefault(position, token)
    return _clean_text(
        " ".join(by_position[position] for position in sorted(by_position)),
        limit=_MAX_ABSTRACT_CHARACTERS,
    )


def _relevance_score(method_id: str, title: str, abstract: str) -> int:
    groups = METHOD_RELEVANCE_GROUPS.get(method_id, ())
    if not groups:
        return 1
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        ("%s %s" % (title, abstract)).lower(),
    )
    matched = []
    for group in groups:
        count = sum(
            1
            for phrase in group
            if re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip() in text
        )
        if count == 0:
            return 0
        matched.append(count)
    return sum(matched)


def _default_transport(
    url: str, headers: Mapping[str, str], timeout_seconds: int
) -> Mapping[str, Any]:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
            context=_TLS_CONTEXT,
        ) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 429:
            raise LiteratureResearchError(
                "OpenAlex rate limited the request (HTTP 429); retry after the "
                "service limit resets"
            ) from exc
        raise LiteratureResearchError(
            "OpenAlex request failed with HTTP %d" % exc.code
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise LiteratureResearchError(
            "OpenAlex request could not connect"
        ) from exc
    if len(body) > _MAX_RESPONSE_BYTES:
        raise LiteratureResearchError("OpenAlex response exceeded the size limit")
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiteratureResearchError(
            "OpenAlex returned an invalid JSON response"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise LiteratureResearchError(
            "OpenAlex returned an invalid response envelope"
        )
    return decoded


class OpenAlexLiteratureSkill:
    """Retrieve a small immutable evidence set from keyless OpenAlex search."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openalex.org",
        timeout_seconds: int = 20,
        max_papers: int = 3,
        min_citation_count: int = 5,
        transport: Optional[LiteratureTransport] = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("literature timeout must be positive")
        if not 1 <= max_papers <= 8:
            raise ValueError("literature max_papers must be between one and eight")
        if min_citation_count < 0:
            raise ValueError("literature min_citation_count must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_papers = max_papers
        self.min_citation_count = min_citation_count
        self.transport = transport or _default_transport
        self._wall_time_ms = 0

    @property
    def resource_delta(self) -> ResourceDelta:
        return ResourceDelta(wall_time_ms=self._wall_time_ms)

    @property
    def requires_citation(self) -> bool:
        return True

    def _headers(self) -> dict[str, str]:
        return {"Accept": "application/json", "User-Agent": "TacoRank/1.0"}

    def _search_url(self, query: str, *, limit: int) -> str:
        parameters = urlencode(
            {
                "search": query,
                "per-page": limit,
                "select": _SEARCH_FIELDS,
            }
        )
        return self.base_url + "/works?" + parameters

    def preflight(self) -> None:
        payload = self.transport(
            self._search_url("recommender systems", limit=1),
            self._headers(),
            min(self.timeout_seconds, 30),
        )
        if not isinstance(payload.get("results"), list):
            raise LiteratureResearchError(
                "OpenAlex preflight returned no searchable paper collection"
            )

    @staticmethod
    def _query(policy_choice: Any) -> str:
        method_id = str(get_value(policy_choice, "method_card_id", ""))
        query = METHOD_QUERIES.get(method_id)
        if query:
            return query
        family = _clean_text(get_value(policy_choice, "family", "recommender"), limit=40)
        method = re.sub(r"[^A-Za-z0-9 ]", " ", method_id)
        return _clean_text(
            "%s %s recommender systems ranking" % (family, method), limit=200
        )

    def _parse(
        self,
        payload: Mapping[str, Any],
        query: str,
        method_id: str,
    ) -> list[LiteratureEvidence]:
        records = payload.get("results")
        if not isinstance(records, list):
            raise LiteratureResearchError(
                "OpenAlex search returned an invalid paper collection"
            )
        ranked_evidence: list[tuple[int, LiteratureEvidence]] = []
        seen: set[str] = set()
        for raw in records:
            if not isinstance(raw, Mapping):
                continue
            openalex_id = _clean_text(raw.get("id"), limit=200)
            paper_id = openalex_id.rsplit("/", 1)[-1]
            title = _clean_text(raw.get("title"), limit=500)
            abstract = _abstract_from_inverted_index(
                raw.get("abstract_inverted_index")
            )
            citation_count = _nonnegative_int(raw.get("cited_by_count"))
            if (
                not paper_id
                or not re.fullmatch(r"W[0-9]+", paper_id)
                or paper_id in seen
                or not title
                or not abstract
                or citation_count < self.min_citation_count
            ):
                continue
            relevance = _relevance_score(method_id, title, abstract)
            if relevance <= 0:
                continue
            raw_authors = raw.get("authorships")
            authors = []
            for author in raw_authors if isinstance(raw_authors, list) else []:
                if not isinstance(author, Mapping):
                    continue
                name = _clean_text(
                    get_value(author.get("author"), "display_name", ""),
                    limit=120,
                )
                if name and name not in authors:
                    authors.append(name)
                if len(authors) >= _MAX_AUTHORS:
                    break
            url = "https://openalex.org/" + paper_id
            raw_year = raw.get("publication_year")
            try:
                year = int(raw_year) if raw_year is not None else None
            except (TypeError, ValueError):
                year = None
            if year is not None and not 1800 <= year <= 2100:
                year = None
            primary_location = raw.get("primary_location")
            source = (
                primary_location.get("source")
                if isinstance(primary_location, Mapping)
                else None
            )
            venue = _clean_text(
                get_value(source, "display_name", ""), limit=200
            ) or None
            digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:16]
            ranked_evidence.append(
                (
                    relevance,
                    LiteratureEvidence(
                        evidence_id="lit_" + digest,
                        provider="openalex",
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        year=year,
                        authors=authors,
                        venue=venue,
                        citation_count=citation_count,
                        influential_citation_count=0,
                        url=url,
                        query=query,
                    ),
                )
            )
            seen.add(paper_id)
        ranked_evidence.sort(
            key=lambda item: (
                -item[0],
                -item[1].citation_count,
                item[1].paper_id,
            )
        )
        evidence = [item[1] for item in ranked_evidence[: self.max_papers]]
        if not evidence:
            raise LiteratureResearchError(
                "OpenAlex returned no usable relevant cited papers for the selected method"
            )
        return evidence

    async def research(
        self, context: Any, policy_choice: Any
    ) -> Sequence[LiteratureEvidence]:
        # The query is derived only from frozen method metadata. No dataset,
        # metric, run, or user identifiers leave the controller.
        del context
        method_id = str(get_value(policy_choice, "method_card_id", ""))
        query = self._query(policy_choice)
        started = time.monotonic()
        self._wall_time_ms = 0
        try:
            payload = await asyncio.to_thread(
                self.transport,
                self._search_url(query, limit=max(20, self.max_papers * 8)),
                self._headers(),
                self.timeout_seconds,
            )
            return self._parse(payload, query, method_id)
        finally:
            self._wall_time_ms = max(
                0, int(round((time.monotonic() - started) * 1_000))
            )


__all__ = [
    "LiteratureResearchError",
    "LiteratureResearchSkill",
    "METHOD_QUERIES",
    "METHOD_RELEVANCE_GROUPS",
    "OpenAlexLiteratureSkill",
]
