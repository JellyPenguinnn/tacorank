"""Bounded online scholarly research for the code-blind planner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import ssl
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
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


def _retry_after_seconds(headers: Any) -> float | None:
    """Parse Retry-After without trusting an unbounded server delay."""

    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return min(15.0, max(0.0, float(str(raw).strip())))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw))
            if target.tzinfo is None:
                return None
            return min(15.0, max(0.0, target.timestamp() - time.time()))
        except (TypeError, ValueError, OverflowError):
            return None


def _retry_delay(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(15.0, max(0.0, retry_after))
    return min(15.0, 2.0**attempt + random.uniform(0.0, 0.25))


METHOD_QUERIES = {
    "objective_pairwise_bpr": (
        "Bayesian personalized ranking pairwise learning to rank recommender "
        "systems implicit feedback"
    ),
    "objective_listwise_user_softmax": (
        "listwise learning to rank user impression softmax recommender systems nDCG"
    ),
    "objective_loss_aligned_features": (
        "feature engineering pairwise learning to rank recommender systems "
        "within user interaction features"
    ),
    "objective_weighted_cross_entropy": (
        "weighted cross entropy implicit feedback recommender ranking "
        "long tail users"
    ),
    "objective_distill_softmax": (
        "knowledge distillation softmax recommender systems ranking teacher "
        "student"
    ),
    "temporal_history_compact": (
        "sequential recommendation temporal user behavior history attention"
    ),
    "temporal_deep_interest_network": (
        "deep interest network DIN sequential recommendation user interest "
        "attention"
    ),
    "temporal_search_interest_model": (
        "search based interest model SIM long term user interest recommender"
    ),
    "temporal_time_series_interest": (
        "time series user interest evolution sequential recommendation"
    ),
    "multitask_single_auxiliary": (
        "multi task learning recommender systems auxiliary engagement ranking"
    ),
    "multitask_shared_bottom": (
        "shared bottom multi task learning recommender systems ranking"
    ),
    "multitask_gsu": (
        "gated shared unit GSU multi task recommender systems"
    ),
    "multitask_esu": (
        "expert sharing unit ESU multi task recommender systems"
    ),
    "multitask_mmoe": (
        "multi gate mixture of experts MMOE recommender systems ranking"
    ),
    "multitask_ple": (
        "progressive layered extraction PLE multi task recommender systems"
    ),
    "duration_bias_censored_watch_time": (
        "duration bias censored watch time video recommendation ranking"
    ),
    "temporal_drift_past_only": (
        "temporal distribution shift recommender systems recency ranking"
    ),
    "model_compact_ranker": (
        "DeepFM deep cross network recommender ranking feature interaction"
    ),
    "features_general_bounded_engineering": (
        "feature engineering train only recommender systems learning to rank"
    ),
    "model_field_aware_fm": (
        "field aware factorization machine FFM recommender systems ranking"
    ),
    "model_deep_cross_network": (
        "deep cross network DCN recommender systems feature interactions ranking"
    ),
    "model_lhuc": (
        "learning hidden unit contribution LHUC recommendation ranking"
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


class LiteratureResearchError(RuntimeError):
    """Raised when required online literature evidence cannot be established."""

    def __init__(
        self,
        message: str,
        *,
        status: str = "unavailable",
        retry_after: float | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.retryable = retryable


@dataclass(frozen=True)
class LiteratureSearchResult:
    evidence: tuple[LiteratureEvidence, ...]
    status: str
    error: str | None = None
    resource_delta: ResourceDelta = ResourceDelta()


class LiteratureResearchSkill(Protocol):
    @property
    def resource_delta(self) -> ResourceDelta:
        """Return resources consumed by the most recent research call."""

    def preflight(self, *, required: bool = True) -> None:
        """Verify that the configured scholarly source is reachable."""

    async def search(self, context: Any, query: str) -> LiteratureSearchResult:
        """Run one bounded, optional query and return a typed status."""

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
        retry_after = _retry_after_seconds(exc.headers)
        status = "rate_limited" if exc.code == 429 else "unavailable"
        raise LiteratureResearchError(
            "OpenAlex request failed with HTTP %d" % exc.code,
            status=status,
            retry_after=retry_after,
            retryable=exc.code in {408, 429, 500, 502, 503, 504},
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise LiteratureResearchError(
            "OpenAlex request could not connect",
            status="unavailable",
            retryable=True,
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
        total_timeout_seconds: int = 45,
        max_attempts: int = 3,
        min_interval_seconds: float = 1.0,
        transport: Optional[LiteratureTransport] = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("literature timeout must be positive")
        if not 1 <= max_papers <= 5:
            raise ValueError("literature max_papers must be between one and five")
        if min_citation_count < 0:
            raise ValueError("literature min_citation_count must be non-negative")
        if total_timeout_seconds <= 0 or max_attempts < 1 or max_attempts > 3:
            raise ValueError("literature retry limits are invalid")
        if min_interval_seconds < 0:
            raise ValueError("literature minimum interval must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_papers = max_papers
        self.min_citation_count = min_citation_count
        self.total_timeout_seconds = total_timeout_seconds
        self.max_attempts = max_attempts
        self.min_interval_seconds = min_interval_seconds
        self.transport = transport or _default_transport
        self._wall_time_ms = 0
        self._cache: dict[tuple[str, str], LiteratureSearchResult] = {}
        self._inflight: dict[tuple[str, str], asyncio.Task[LiteratureSearchResult]] = {}

    _limiter_lock = threading.Lock()
    _next_request_at = 0.0

    @property
    def resource_delta(self) -> ResourceDelta:
        return ResourceDelta(wall_time_ms=self._wall_time_ms)

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

    def preflight(self, *, required: bool = True) -> None:
        started = time.monotonic()
        error: LiteratureResearchError | None = None
        payload = None
        for attempt in range(self.max_attempts):
            remaining = self.total_timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            self._throttle_sync(self.min_interval_seconds)
            try:
                payload = self.transport(
                    self._search_url("recommender systems", limit=1),
                    self._headers(),
                    max(1, min(self.timeout_seconds, int(remaining))),
                )
                break
            except Exception as caught:
                status, retryable, retry_after = self._error_status(caught)
                error = LiteratureResearchError(
                    "OpenAlex request unavailable after bounded retry policy",
                    status=status,
                    retry_after=retry_after,
                )
                if not retryable or attempt + 1 >= self.max_attempts:
                    break
                delay = _retry_delay(attempt, retry_after)
                if time.monotonic() + delay - started >= self.total_timeout_seconds:
                    break
                time.sleep(min(15.0, max(0.0, delay)))
        if error is not None or not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
            error = error or LiteratureResearchError(
                "OpenAlex preflight returned no searchable paper collection",
                status="unavailable",
            )
            if not required:
                return
            if error.status == "rate_limited":
                raise LiteratureResearchError(
                    "OpenAlex rate limited the request (service limit)",
                    status=error.status,
                ) from error
            raise LiteratureResearchError(
                "OpenAlex preflight unavailable", status=error.status
            ) from error

    @staticmethod
    def _context_run_id(context: Any) -> str:
        return str(get_value(context, "run_id", "anonymous"))

    @staticmethod
    def _bounded_query(query: str) -> str:
        query = _clean_text(query, limit=240)
        if not query:
            raise LiteratureResearchError("literature query is empty", status="unavailable")
        lowered = query.lower()
        forbidden = ("run_", "evt_", "user_id", "label", "secret", "/", "\\")
        if any(token in lowered for token in forbidden):
            raise LiteratureResearchError("literature query is not code-blind", status="unavailable")
        return query

    @staticmethod
    def _error_status(error: BaseException) -> tuple[str, bool, float | None]:
        code = getattr(error, "code", None)
        retry_after = _retry_after_seconds(getattr(error, "headers", None))
        if code is None:
            code = getattr(error, "status_code", None)
        if code in {408, 429, 500, 502, 503, 504}:
            return ("rate_limited" if code == 429 else "unavailable", True, retry_after)
        if code in {400, 401, 403, 404}:
            return ("unavailable", False, retry_after)
        if isinstance(error, LiteratureResearchError):
            retryable = error.retryable
            if retryable is None:
                retryable = error.status == "rate_limited"
            return (error.status, retryable, error.retry_after)
        if isinstance(
            error,
            (ConnectionError, URLError, TimeoutError, asyncio.TimeoutError),
        ):
            return ("unavailable", True, retry_after)
        return ("unavailable", False, retry_after)

    @classmethod
    async def _throttle(cls, interval: float) -> None:
        if interval <= 0:
            return
        now = time.monotonic()
        with cls._limiter_lock:
            wait = max(0.0, cls._next_request_at - now)
            cls._next_request_at = max(now, cls._next_request_at) + interval
        if wait:
            await asyncio.sleep(wait)

    @classmethod
    def _throttle_sync(cls, interval: float) -> None:
        if interval <= 0:
            return
        now = time.monotonic()
        with cls._limiter_lock:
            wait = max(0.0, cls._next_request_at - now)
            cls._next_request_at = max(now, cls._next_request_at) + interval
        if wait:
            time.sleep(wait)

    async def _request_with_retry(self, query: str, *, limit: int) -> Mapping[str, Any]:
        started = time.monotonic()
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            remaining = self.total_timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            await self._throttle(self.min_interval_seconds)
            timeout = max(1, min(self.timeout_seconds, int(remaining)))
            try:
                return await asyncio.to_thread(
                    self.transport,
                    self._search_url(query, limit=limit),
                    self._headers(),
                    timeout,
                )
            except Exception as error:
                status, retryable, retry_after = self._error_status(error)
                last_error = error
                if not retryable or attempt + 1 >= self.max_attempts:
                    raise LiteratureResearchError(
                        "OpenAlex request unavailable after bounded retry policy",
                        status=status,
                        retry_after=retry_after,
                    ) from error
                delay = _retry_delay(attempt, retry_after)
                if time.monotonic() + delay - started >= self.total_timeout_seconds:
                    break
                await asyncio.sleep(delay)
        raise LiteratureResearchError(
            "OpenAlex request exceeded its bounded deadline",
            status="unavailable",
        ) from last_error

    async def _search_uncached(self, context: Any, query: str) -> LiteratureSearchResult:
        started = time.monotonic()
        result: LiteratureSearchResult
        try:
            payload = await self._request_with_retry(query, limit=max(10, self.max_papers * 3))
            if not isinstance(payload.get("results"), list):
                result = LiteratureSearchResult((), "unavailable", "invalid paper collection")
            else:
                try:
                    evidence = tuple(self._parse(payload, query))
                except LiteratureResearchError:
                    result = LiteratureSearchResult((), "empty", "no usable cited papers")
                except Exception:
                    result = LiteratureSearchResult(
                        (), "unavailable", "OpenAlex returned unusable paper records"
                    )
                else:
                    result = LiteratureSearchResult(evidence, "available")
        except LiteratureResearchError as error:
            result = LiteratureSearchResult((), error.status, str(error))
        finally:
            self._wall_time_ms = max(0, int(round((time.monotonic() - started) * 1_000)))
        elapsed_ms = self._wall_time_ms
        return LiteratureSearchResult(
            result.evidence,
            result.status,
            result.error,
            ResourceDelta(wall_time_ms=elapsed_ms),
        )

    async def search(self, context: Any, query: str) -> LiteratureSearchResult:
        query = self._bounded_query(query)
        key = (self._context_run_id(context), query)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._search_uncached(context, query))
            self._inflight[key] = task
        try:
            result = await asyncio.shield(task)
            self._cache[key] = result
            return result
        finally:
            if self._inflight.get(key) is task:
                self._inflight.pop(key, None)

    async def research(
        self, context: Any, policy_choice: Any
    ) -> Sequence[LiteratureEvidence]:
        query = self._query(policy_choice)
        result = await self.search(context, query)
        if result.status != "available":
            raise LiteratureResearchError(
                "OpenAlex literature is %s%s"
                % (
                    result.status,
                    ": no usable cited papers" if result.status == "empty" else "",
                ),
                status=result.status,
            )
        return result.evidence

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

    def _parse(self, payload: Mapping[str, Any], query: str) -> list[LiteratureEvidence]:
        records = payload.get("results")
        if not isinstance(records, list):
            raise LiteratureResearchError(
                "OpenAlex search returned an invalid paper collection"
            )
        evidence: list[LiteratureEvidence] = []
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
            evidence.append(
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
                )
            )
            seen.add(paper_id)
            if len(evidence) >= self.max_papers:
                break
        if not evidence:
            raise LiteratureResearchError(
                "OpenAlex returned no usable cited papers for the selected method"
            )
        return evidence



__all__ = [
    "LiteratureResearchError",
    "LiteratureSearchResult",
    "LiteratureResearchSkill",
    "METHOD_QUERIES",
    "OpenAlexLiteratureSkill",
]
