import asyncio
import io
from urllib.error import HTTPError

from tacorank.research.literature import OpenAlexLiteratureSkill

from .test_literature import _paper


def test_openalex_retries_429_honors_bounded_retry_after(monkeypatch):
    calls = []

    def transport(url, headers, timeout):
        del headers, timeout
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError(url, 429, "limited", {"Retry-After": "30"}, io.BytesIO())
        return {"results": [_paper("W200", citations=20)]}

    async def no_sleep(delay):
        assert delay == 15.0

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    result = asyncio.run(
        OpenAlexLiteratureSkill(
            transport=transport,
            min_interval_seconds=0,
            max_attempts=3,
        ).search(type("Context", (), {"run_id": "run_one"})(), "ranking mechanism")
    )

    assert result.status == "available"
    assert len(calls) == 2


def test_openalex_does_not_retry_permanent_errors():
    calls = []

    def transport(url, headers, timeout):
        del headers, timeout
        calls.append(url)
        raise HTTPError(url, 403, "forbidden", {}, io.BytesIO())

    result = asyncio.run(
        OpenAlexLiteratureSkill(
            transport=transport,
            min_interval_seconds=0,
        ).search(type("Context", (), {"run_id": "run_two"})(), "ranking mechanism")
    )

    assert result.status == "unavailable"
    assert len(calls) == 1


def test_openalex_retries_timeout_and_server_errors(monkeypatch):
    calls = []
    delays = []

    def transport(url, headers, timeout):
        del headers, timeout
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("temporary timeout")
        if len(calls) == 2:
            raise HTTPError(url, 503, "busy", {}, io.BytesIO())
        return {"results": [_paper("W201", citations=20)]}

    async def no_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    result = asyncio.run(
        OpenAlexLiteratureSkill(
            transport=transport,
            min_interval_seconds=0,
            max_attempts=3,
        ).search(type("Context", (), {"run_id": "run_three"})(), "ranking mechanism")
    )

    assert result.status == "available"
    assert len(calls) == 3
    assert len(delays) == 2
    assert 1.0 <= delays[0] <= 1.25
    assert 2.0 <= delays[1] <= 2.25


def test_openalex_cache_is_scoped_to_run_and_coalesces_repeat_queries():
    calls = []

    def transport(url, headers, timeout):
        del headers, timeout
        calls.append(url)
        return {"results": [_paper("W200", citations=20)]}

    skill = OpenAlexLiteratureSkill(transport=transport, min_interval_seconds=0)

    async def run():
        first, second = await asyncio.gather(
            skill.search(type("Context", (), {"run_id": "run_one"})(), "ranking mechanism"),
            skill.search(type("Context", (), {"run_id": "run_one"})(), "ranking mechanism"),
        )
        third = await skill.search(type("Context", (), {"run_id": "run_two"})(), "ranking mechanism")
        return first, second, third

    first, second, third = asyncio.run(run())
    assert first.evidence == second.evidence
    assert third.evidence == first.evidence
    assert len(calls) == 2
