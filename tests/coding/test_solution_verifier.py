from __future__ import annotations

import json
from typing import Any, Mapping, Union

import pytest
import tacorank.coding.solution_verifier as verifier_module

from tacorank.coding.solution_verifier import (
    DeepSeekSolutionVerifier,
    SolutionVerifierError,
)


def _response(
    document: Union[Mapping[str, Any], str],
    *,
    input_tokens: int = 5,
    output_tokens: int = 3,
):
    content = document if isinstance(document, str) else json.dumps(document)
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content},
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        },
    }


def _request(provider: DeepSeekSolutionVerifier):
    return provider.verify(
        experiment_spec={
            "hypothesis": "add a bounded pairwise residual",
            "target_files": ["solution/candidate.py"],
        },
        target_interface_excerpts={"solution/candidate.py": "def run(invocation)"},
        selected_method_cards=[{"method_id": "pairwise_bpr"}],
        active_lessons=[{"summary": "never leak secret"}],
        diff_sha256="a" * 64,
        diff_text="diff --git a/solution/candidate.py b/solution/candidate.py\n",
        source_by_path={"solution/candidate.py": "def run(invocation):\n    return None\n"},
        timeout_seconds=30,
    )


def test_deepseek_solution_verifier_accepts_strict_grounded_json() -> None:
    captured = []

    def transport(url, headers, payload, timeout):
        captured.append((url, headers, payload, timeout))
        return _response(
            {
                "accepted": True,
                "summary": "The bounded residual and required entrypoint are present.",
                "findings": [
                    {
                        "code": "MINOR_UNCERTAINTY",
                        "severity": "warning",
                        "path": "solution/candidate.py",
                        "message": "Runtime behavior still requires CPU smoke.",
                    }
                ],
                "required_changes": [],
            }
        )

    result = _request(
        DeepSeekSolutionVerifier(
            api_key="secret",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            transport=transport,
        )
    )

    assert result.accepted
    assert result.input_tokens == 5
    assert result.output_tokens == 3
    assert result.provider_calls == 1
    assert captured[0][0] == "https://api.deepseek.com/chat/completions"
    assert captured[0][2]["response_format"] == {"type": "json_object"}
    assert "untrusted evidence" in captured[0][2]["messages"][0]["content"]
    assert "never leak secret" not in captured[0][2]["messages"][1]["content"]
    assert "[REDACTED]" in captured[0][2]["messages"][1]["content"]


def test_verifier_retries_malformed_json_once_and_accounts_both_calls() -> None:
    responses = [
        _response("{malformed", input_tokens=7, output_tokens=2),
        _response(
            {
                "accepted": False,
                "summary": "The planned residual is missing.",
                "findings": [
                    {
                        "code": "MECHANISM_MISSING",
                        "severity": "error",
                        "path": "solution/candidate.py",
                        "message": "run only copies the parent score.",
                    }
                ],
                "required_changes": ["Implement the approved bounded residual."],
            },
            input_tokens=8,
            output_tokens=4,
        ),
    ]
    payloads = []

    def transport(_url, _headers, payload, _timeout):
        payloads.append(payload)
        return responses.pop(0)

    result = _request(
        DeepSeekSolutionVerifier(
            api_key="secret",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            transport=transport,
        )
    )

    assert not result.accepted
    assert result.provider_calls == 2
    assert (result.input_tokens, result.output_tokens) == (15, 6)
    assert payloads[1]["thinking"] == {"type": "disabled"}
    retry_prompt = payloads[1]["messages"][0]["content"]
    assert "COMPLETION_JSON_MALFORMED" in retry_prompt
    assert "Correct only the response protocol" in retry_prompt
    assert "code, severity, path, and message" in retry_prompt


def test_verifier_retry_identifies_finding_key_mismatch_and_proposed_fix() -> None:
    responses = [
        _response(
            {
                "accepted": False,
                "summary": "The planned residual is missing.",
                "findings": [
                    {
                        "code": "MECHANISM_MISSING",
                        "severity": "error",
                        "path": "solution/candidate.py",
                        "message": "run only copies the parent score.",
                        "line": 12,
                    }
                ],
                "required_changes": ["Implement the approved bounded residual."],
            }
        ),
        _response(
            {
                "accepted": False,
                "summary": "The planned residual is missing.",
                "findings": [
                    {
                        "code": "MECHANISM_MISSING",
                        "severity": "error",
                        "path": "solution/candidate.py",
                        "message": "run only copies the parent score.",
                    }
                ],
                "required_changes": ["Implement the approved bounded residual."],
            }
        ),
    ]
    payloads = []

    def transport(_url, _headers, payload, _timeout):
        payloads.append(payload)
        return responses.pop(0)

    result = _request(
        DeepSeekSolutionVerifier(
            api_key="secret",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            transport=transport,
        )
    )

    assert not result.accepted
    retry_prompt = payloads[1]["messages"][0]["content"]
    assert "FINDING_KEYS_MISMATCH" in retry_prompt
    assert "unexpected=['line']" in retry_prompt
    assert "code, severity, path, and message" in retry_prompt


def test_verifier_compact_retry_shares_one_total_wall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        _response("{malformed"),
        _response(
            {
                "accepted": True,
                "summary": "The implementation matches the approved plan.",
                "findings": [],
                "required_changes": [],
            }
        ),
    ]
    timeouts = []
    clock = iter((0.0, 0.0, 5.0, 5.0, 6.0, 6.0))
    monkeypatch.setattr(verifier_module.time, "monotonic", lambda: next(clock))

    def transport(_url, _headers, _payload, timeout):
        timeouts.append(timeout)
        return responses.pop(0)

    result = _request(
        DeepSeekSolutionVerifier(
            api_key="secret",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            transport=transport,
        )
    )

    assert result.accepted
    assert timeouts == [30, 25]
    assert result.wall_time_ms == 6000


def test_verifier_fails_closed_after_two_invalid_responses() -> None:
    def transport(_url, _headers, _payload, _timeout):
        return _response(
            {
                "accepted": True,
                "summary": "contradictory",
                "findings": [
                    {
                        "code": "ERROR",
                        "severity": "error",
                        "path": "solution/candidate.py",
                        "message": "blocking",
                    }
                ],
                "required_changes": ["fix it"],
            }
        )

    with pytest.raises(SolutionVerifierError) as failure:
        _request(
            DeepSeekSolutionVerifier(
                api_key="secret",
                model="deepseek-v4-flash",
                base_url="https://api.deepseek.com",
                transport=transport,
            )
        )
    assert failure.value.code == "SOLUTION_VERIFIER_MALFORMED"
    assert failure.value.input_tokens == 10
    assert failure.value.output_tokens == 6


def test_verifier_retries_transient_provider_without_rerunning_coder() -> None:
    payloads = []

    def transport(_url, _headers, payload, _timeout):
        payloads.append(payload)
        if len(payloads) == 1:
            raise RuntimeError("provider temporarily unavailable")
        return _response(
            {
                "accepted": True,
                "summary": "The implementation matches the approved plan.",
                "findings": [],
                "required_changes": [],
            }
        )

    result = _request(
        DeepSeekSolutionVerifier(
            api_key="secret",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            transport=transport,
        )
    )

    assert result.accepted
    assert result.provider_calls == 2
    assert "VERIFIER_PROVIDER_UNAVAILABLE" in payloads[1]["messages"][0]["content"]


def test_verifier_fails_after_bounded_provider_owner_retry() -> None:
    calls = 0

    def transport(_url, _headers, _payload, _timeout):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider temporarily unavailable")

    with pytest.raises(SolutionVerifierError) as failure:
        _request(
            DeepSeekSolutionVerifier(
                api_key="secret",
                model="deepseek-v4-flash",
                base_url="https://api.deepseek.com",
                transport=transport,
            )
        )

    assert calls == 2
    assert failure.value.code == "TRAE_PROVIDER_UNAVAILABLE"
    assert "after one owner-stage retry" in failure.value.summary
