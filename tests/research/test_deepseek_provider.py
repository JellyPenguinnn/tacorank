from __future__ import annotations

import asyncio
import io
import json
from urllib.error import HTTPError

import pytest

from tacorank.agents.research_planner import ResearchPlanner
from tacorank.cli import _planner_for
from tacorank.providers.deepseek import DeepSeekResearchProvider
from tacorank.providers.research_provider import ProviderError, ProviderRequest
from tacorank.research.duplicate_detection import compute_duplicate_key
from tacorank.research.search_policy import SearchPolicy
from tacorank.schemas import TokenMeasurement


def candidate(**updates):
    value = {
        "hypothesis": "Pairwise BPR should improve within-user ranking.",
        "change_summary": "Replace pointwise loss with pairwise BPR.",
        "target_stage": "objective",
        "target_files": ["solution/loss.py"],
        "fidelity_plan": ["smoke", "proxy", "full"],
        "expected_mechanism": "Optimize relative positive-negative ordering.",
        "success_criteria": "Full primary delta is at least 0.002.",
        "falsification_condition": "No trusted full-fidelity gain.",
        "estimated_cost": {
            "llm_tokens_upper_bound": 800,
            "wall_time_seconds_upper_bound": 300,
            "gpu_seconds_upper_bound": 60,
            "cost_tier": "high",
        },
        "method_card_ids": ["objective_pairwise_bpr"],
        "evidence_event_ids": ["evt_000001", "evt_not_in_context"],
        # These are deliberately hostile: the adapter must replace policy-owned fields.
        "run_id": "wrong_run",
        "experiment_id": "wrong_experiment",
        "parent_experiment_id": "wrong_parent",
        "parent_commit_sha": "f" * 40,
        "context_id": "wrong_context",
        "family": "model",
    }
    value.update(updates)
    return value


def response(value, *, finish_reason="stop", prompt_tokens=101, completion_tokens=37):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": json.dumps(value)},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def output_factory(action, spec, reason_code, reason, supporting_event_ids):
    return {
        "action": action,
        "spec": spec,
        "reason_code": reason_code,
        "reason": reason,
        "supporting_event_ids": supporting_event_ids,
    }


def test_deepseek_provider_constrains_policy_fields_and_records_usage(planner_context):
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return response(candidate())

    choice = SearchPolicy().choose(planner_context)
    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    result = asyncio.run(
        provider.generate(
            ProviderRequest(
                context=planner_context,
                policy_choice=choice,
                input_token_limit=2_000,
                output_token_limit=1_000,
            )
        )
    )

    assert result["run_id"] == planner_context.run_id
    assert result["context_id"] == planner_context.context_id
    assert result["experiment_id"] == "exp_0001"
    assert result["parent_experiment_id"] == choice.parent.experiment_id
    assert result["parent_commit_sha"] == choice.parent.parent_commit_sha
    assert result["family"] == choice.family
    assert result["estimated_cost"]["cost_tier"] == choice.cost_tier
    assert result["evidence_event_ids"] == ["evt_000001"]
    assert result["duplicate_key"] == compute_duplicate_key(result)

    url, headers, payload, timeout = calls[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert headers["Authorization"] == "Bearer secret-key"
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 1_000
    assert payload["thinking"] == {"type": "enabled"}
    assert "secret-key" not in json.dumps(payload)
    assert timeout == 120
    assert provider.resource_delta.llm_input_tokens == 101
    assert provider.resource_delta.llm_output_tokens == 37
    assert provider.resource_delta.token_measurement == TokenMeasurement.PROVIDER


def test_research_planner_requests_one_deepseek_repair(planner_context):
    responses = [
        response(candidate(target_files=[]), prompt_tokens=100, completion_tokens=20),
        response(candidate(), prompt_tokens=90, completion_tokens=30),
    ]
    requests = []

    def transport(url, headers, payload, timeout):
        requests.append(payload)
        return responses.pop(0)

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    planner = ResearchPlanner(
        provider,
        output_factory=output_factory,
        input_token_limit=2_000,
        output_token_limit=1_000,
    )

    result = asyncio.run(planner.propose(planner_context))

    assert result["action"] == "propose"
    assert len(requests) == 2
    repair_prompt = json.loads(requests[1]["messages"][1]["content"])
    assert "NO_TARGET_FILES" in repair_prompt["repair"]["validation_errors"]
    assert result["resource_delta"].llm_input_tokens == 190
    assert result["resource_delta"].llm_output_tokens == 50


def test_deepseek_provider_rejects_truncated_completion(planner_context):
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return response(candidate(), finish_reason="length")

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    choice = SearchPolicy().choose(planner_context)

    with pytest.raises(ProviderError, match="did not finish cleanly"):
        asyncio.run(provider.generate(ProviderRequest(planner_context, choice)))

    assert len(calls) == 2
    assert calls[0]["thinking"] == {"type": "enabled"}
    assert calls[1]["thinking"] == {"type": "disabled"}
    assert "compactly" in calls[1]["messages"][0]["content"]


def test_deepseek_provider_retries_length_once_without_thinking(planner_context):
    responses = [
        response(candidate(), finish_reason="length", completion_tokens=1_000),
        response(candidate(), completion_tokens=100),
    ]
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return responses.pop(0)

    provider = DeepSeekResearchProvider(api_key="secret-key", transport=transport)
    choice = SearchPolicy().choose(planner_context)

    result = asyncio.run(provider.generate(ProviderRequest(planner_context, choice)))

    assert result["experiment_id"] == "exp_0001"
    assert len(calls) == 2
    assert calls[1]["thinking"] == {"type": "disabled"}
    assert provider.resource_delta.llm_output_tokens == 1_100


def test_cli_selects_deepseek_without_putting_secret_in_config(config, monkeypatch):
    configured = config.model_copy(update={"research_provider": "deepseek"})
    monkeypatch.setenv(configured.deepseek_api_key_env, "secret-key")

    planner = _planner_for(configured)

    assert isinstance(planner, ResearchPlanner)
    assert isinstance(planner.provider, DeepSeekResearchProvider)
    assert "secret-key" not in json.dumps(configured.canonical_dict(), sort_keys=True)


def test_cli_fails_closed_when_deepseek_key_is_missing(config, monkeypatch):
    configured = config.model_copy(update={"research_provider": "deepseek"})
    monkeypatch.delenv(configured.deepseek_api_key_env, raising=False)

    with pytest.raises(ProviderError, match=configured.deepseek_api_key_env):
        _planner_for(configured)


def test_deepseek_preflight_authenticates_and_requires_configured_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @staticmethod
        def read(limit):
            assert limit == 1024 * 1024
            return json.dumps(
                {"object": "list", "data": [{"id": "deepseek-v4-pro"}]}
            ).encode("utf-8")

    requests = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr("tacorank.providers.deepseek.urlopen", open_request)
    provider = DeepSeekResearchProvider(api_key="secret-key")

    provider.preflight()

    request, timeout = requests[0]
    assert request.full_url == "https://api.deepseek.com/models"
    assert request.headers["Authorization"] == "Bearer secret-key"
    assert timeout == 30


def test_deepseek_preflight_redacts_http_failure_detail(monkeypatch):
    def reject(request, timeout):
        del request, timeout
        raise HTTPError(
            "https://api.deepseek.com/models",
            401,
            "bad secret-key",
            {},
            io.BytesIO(b'{"error":"secret-key"}'),
        )

    monkeypatch.setattr("tacorank.providers.deepseek.urlopen", reject)
    provider = DeepSeekResearchProvider(api_key="secret-key")

    with pytest.raises(ProviderError, match="HTTP 401") as captured:
        provider.preflight()

    assert "secret-key" not in str(captured.value)
