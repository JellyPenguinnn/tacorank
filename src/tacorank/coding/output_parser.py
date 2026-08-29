"""Strict normalization of Trae Agent trajectory output.

The parser targets Trae Agent's documented JSON trajectory format.  It never
infers missing usage or step metadata because fabricated accounting would break
the evidence chain consumed by the controller.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_MAX_TRAJECTORY_BYTES = 50 * 1024 * 1024


class TrajectoryParseError(ValueError):
    """A classified malformed or incomplete Trae trajectory."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    measurement: str = "provider"


@dataclass(frozen=True)
class ParsedTrajectory:
    provider: str
    model: str
    max_steps: int
    steps_used: int
    success: bool
    final_result: Optional[str]
    usage: TokenUsage
    raw: Mapping[str, Any]


def parse_trajectory_file(
    path: Path, *, max_bytes: int = DEFAULT_MAX_TRAJECTORY_BYTES
) -> ParsedTrajectory:
    """Read and parse a bounded UTF-8 Trae trajectory file."""

    trajectory_path = Path(path)
    try:
        size = trajectory_path.stat().st_size
    except FileNotFoundError as exc:
        raise TrajectoryParseError("TRAJECTORY_MISSING", "Trae trajectory is missing") from exc
    if not trajectory_path.is_file() or trajectory_path.is_symlink():
        raise TrajectoryParseError(
            "TRAJECTORY_INVALID_FILE", "trajectory must be a regular non-symlink file"
        )
    if size > max_bytes:
        raise TrajectoryParseError(
            "TRAJECTORY_TOO_LARGE", f"trajectory exceeds {max_bytes} bytes"
        )
    return parse_trajectory_bytes(trajectory_path.read_bytes(), max_bytes=max_bytes)


def parse_trajectory_bytes(
    value: bytes, *, max_bytes: int = DEFAULT_MAX_TRAJECTORY_BYTES
) -> ParsedTrajectory:
    """Parse a complete Trae trajectory without filling absent evidence."""

    if len(value) > max_bytes:
        raise TrajectoryParseError(
            "TRAJECTORY_TOO_LARGE", f"trajectory exceeds {max_bytes} bytes"
        )
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrajectoryParseError(
            "TRAJECTORY_ENCODING", "trajectory is not valid UTF-8"
        ) from exc
    try:
        document = json.loads(decoded)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", "trajectory is not a valid JSON document"
        ) from exc
    if not isinstance(document, dict):
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", "trajectory root must be an object"
        )

    _required_nonempty_string(document, "task")
    _required_nonempty_string(document, "start_time")
    _required_nonempty_string(document, "end_time")
    provider = _required_nonempty_string(document, "provider")
    model = _required_nonempty_string(document, "model")
    max_steps = _required_nonnegative_int(document, "max_steps", allow_zero=False)
    success = document.get("success")
    if not isinstance(success, bool):
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", "trajectory success must be boolean"
        )
    final_result = document.get("final_result")
    if final_result is not None and not isinstance(final_result, str):
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", "trajectory final_result must be text or null"
        )

    execution_time = document.get("execution_time")
    if (
        isinstance(execution_time, bool)
        or not isinstance(execution_time, (int, float))
        or not math.isfinite(execution_time)
        or execution_time < 0
    ):
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", "trajectory execution_time must be finite and non-negative"
        )

    interactions = document.get("llm_interactions")
    if not isinstance(interactions, list) or not interactions:
        raise TrajectoryParseError(
            "TOKEN_USAGE_MISSING", "trajectory has no LLM interactions to account"
        )
    usage = _aggregate_usage(interactions, provider=provider, model=model)

    steps = document.get("agent_steps")
    if not isinstance(steps, list):
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", "trajectory agent_steps must be a list"
        )
    step_numbers = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise TrajectoryParseError(
                "TRAJECTORY_MALFORMED", f"agent_steps[{index}] must be an object"
            )
        number = step.get("step_number")
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise TrajectoryParseError(
                "TRAJECTORY_MALFORMED",
                f"agent_steps[{index}].step_number must be a positive integer",
            )
        step_numbers.append(number)
    steps_used = len(step_numbers)
    if steps_used > max_steps or (step_numbers and max(step_numbers) > max_steps):
        raise TrajectoryParseError(
            "STEP_LIMIT_EXCEEDED", "trajectory exceeds its declared maximum steps"
        )
    if step_numbers != list(range(1, len(step_numbers) + 1)):
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", "agent step numbers must be contiguous from one"
        )
    if success and steps_used == 0:
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", "successful trajectory has no agent steps"
        )

    return ParsedTrajectory(
        provider=provider,
        model=model,
        max_steps=max_steps,
        steps_used=steps_used,
        success=success,
        final_result=final_result,
        usage=usage,
        raw=document,
    )


def _aggregate_usage(
    interactions: Sequence[Any], *, provider: str, model: str
) -> TokenUsage:
    totals: Dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    for index, interaction in enumerate(interactions):
        if not isinstance(interaction, dict):
            raise TrajectoryParseError(
                "TRAJECTORY_MALFORMED", f"llm_interactions[{index}] must be an object"
            )
        if interaction.get("provider") != provider or interaction.get("model") != model:
            raise TrajectoryParseError(
                "TRAJECTORY_MALFORMED",
                f"llm_interactions[{index}] provider/model differs from trajectory",
            )
        response = interaction.get("response")
        if not isinstance(response, dict):
            raise TrajectoryParseError(
                "TOKEN_USAGE_MISSING", f"llm_interactions[{index}] has no response"
            )
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise TrajectoryParseError(
                "TOKEN_USAGE_MISSING", f"llm_interactions[{index}] has no usage"
            )
        for field in ("input_tokens", "output_tokens"):
            value = usage.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrajectoryParseError(
                    "TOKEN_USAGE_MISSING",
                    f"llm_interactions[{index}].usage.{field} is missing or invalid",
                )
            totals[field] += value
        for field in (
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "reasoning_tokens",
        ):
            value = usage.get(field, 0)
            if value is None:
                value = 0
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrajectoryParseError(
                    "TOKEN_USAGE_INVALID",
                    f"llm_interactions[{index}].usage.{field} is invalid",
                )
            totals[field] += value
    return TokenUsage(
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        cache_creation_input_tokens=totals["cache_creation_input_tokens"],
        cache_read_input_tokens=totals["cache_read_input_tokens"],
        reasoning_tokens=totals["reasoning_tokens"],
    )


def _required_nonempty_string(document: Mapping[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", f"trajectory {field} must be non-empty text"
        )
    return value


def _required_nonnegative_int(
    document: Mapping[str, Any], field: str, *, allow_zero: bool
) -> int:
    value = document.get(field)
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrajectoryParseError(
            "TRAJECTORY_MALFORMED", f"trajectory {field} must be >= {minimum}"
        )
    return value
