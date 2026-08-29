"""Live telemetry sampling and append-only JSONL persistence."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple

from tacorank.execution.interfaces import (
    ModelFactory,
    default_model_factory,
    model_to_mapping,
)
from tacorank.execution.process import ManagedProcess
from tacorank.execution.resources import (
    DockerStatsUsageReader,
    ProcessUsageReader,
    UsageReader,
)
from tacorank.execution.sandbox import disk_free_mb


_LOSS_PATTERN = re.compile(
    r"(?i)(?:^|[\s,;])loss\s*[:=]\s*"
    r"([-+]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?))"
)
_GRADIENT_PATTERN = re.compile(
    r"(?i)(?:gradient[_\s-]*norm|grad[_\s-]*norm)\s*[:=]\s*"
    r"([-+]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?))"
)


class GPUMetricsReader(Protocol):
    def sample(self, process_group_id: int) -> Tuple[Optional[float], Optional[float]]:
        ...


class NullGPUMetricsReader:
    def sample(self, process_group_id: int) -> Tuple[None, None]:
        del process_group_id
        return None, None


class NvidiaSMIMetricsReader:
    """Best-effort aggregate NVIDIA telemetry without adding a Python SDK."""

    def __init__(self, executable: Optional[str] = None) -> None:
        located = executable or shutil.which("nvidia-smi")
        self.executable = None if located is None else str(Path(located).resolve())

    def sample(self, process_group_id: int) -> Tuple[Optional[float], Optional[float]]:
        del process_group_id  # nvidia-smi cannot portably filter a process group
        if self.executable is None:
            return None, None
        try:
            completed = subprocess.run(
                [
                    self.executable,
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.0,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None, None
        utilization = []
        memory = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                utilization.append(float(fields[0]))
                memory.append(float(fields[1]))
            except ValueError:
                continue
        if not utilization:
            return None, None
        return sum(utilization) / len(utilization), sum(memory)


class TelemetryCollector:
    """Collect one canonical sample for a managed process."""

    def __init__(
        self,
        process: ManagedProcess,
        *,
        run_id: str,
        experiment_id: str,
        attempt: int,
        disk_path: Path,
        started_monotonic: Optional[float] = None,
        deadline_monotonic: Optional[float] = None,
        usage_reader: Optional[UsageReader] = None,
        gpu_reader: Optional[GPUMetricsReader] = None,
        model_factory: ModelFactory = default_model_factory,
    ) -> None:
        self.process = process
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.attempt = attempt
        self.disk_path = Path(disk_path)
        self.started_monotonic = (
            time.monotonic()
            if started_monotonic is None
            else started_monotonic
        )
        self.deadline_monotonic = deadline_monotonic
        if usage_reader is not None:
            self.usage_reader = usage_reader
        elif process.runtime_metrics is not None:
            self.usage_reader = DockerStatsUsageReader(process.runtime_metrics)
        else:
            self.usage_reader = ProcessUsageReader()
        self.gpu_reader = gpu_reader or NullGPUMetricsReader()
        self.model_factory = model_factory
        self.last_cpu_seconds: Optional[float] = None

    def sample(self) -> Any:
        now = time.monotonic()
        if isinstance(self.usage_reader, DockerStatsUsageReader):
            remaining = (
                None
                if self.deadline_monotonic is None
                else max(0.0, self.deadline_monotonic - now)
            )
            usage = self.usage_reader.sample(
                self.process.process_group_id,
                timeout_seconds=remaining,
            )
        else:
            usage = self.usage_reader.sample(self.process.process_group_id)
        self.last_cpu_seconds = usage.cpu_seconds
        gpu_utilization, gpu_memory = self.gpu_reader.sample(
            self.process.process_group_id
        )
        output_tail = self.process.recent_output_tail()
        loss = _last_metric(_LOSS_PATTERN, output_tail)
        gradient_norm = _last_metric(_GRADIENT_PATTERN, output_tail)
        return self.model_factory(
            "TelemetrySample",
            timestamp=datetime.now(timezone.utc),
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            attempt=self.attempt,
            elapsed_ms=max(0, int((now - self.started_monotonic) * 1000)),
            process_alive=self.process.is_alive(),
            last_output_age_ms=self.process.last_output_age_ms(now),
            cpu_percent=max(0.0, usage.cpu_percent),
            rss_mb=max(0.0, usage.rss_mb),
            gpu_utilization_percent=gpu_utilization,
            gpu_memory_mb=(
                None if gpu_memory is None else max(0, int(round(gpu_memory)))
            ),
            loss=loss,
            gradient_norm=gradient_norm,
            disk_free_mb=disk_free_mb(self.disk_path),
            recent_output_tail=output_tail or None,
        )


class TelemetryJournal:
    """Append complete samples to a bounded-lifetime JSONL artifact."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("telemetry path cannot be a symlink")
        self._handle = self.path.open("x", encoding="utf-8")
        self._closed = False

    def append(self, sample: Any) -> None:
        if self._closed:
            raise RuntimeError("telemetry journal is closed")
        payload = json.dumps(
            _json_safe(model_to_mapping(sample)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        )
        self._handle.write(payload + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True

    def __enter__(self) -> "TelemetryJournal":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def directive_action(directive: Any) -> str:
    action = _field(directive, "action")
    value = getattr(action, "value", action)
    normalized = str(value).lower()
    if normalized not in {"continue", "terminate"}:
        raise ValueError("observer returned an invalid directive action")
    return normalized


def directive_reason(directive: Any) -> Tuple[Optional[str], Optional[str]]:
    reason = _field(directive, "reason_code", None)
    summary = _field(directive, "summary", None)
    return (
        None if reason is None else str(reason),
        None if summary is None else str(summary)[:512],
    )


def _field(value: Any, name: str, default: Any = ...) -> Any:
    if isinstance(value, Mapping):
        if default is ...:
            return value[name]
        return value.get(name, default)
    if default is ...:
        return getattr(value, name)
    return getattr(value, name, default)


def _last_metric(pattern: re.Pattern[str], text: str) -> Optional[float]:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    value = matches[-1].group(1).lower()
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return float("nan")
    if math.isinf(parsed):
        return parsed
    return parsed


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError("telemetry value is not JSON serializable")


def _json_safe(value: Any) -> Any:
    """Encode non-finite observed values explicitly while keeping valid JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
