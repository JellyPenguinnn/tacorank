"""Portable process-group telemetry and action-local resource accounting."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

from tacorank.execution.interfaces import ModelFactory, default_model_factory
from tacorank.execution.sandbox import ResourceLimits, RuntimeMetricsSpec


@dataclass(frozen=True)
class ProcessUsage:
    cpu_percent: float
    cpu_seconds: Optional[float]
    rss_mb: float


class RuntimeMetricsError(RuntimeError):
    """Raised when candidate-container usage cannot be measured safely."""


class UsageReader(Protocol):
    def sample(self, process_group_id: int) -> ProcessUsage:
        ...


class ProcessUsageReader:
    """Read aggregate CPU/RSS for an isolated process group."""

    def __init__(self) -> None:
        self._previous_cpu_seconds: Optional[float] = None
        self._previous_monotonic: Optional[float] = None

    def sample(self, process_group_id: int) -> ProcessUsage:
        now = time.monotonic()
        if Path("/proc").is_dir():
            cpu_seconds, rss_mb = _linux_group_usage(process_group_id)
            cpu_percent = 0.0
            if (
                self._previous_cpu_seconds is not None
                and self._previous_monotonic is not None
                and now > self._previous_monotonic
            ):
                cpu_percent = max(
                    0.0,
                    (cpu_seconds - self._previous_cpu_seconds)
                    / (now - self._previous_monotonic)
                    * 100.0,
                )
            self._previous_cpu_seconds = cpu_seconds
            self._previous_monotonic = now
            return ProcessUsage(cpu_percent, cpu_seconds, rss_mb)
        return _ps_group_usage(process_group_id)


class DockerStatsUsageReader:
    """Measure the exact Docker container instead of the host Docker client."""

    _MEMORY = re.compile(
        r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)\s*/",
        re.IGNORECASE,
    )

    def __init__(self, specification: RuntimeMetricsSpec) -> None:
        self.specification = specification
        self._previous_monotonic: Optional[float] = None
        self._cpu_seconds = 0.0

    def sample(
        self,
        process_group_id: int,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ProcessUsage:
        del process_group_id
        line = self._read_stats_line(timeout_seconds=timeout_seconds)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeMetricsError("Docker candidate telemetry is malformed") from error
        if not isinstance(payload, dict):
            raise RuntimeMetricsError("Docker candidate telemetry is malformed")
        cpu_percent = _docker_cpu_percent(payload.get("CPUPerc"))
        rss_mb = _docker_memory_mb(payload.get("MemUsage"), self._MEMORY)
        now = time.monotonic()
        if self._previous_monotonic is not None and now >= self._previous_monotonic:
            self._cpu_seconds += (
                cpu_percent / 100.0 * (now - self._previous_monotonic)
            )
        self._previous_monotonic = now
        return ProcessUsage(cpu_percent, self._cpu_seconds, rss_mb)

    def _read_stats_line(self, *, timeout_seconds: Optional[float]) -> str:
        # The Docker client can be running a few milliseconds before the daemon
        # has registered the named container. Retry only initial discovery;
        # telemetry loss after the first successful sample remains fail-closed.
        attempts = 3 if self._previous_monotonic is None else 1
        budget = self.specification.timeout_seconds
        if timeout_seconds is not None:
            budget = min(budget, max(0.0, timeout_seconds))
        deadline = time.monotonic() + budget
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                completed = subprocess.run(
                    list(self.specification.argv),
                    cwd=str(self.specification.cwd),
                    env=dict(self.specification.environment),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    text=True,
                    shell=False,
                    close_fds=True,
                    timeout=max(0.001, remaining),
                )
            except (OSError, subprocess.SubprocessError) as error:
                if attempt + 1 == attempts:
                    raise RuntimeMetricsError(
                        "Docker candidate telemetry failed"
                    ) from error
            else:
                lines = [
                    line for line in completed.stdout.splitlines() if line.strip()
                ]
                if completed.returncode == 0 and len(lines) == 1:
                    return lines[0]
            if attempt + 1 < attempts:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))
        raise RuntimeMetricsError("Docker candidate telemetry is unavailable")


def _docker_cpu_percent(value: Any) -> float:
    if not isinstance(value, str) or not value.endswith("%"):
        raise RuntimeMetricsError("Docker CPU telemetry is malformed")
    try:
        parsed = float(value[:-1].strip())
    except ValueError as error:
        raise RuntimeMetricsError("Docker CPU telemetry is malformed") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise RuntimeMetricsError("Docker CPU telemetry is malformed")
    return parsed


def _docker_memory_mb(value: Any, pattern: re.Pattern[str]) -> float:
    if not isinstance(value, str):
        raise RuntimeMetricsError("Docker memory telemetry is malformed")
    match = pattern.match(value)
    if match is None:
        raise RuntimeMetricsError("Docker memory telemetry is malformed")
    amount = float(match.group(1))
    units = {
        "b": 1.0,
        "kb": 1000.0,
        "kib": 1024.0,
        "mb": 1000.0**2,
        "mib": 1024.0**2,
        "gb": 1000.0**3,
        "gib": 1024.0**3,
        "tb": 1000.0**4,
        "tib": 1024.0**4,
    }
    multiplier = units.get(match.group(2).lower())
    if multiplier is None or not math.isfinite(amount) or amount < 0:
        raise RuntimeMetricsError("Docker memory telemetry is malformed")
    return amount * multiplier / (1024.0**2)


class ResourceTracker:
    """Measure one coding/execution action without aggregating run totals."""

    def __init__(
        self,
        *,
        gpu_count: int = 0,
        model_factory: ModelFactory = default_model_factory,
    ) -> None:
        self._started = time.monotonic()
        self._gpu_count = max(0, gpu_count)
        self._model_factory = model_factory
        self._peak_rss_mb: Optional[float] = None
        self._peak_gpu_memory_mb: Optional[float] = None
        self._latest_cpu_seconds: Optional[float] = None

    def observe(
        self,
        *,
        rss_mb: Optional[float],
        gpu_memory_mb: Optional[float],
        cpu_seconds: Optional[float] = None,
    ) -> None:
        if rss_mb is not None:
            self._peak_rss_mb = max(self._peak_rss_mb or 0.0, rss_mb)
        if gpu_memory_mb is not None:
            self._peak_gpu_memory_mb = max(
                self._peak_gpu_memory_mb or 0.0, gpu_memory_mb
            )
        if cpu_seconds is not None:
            self._latest_cpu_seconds = max(
                self._latest_cpu_seconds or 0.0, cpu_seconds
            )

    def finish(self) -> Any:
        wall_time_ms = max(0, int((time.monotonic() - self._started) * 1000))
        cpu_time_ms = max(0, int((self._latest_cpu_seconds or 0.0) * 1000))
        gpu_time_ms = wall_time_ms if self._gpu_count > 0 else 0
        return self._model_factory(
            "ResourceDelta",
            llm_input_tokens=0,
            llm_output_tokens=0,
            token_measurement="none",
            wall_time_ms=wall_time_ms,
            cpu_time_ms=cpu_time_ms,
            gpu_time_ms=gpu_time_ms,
            gpu_count=self._gpu_count,
            peak_rss_mb=(
                None
                if self._peak_rss_mb is None
                else int(round(self._peak_rss_mb))
            ),
            peak_gpu_memory_mb=(
                None
                if self._peak_gpu_memory_mb is None
                else int(round(self._peak_gpu_memory_mb))
            ),
            manual_interventions=0,
        )


def observed_limit_pressure(sample: Any, limits: ResourceLimits) -> Optional[str]:
    """Return an observed threshold crossing, never an enforcement claim.

    Hard RAM/GPU enforcement belongs to the selected sandbox backend. Telemetry
    can still request an earlier bounded shutdown and classify the observation.
    """

    rss_mb = _field(sample, "rss_mb", None)
    if rss_mb is not None and float(rss_mb) > limits.memory_limit_mb:
        return "OBSERVED_CPU_MEMORY_LIMIT"
    gpu_memory_mb = _field(sample, "gpu_memory_mb", None)
    if (
        limits.gpu_memory_limit_mb > 0
        and gpu_memory_mb is not None
        and float(gpu_memory_mb) > limits.gpu_memory_limit_mb
    ):
        return "OBSERVED_GPU_MEMORY_LIMIT"
    disk_mb = _field(sample, "disk_free_mb", None)
    if disk_mb is not None and int(disk_mb) < limits.disk_free_floor_mb:
        return "OBSERVED_DISK_FREE_FLOOR"
    return None


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _linux_group_usage(process_group_id: int) -> Tuple[float, float]:
    clock_ticks = float(os.sysconf("SC_CLK_TCK"))
    total_ticks = 0
    total_rss_kb = 0
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            close_paren = stat_text.rfind(")")
            if close_paren < 0:
                continue
            fields = stat_text[close_paren + 2 :].split()
            if int(fields[2]) != process_group_id:
                continue
            total_ticks += int(fields[11]) + int(fields[12])
            status_text = (entry / "status").read_text(encoding="utf-8")
            for line in status_text.splitlines():
                if line.startswith("VmRSS:"):
                    total_rss_kb += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total_ticks / clock_ticks, total_rss_kb / 1024.0


def _ps_group_usage(process_group_id: int) -> ProcessUsage:
    ps = Path("/bin/ps")
    if not ps.exists():
        return ProcessUsage(0.0, None, 0.0)
    try:
        completed = subprocess.run(
            [str(ps), "-axo", "pgid=,%cpu=,rss="],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ProcessUsage(0.0, None, 0.0)
    cpu_percent = 0.0
    rss_kb = 0.0
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            if int(fields[0]) != process_group_id:
                continue
            cpu_percent += float(fields[1])
            rss_kb += float(fields[2])
        except ValueError:
            continue
    return ProcessUsage(max(0.0, cpu_percent), None, max(0.0, rss_kb / 1024.0))
