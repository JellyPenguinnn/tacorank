from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tacorank.execution import resources
from tacorank.execution.sandbox import RuntimeMetricsSpec
from tacorank.execution.telemetry import (
    TelemetryCollector,
    TelemetryJournal,
    directive_action,
)

from .conftest import StubModel


def test_telemetry_jsonl_encodes_nonfinite_observation_explicitly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    sample = StubModel(loss=float("nan"), gradient_norm=float("inf"))
    with TelemetryJournal(path) as journal:
        journal.append(sample)

    payload = json.loads(path.read_text())
    assert payload == {"gradient_norm": "Infinity", "loss": "NaN"}


def test_invalid_observer_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid directive"):
        directive_action(SimpleNamespace(action="retry"))


def test_container_runtime_stats_drive_candidate_cpu_and_rss_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = RuntimeMetricsSpec(
        argv=("/reviewed/docker", "stats", "--no-stream", "container-id"),
        cwd=tmp_path,
        environment={"PATH": "/usr/bin"},
    )
    calls = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"CPUPerc":"12.5%","MemUsage":"256MiB / 4GiB"}\n',
            stderr="",
        )

    monkeypatch.setattr(resources.subprocess, "run", fake_run)

    class ContainerProcess:
        runtime_metrics = specification
        process_group_id = 999

        @staticmethod
        def recent_output_tail() -> str:
            return "loss=0.5"

        @staticmethod
        def is_alive() -> bool:
            return True

        @staticmethod
        def last_output_age_ms(now: float) -> int:
            del now
            return 1

    collector = TelemetryCollector(
        ContainerProcess(),  # type: ignore[arg-type]
        run_id="run_001",
        experiment_id="exp_0001",
        attempt=1,
        disk_path=tmp_path,
        model_factory=lambda name, **fields: StubModel(_model_name=name, **fields),
    )
    sample = collector.sample()

    assert sample.cpu_percent == 12.5
    assert sample.rss_mb == 256
    assert collector.last_cpu_seconds == 0.0
    assert calls[0][0] == list(specification.argv)
    assert calls[0][1]["shell"] is False


def test_container_stats_probe_retries_stay_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = RuntimeMetricsSpec(
        argv=("/reviewed/docker", "stats", "container-id"),
        cwd=tmp_path,
        environment={},
    )
    success = subprocess.CompletedProcess(
        specification.argv,
        0,
        stdout='{"CPUPerc":"1.0%","MemUsage":"1MiB / 1GiB"}\n',
        stderr="",
    )
    failure = subprocess.CompletedProcess(specification.argv, 1, stdout="", stderr="")
    responses = iter(
        (
            # Initial discovery tolerates a slow container registration.
            failure,
            success,
            # A transient post-discovery stall recovers within the bounded
            # retry budget instead of failing the whole execution.
            failure,
            failure,
            success,
            # A sustained loss still fails closed after the bounded retries.
            failure,
            failure,
            failure,
        )
    )
    calls = 0

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        del args, kwargs
        calls += 1
        return next(responses)

    monkeypatch.setattr(resources.subprocess, "run", fake_run)
    monkeypatch.setattr(resources.time, "sleep", lambda seconds: None)
    reader = resources.DockerStatsUsageReader(specification)

    assert reader.sample(0).rss_mb == 1.0
    assert calls == 2
    assert reader.sample(0).rss_mb == 1.0
    assert calls == 5
    with pytest.raises(resources.RuntimeMetricsError, match="unavailable"):
        reader.sample(0)
    assert calls == 8


def test_container_stats_initial_retry_never_exceeds_run_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = RuntimeMetricsSpec(
        argv=("/reviewed/docker", "stats", "container-id"),
        cwd=tmp_path,
        environment={},
        timeout_seconds=2.0,
    )
    calls: list[float] = []

    def unavailable(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del argv
        timeout = kwargs["timeout"]
        assert isinstance(timeout, float)
        calls.append(timeout)
        return subprocess.CompletedProcess((), 1, stdout="", stderr="")

    times = iter((0.0, 0.0, 0.02))
    monkeypatch.setattr(resources.subprocess, "run", unavailable)
    monkeypatch.setattr(resources.time, "monotonic", lambda: next(times))
    reader = resources.DockerStatsUsageReader(specification)

    with pytest.raises(resources.RuntimeMetricsError, match="unavailable"):
        reader.sample(0, timeout_seconds=0.01)
    assert len(calls) == 1
    assert 0 < calls[0] <= 0.01
