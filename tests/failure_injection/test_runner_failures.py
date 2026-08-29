from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from tacorank.execution.process import ManagedProcess, ProcessLaunchError, ProcessLauncher
from tacorank.execution.sandbox import (
    IsolationGuarantees,
    LaunchSpec,
    ResourceLimits,
    RuntimeCleanupSpec,
)

from tests.execution.conftest import (
    ContinuingObserver,
    StubModel,
    build_runner,
    command_registry,
    request,
)


pytestmark = pytest.mark.failure_injection


class TerminatingObserver:
    def __init__(self, reason_code: str, summary: str) -> None:
        self.reason_code = reason_code
        self.summary = summary
        self.calls = 0

    def observe(self, sample: Any) -> StubModel:
        self.calls += 1
        return StubModel(
            action="terminate",
            reason_code=self.reason_code,
            summary=self.summary,
        )


class RecordingLauncher(ProcessLauncher):
    def __init__(self) -> None:
        self.managed: Optional[ManagedProcess] = None

    def launch(self, specification: Any, log_path: Path, limits: Any):
        self.managed = super().launch(specification, log_path, limits)
        return self.managed


def test_timeout_kills_full_process_group_and_reaps_leader(
    execution_layout: SimpleNamespace,
) -> None:
    code = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "print('spawned', child.pid, flush=True); time.sleep(60)"
    )
    registry = command_registry(code, extra_arguments=("{artifact_dir}/child.pid",))
    launcher = RecordingLauncher()
    runner, _ = build_runner(
        execution_layout,
        registry,
        process_launcher=launcher,
        interval=0.02,
    )

    result = runner.run_sync(
        request(timeout_seconds=0.18), ContinuingObserver()
    )

    assert result.outcome == "timeout"
    assert result.error_class == "WALL_TIMEOUT"
    assert launcher.managed is not None
    assert launcher.managed.return_code is not None
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.killpg(launcher.managed.process_group_id, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("timed-out process group still exists")


def test_observer_termination_maps_hang_and_cleans_process(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("import time; print('ready', flush=True); time.sleep(60)")
    launcher = RecordingLauncher()
    runner, _ = build_runner(
        execution_layout,
        registry,
        process_launcher=launcher,
    )
    observer = TerminatingObserver("HEARTBEAT_STALE", "no progress")

    result = runner.run_sync(request(timeout_seconds=5), observer)

    assert result.outcome == "hang"
    assert result.error_class == "HEARTBEAT_STALE"
    assert result.error_summary == "no progress"
    assert observer.calls == 1
    assert launcher.managed is not None
    assert launcher.managed.return_code is not None


def test_normal_parent_exit_still_cleans_background_descendants(
    execution_layout: SimpleNamespace,
) -> None:
    code = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print('leader complete', flush=True)"
    )
    launcher = RecordingLauncher()
    runner, _ = build_runner(
        execution_layout,
        command_registry(code),
        process_launcher=launcher,
    )

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "success"
    assert launcher.managed is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(launcher.managed.process_group_id, 0)


def test_nonzero_oom_signature_is_normalized(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry(
        "import sys; print('CUDA out of memory', flush=True); sys.exit(2)"
    )
    runner, _ = build_runner(execution_layout, registry)

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "oom"
    assert result.error_class == "OUT_OF_MEMORY"


class HighMemoryCollector:
    def __init__(self, process: Any, **kwargs: Any) -> None:
        del kwargs
        self.process = process
        self.last_cpu_seconds = 0.0

    def sample(self) -> StubModel:
        return StubModel(
            timestamp="2026-08-29T00:00:00Z",
            run_id="run_001",
            experiment_id="exp_0001",
            attempt=1,
            elapsed_ms=1,
            process_alive=self.process.is_alive(),
            last_output_age_ms=1,
            cpu_percent=0.0,
            rss_mb=5000.0,
            gpu_utilization_percent=None,
            gpu_memory_mb=None,
            loss=None,
            gradient_norm=None,
            disk_free_mb=10000,
            recent_output_tail=None,
        )


def test_telemetry_memory_pressure_requests_early_bounded_shutdown(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("import time; time.sleep(60)")
    runner, _ = build_runner(
        execution_layout,
        registry,
        telemetry_collector_factory=HighMemoryCollector,
    )

    result = runner.run_sync(
        request(memory_limit_mb=4096), ContinuingObserver()
    )

    assert result.outcome == "oom"
    assert result.error_class == "OBSERVED_CPU_MEMORY_LIMIT"


def _direct_launch_spec(
    argv: tuple[str, ...],
    cwd: Path,
    runtime_cleanup: Optional[RuntimeCleanupSpec] = None,
) -> LaunchSpec:
    return LaunchSpec(
        argv=argv,
        cwd=cwd,
        environment={"PATH": os.environ.get("PATH", os.defpath)},
        preexec_fn=None,
        start_new_session=True,
        guarantees=IsolationGuarantees(False, False, False, False, False, False, True),
        runtime_cleanup=runtime_cleanup,
    )


def test_termination_removes_runtime_even_after_cli_has_already_exited(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "container-alive"
    marker.write_text("alive", encoding="utf-8")
    python = str(Path(sys.executable).resolve())
    environment = {"PATH": os.environ.get("PATH", os.defpath)}
    cleanup = RuntimeCleanupSpec(
        terminate_argv=(
            python,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).unlink(missing_ok=True)",
            str(marker),
        ),
        inspect_argv=(
            python,
            "-c",
            "from pathlib import Path; import sys; print('container-alive' if Path(sys.argv[1]).exists() else '')",
            str(marker),
        ),
        healthcheck_argv=(python, "-c", "raise SystemExit(0)"),
        cwd=tmp_path,
        environment=environment,
    )
    launch = _direct_launch_spec(
        (python, "-c", "raise SystemExit(0)"),
        tmp_path,
        cleanup,
    )
    managed = ProcessLauncher().launch(
        launch,
        tmp_path / "execution.log",
        ResourceLimits(5, 1024, 0),
    )
    assert managed.wait(timeout=2) == 0
    assert marker.exists()

    assert managed.terminate_group(grace_seconds=0) == 0
    managed.close_after_termination()
    assert not marker.exists()


def test_runtime_absence_probe_failure_is_not_treated_as_absence(
    tmp_path: Path,
) -> None:
    python = str(Path(sys.executable).resolve())
    cleanup = RuntimeCleanupSpec(
        terminate_argv=(python, "-c", "raise SystemExit(0)"),
        inspect_argv=(python, "-c", "raise SystemExit(1)"),
        healthcheck_argv=(python, "-c", "raise SystemExit(0)"),
        cwd=tmp_path,
        environment={"PATH": os.environ.get("PATH", os.defpath)},
    )
    managed = ProcessLauncher().launch(
        _direct_launch_spec((python, "-c", "raise SystemExit(0)"), tmp_path, cleanup),
        tmp_path / "execution.log",
        ResourceLimits(5, 1024, 0),
    )
    assert managed.wait(timeout=2) == 0
    with pytest.raises(ProcessLaunchError, match="absence probe failed"):
        managed.terminate_group(grace_seconds=0)
    managed.close_after_termination()


def test_authoritative_runtime_oom_state_overrides_log_heuristics(
    execution_layout: SimpleNamespace,
) -> None:
    python = str(Path(sys.executable).resolve())
    cleanup = RuntimeCleanupSpec(
        terminate_argv=(python, "-c", "raise SystemExit(0)"),
        inspect_argv=(python, "-c", "print('')"),
        healthcheck_argv=(python, "-c", "raise SystemExit(0)"),
        state_argv=(
            python,
            "-c",
            "print('{\"OOMKilled\":true,\"ExitCode\":137}')",
        ),
        cwd=execution_layout.workspace,
        environment={"PATH": os.environ.get("PATH", os.defpath)},
    )

    class RuntimeStateSandbox:
        @staticmethod
        def prepare(command: Any, configuration: Any) -> LaunchSpec:
            del configuration
            return _direct_launch_spec(command.argv, command.cwd, cleanup)

    runner, _ = build_runner(
        execution_layout,
        command_registry("import sys; sys.exit(137)"),
    )
    runner.sandbox = RuntimeStateSandbox()
    result = runner.run_sync(request(), ContinuingObserver())
    assert result.outcome == "oom"
    assert result.error_class == "OUT_OF_MEMORY"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_log_writer_keeps_exclusive_descriptor_across_path_substitution(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "execution.log"
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched\n", encoding="utf-8")
    code = (
        "from pathlib import Path; import sys; "
        "log=Path(sys.argv[1]); log.unlink(); log.symlink_to(Path(sys.argv[2])); "
        "print('candidate output', flush=True)"
    )
    launch = _direct_launch_spec(
        (str(Path(sys.executable).resolve()), "-c", code, str(log_path), str(victim)),
        tmp_path,
    )
    managed = ProcessLauncher().launch(
        launch,
        log_path,
        ResourceLimits(5, 1024, 0),
    )

    assert managed.finish() == 0
    assert log_path.is_symlink()
    assert victim.read_text(encoding="utf-8") == "untouched\n"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_preexisting_log_symlink_is_rejected_without_touching_target(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "execution.log"
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched\n", encoding="utf-8")
    log_path.symlink_to(victim)
    launch = _direct_launch_spec(
        (str(Path(sys.executable).resolve()), "-c", "print('never')"),
        tmp_path,
    )

    with pytest.raises(ProcessLaunchError, match="exclusively"):
        ProcessLauncher().launch(launch, log_path, ResourceLimits(5, 1024, 0))
    assert victim.read_text(encoding="utf-8") == "untouched\n"


class FailingLauncher:
    def launch(self, specification: Any, log_path: Path, limits: Any) -> None:
        del specification, log_path, limits
        raise ProcessLaunchError("container runtime unavailable")


def test_executor_launch_loss_returns_infrastructure_result(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("print('never launched')")
    runner, _ = build_runner(
        execution_layout,
        registry,
        process_launcher=FailingLauncher(),
    )

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "infrastructure_error"
    assert result.exit_code is None
    assert result.error_class == "ProcessLaunchError"
    assert "container runtime unavailable" in result.error_summary


class FailingTelemetryCollector:
    def __init__(self, process: Any, **kwargs: Any) -> None:
        del process, kwargs
        self.last_cpu_seconds = None

    def sample(self) -> StubModel:
        raise RuntimeError("injected telemetry loss")


def test_telemetry_loss_terminates_process_as_infrastructure_error(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("import time; time.sleep(60)")
    runner, _ = build_runner(
        execution_layout,
        registry,
        telemetry_collector_factory=FailingTelemetryCollector,
    )

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "infrastructure_error"
    assert result.error_class == "TELEMETRY_FAILURE"
