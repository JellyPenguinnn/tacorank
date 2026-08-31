"""Sealed, observable execution of reviewed symbolic commands."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import math
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping, Optional, Protocol, Tuple

from tacorank.execution.artifacts import (
    CapturedOutputs,
    RunArtifactManager,
    safe_request_summary,
)
from tacorank.execution.interfaces import (
    ExecutionArtifactStore,
    ModelFactory,
    default_model_factory,
)
from tacorank.execution.commands import CommandContext, CommandPolicyError, CommandRegistry
from tacorank.execution.process import (
    DiskSpaceExhausted,
    OutputQuotaExceeded,
    ProcessLaunchError,
    ProcessLauncher,
    redact_runtime_output,
)
from tacorank.execution.resources import ResourceTracker, observed_limit_pressure
from tacorank.execution.sandbox import (
    ExecutionSandbox,
    ResourceLimits,
    SandboxConfig,
    SandboxPolicyError,
    validate_launch_spec,
)
from tacorank.execution.telemetry import (
    GPUMetricsReader,
    NullGPUMetricsReader,
    TelemetryCollector,
    TelemetryJournal,
    directive_action,
    directive_reason,
)


# Docker Desktop can stall a single ``docker stats`` probe for several
# seconds while the candidate is CPU-bound; only a sustained loss of
# telemetry is evidence of an unhealthy runtime.
_MAX_CONSECUTIVE_TELEMETRY_MISSES = 3


class ExecutionAuthorizationError(RuntimeError):
    """Raised when receipt, commit, diff, or protected hashes do not match."""


class InvalidRunRequest(ValueError):
    """Raised when a request could not have passed the canonical schema."""


class ExecutionSealVerifier(Protocol):
    """Verify the exact accepted receipt and workspace identity before launch."""

    def verify(self, request: Any, workspace: Path) -> Any:
        ...

    def acquire_lease(
        self,
        request: Any,
        workspace: Path,
        *,
        timeout_seconds: float,
    ) -> ContextManager[Any]:
        ...


class HealthObserver(Protocol):
    def observe(self, sample: Any) -> Any:
        ...


class SubmissionArtifactResolver(Protocol):
    """Resolve Person 2's prior verified prediction ArtifactRef."""

    def resolve(self, request: Any) -> Any:
        ...


class DenyUnverifiedExecution:
    """Secure default: a runner cannot execute without receipt integration."""

    def verify(self, request: Any, workspace: Path) -> None:
        del request, workspace
        raise ExecutionAuthorizationError("accepted patch receipt verifier is required")

    def acquire_lease(
        self,
        request: Any,
        workspace: Path,
        *,
        timeout_seconds: float,
    ) -> ContextManager[Any]:
        del request, workspace, timeout_seconds
        raise ExecutionAuthorizationError("accepted patch receipt verifier is required")


@dataclass(frozen=True)
class RunnerPolicy:
    max_timeout_seconds: float = 24 * 60 * 60
    max_memory_limit_mb: int = 1024 * 1024
    max_gpu_memory_limit_mb: int = 1024 * 1024
    disk_free_floor_mb: int = 128
    max_log_bytes: int = 16 * 1024 * 1024
    max_open_files: int = 256
    max_processes: int = 128
    termination_grace_seconds: float = 2.0
    telemetry_interval_seconds: float = 2.0
    worktree_lease_timeout_seconds: float = 30.0
    allow_trusted_local_backend: bool = False

    def __post_init__(self) -> None:
        if self.max_timeout_seconds <= 0:
            raise ValueError("maximum timeout must be positive")
        if self.max_memory_limit_mb <= 0:
            raise ValueError("maximum memory must be positive")
        if self.max_gpu_memory_limit_mb < 0:
            raise ValueError("maximum GPU memory must be non-negative")
        if self.telemetry_interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        if not 0 < self.worktree_lease_timeout_seconds <= 300:
            raise ValueError("worktree lease timeout must be in (0, 300] seconds")


WorkspaceResolver = Callable[[str, str], Path]
TelemetryCollectorFactory = Callable[..., TelemetryCollector]


class ExecutionRunner:
    """Real Person 3 adapter implementing the shared ``ExecutionRunner`` port."""

    def __init__(
        self,
        *,
        repository_root: Path,
        artifacts: ExecutionArtifactStore,
        commands: CommandRegistry,
        sandbox: ExecutionSandbox,
        workspace_resolver: WorkspaceResolver,
        seal_verifier: ExecutionSealVerifier = DenyUnverifiedExecution(),
        policy: RunnerPolicy = RunnerPolicy(),
        process_launcher: Optional[ProcessLauncher] = None,
        gpu_reader: Optional[GPUMetricsReader] = None,
        telemetry_collector_factory: TelemetryCollectorFactory = TelemetryCollector,
        submission_artifact_resolver: Optional[SubmissionArtifactResolver] = None,
        model_factory: ModelFactory = default_model_factory,
    ) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)
        if not isinstance(artifacts, ExecutionArtifactStore):
            raise TypeError("a compatible Person 2 artifact store is required")
        self.artifacts = artifacts
        self.commands = commands
        self.sandbox = sandbox
        self.workspace_resolver = workspace_resolver
        self.seal_verifier = seal_verifier
        self.policy = policy
        self.process_launcher = process_launcher or ProcessLauncher()
        self.gpu_reader = gpu_reader or NullGPUMetricsReader()
        self.telemetry_collector_factory = telemetry_collector_factory
        self.submission_artifact_resolver = submission_artifact_resolver
        self.model_factory = model_factory

    async def run(self, request: Any, observer: HealthObserver) -> Any:
        """Execute without blocking the orchestrator's event loop."""

        return await asyncio.to_thread(self.run_sync, request, observer)

    def run_sync(self, request: Any, observer: HealthObserver) -> Any:
        identity = _validate_request(request, self.policy)
        workspace = _canonical_workspace(
            self.workspace_resolver(identity.run_id, identity.experiment_id)
        )

        manager = RunArtifactManager(
            self.artifacts,
            identity.run_id,
            identity.experiment_id,
            identity.attempt,
        )
        tracker = ResourceTracker(
            gpu_count=0,
            model_factory=self.model_factory,
        )
        try:
            lease = self.seal_verifier.acquire_lease(
                request,
                workspace,
                timeout_seconds=self.policy.worktree_lease_timeout_seconds,
            )
            with lease:
                return self._run_locked(
                    request,
                    observer,
                    identity,
                    workspace,
                    manager,
                )
        except ExecutionAuthorizationError as error:
            return self._result_without_process(
                request,
                manager,
                tracker,
                outcome="contract_error",
                error_class=type(error).__name__,
                summary=str(error),
            )

    def _run_locked(
        self,
        request: Any,
        observer: HealthObserver,
        identity: "_RequestIdentity",
        workspace: Path,
        manager: RunArtifactManager,
    ) -> Any:
        limits = _resource_limits(request, self.policy)
        tracker = ResourceTracker(
            gpu_count=0,
            model_factory=self.model_factory,
        )

        exit_code: Optional[int]
        try:
            initial_seal = self.seal_verifier.verify(request, workspace)
            submission_artifact = self._submission_artifact(
                request, identity.command_id
            )
            submission_prediction = (
                submission_artifact[0] if submission_artifact is not None else None
            )
            context = CommandContext(
                repository_root=self.repository_root,
                worktree=workspace,
                artifact_dir=manager.output_directory,
                run_id=identity.run_id,
                experiment_id=identity.experiment_id,
                attempt=identity.attempt,
                fidelity=identity.fidelity,
                seed=identity.seed,
                submission_prediction_path=submission_prediction,
            )
            resolved = self.commands.resolve(
                identity.command_id,
                context,
                network_enabled=identity.network_enabled,
            )
            tracker = ResourceTracker(
                gpu_count=resolved.gpu_count,
                model_factory=self.model_factory,
            )
            manager.write_resolved_configuration(
                resolved,
                request_summary=safe_request_summary(request),
                limits_summary=_limits_summary(limits),
            )
            manager.write_environment_identity(resolved)
            _require_fresh_execution_paths(manager, resolved.expected_artifacts)
            configuration = SandboxConfig(
                workspace=workspace,
                artifact_directory=manager.output_directory,
                temporary_directory=manager.temporary_directory,
                network_enabled=identity.network_enabled,
                limits=limits,
                fidelity=identity.fidelity,
                data_manifest_sha256=str(_field(request, "data_manifest_sha256")),
            )
            launch_specification = self.sandbox.prepare(resolved, configuration)
            validate_launch_spec(
                resolved,
                configuration,
                launch_specification,
                allow_trusted_local=self.policy.allow_trusted_local_backend,
            )
        except (ExecutionAuthorizationError, CommandPolicyError, SandboxPolicyError) as error:
            return self._result_without_process(
                request,
                manager,
                tracker,
                outcome="contract_error",
                error_class=type(error).__name__,
                summary=str(error),
            )

        started = time.monotonic()
        try:
            process = self.process_launcher.launch(
                launch_specification, manager.log_path, limits
            )
        except ProcessLaunchError as error:
            return self._result_without_process(
                request,
                manager,
                tracker,
                outcome="infrastructure_error",
                error_class=type(error).__name__,
                summary=str(error),
            )

        try:
            collector = self.telemetry_collector_factory(
                process,
                run_id=identity.run_id,
                experiment_id=identity.experiment_id,
                attempt=identity.attempt,
                disk_path=manager.directory,
                started_monotonic=started,
                deadline_monotonic=started + limits.wall_time_seconds,
                gpu_reader=self.gpu_reader,
                model_factory=self.model_factory,
            )
        except BaseException as error:
            summary = "telemetry collector initialization failed"
            try:
                process.terminate_group()
                process.close_after_termination()
            except ProcessLaunchError as cleanup_error:
                summary = str(cleanup_error)
            return self._result_without_process(
                request,
                manager,
                tracker,
                outcome="infrastructure_error",
                error_class="TELEMETRY_INITIALIZATION_FAILURE",
                summary=summary or type(error).__name__,
            )
        termination: Optional[Tuple[str, str, str]] = None
        telemetry_error: Optional[BaseException] = None

        consecutive_sample_failures = 0
        try:
            with TelemetryJournal(manager.telemetry_path) as journal:
                while True:
                    if not process.is_alive():
                        break
                    if time.monotonic() - started >= limits.wall_time_seconds:
                        termination = (
                            "timeout",
                            "WALL_TIMEOUT",
                            "hard wall-time limit exceeded",
                        )
                        break
                    try:
                        sample = collector.sample()
                    except BaseException:
                        # A short-lived ``docker run --rm`` can disappear after
                        # the liveness check but before ``docker stats``.  That
                        # is normal completion; metrics loss while the launcher
                        # remains alive stays fail-closed, but only after the
                        # bounded consecutive-miss window below, because Docker
                        # Desktop stalls ``docker stats`` for seconds under
                        # host load without the candidate being unhealthy.
                        if not process.is_alive():
                            break
                        if time.monotonic() - started >= limits.wall_time_seconds:
                            termination = (
                                "timeout",
                                "WALL_TIMEOUT",
                                "hard wall-time limit exceeded",
                            )
                            break
                        consecutive_sample_failures += 1
                        if consecutive_sample_failures >= _MAX_CONSECUTIVE_TELEMETRY_MISSES:
                            raise
                        remaining = limits.wall_time_seconds - (
                            time.monotonic() - started
                        )
                        time.sleep(
                            min(
                                self.policy.telemetry_interval_seconds,
                                max(0.001, remaining),
                            )
                        )
                        continue
                    consecutive_sample_failures = 0
                    journal.append(sample)
                    tracker.observe(
                        rss_mb=_field(sample, "rss_mb", None),
                        gpu_memory_mb=_field(sample, "gpu_memory_mb", None),
                        cpu_seconds=getattr(collector, "last_cpu_seconds", None),
                    )
                    if not process.is_alive():
                        break

                    elapsed = time.monotonic() - started
                    if elapsed >= limits.wall_time_seconds:
                        termination = (
                            "timeout",
                            "WALL_TIMEOUT",
                            "hard wall-time limit exceeded",
                        )
                        break

                    observed_limit = observed_limit_pressure(sample, limits)
                    if observed_limit is not None:
                        outcome = (
                            "oom"
                            if observed_limit
                            in {
                                "OBSERVED_CPU_MEMORY_LIMIT",
                                "OBSERVED_GPU_MEMORY_LIMIT",
                            }
                            else "infrastructure_error"
                        )
                        termination = (
                            outcome,
                            observed_limit,
                            _observed_limit_summary(observed_limit),
                        )
                        break

                    try:
                        directive = observer.observe(sample)
                        action = directive_action(directive)
                    except BaseException as error:
                        telemetry_error = error
                        termination = (
                            "infrastructure_error",
                            "HEALTH_OBSERVER_ERROR",
                            "health observer failed",
                        )
                        break
                    if action == "terminate":
                        reason, observer_summary = directive_reason(directive)
                        reason_code = reason or "OBSERVER_TERMINATED"
                        termination = (
                            _observer_outcome(reason_code),
                            reason_code,
                            observer_summary or "health observer requested termination",
                        )
                        break

                    try:
                        outputs_ready = process.runtime_outputs_ready()
                    except ProcessLaunchError:
                        termination = (
                            "infrastructure_error",
                            "RUNTIME_OUTPUT_HANDSHAKE_FAILURE",
                            "container output completion probe failed",
                        )
                        break
                    if outputs_ready:
                        try:
                            process.extract_ready_runtime_outputs()
                        except OutputQuotaExceeded:
                            termination = (
                                "infrastructure_error",
                                "DISK_QUOTA_EXHAUSTED",
                                "allowlisted runtime output exceeded its hard disk quota",
                            )
                            break
                        except DiskSpaceExhausted:
                            termination = (
                                "infrastructure_error",
                                "DISK_SPACE_EXHAUSTED",
                                "runtime output could not be written because storage is full",
                            )
                            break
                        except ProcessLaunchError as error:
                            detail = str(error).strip()
                            summary = "bounded container output extraction failed"
                            if detail:
                                summary = f"{summary}: {detail}"
                            termination = (
                                "infrastructure_error",
                                "RUNTIME_OUTPUT_EXTRACTION_FAILURE",
                                summary,
                            )
                            break
                        release_wait = limits.wall_time_seconds - elapsed
                        process.wait(timeout=min(0.1, max(0.001, release_wait)))
                        continue

                    remaining = limits.wall_time_seconds - elapsed
                    time.sleep(
                        min(self.policy.telemetry_interval_seconds, max(0.001, remaining))
                    )
        except BaseException as error:
            telemetry_error = error
            detail = str(error).strip()
            summary = "telemetry collection failed"
            if detail:
                summary = f"{summary}: {type(error).__name__}: {detail}"
            termination = (
                "infrastructure_error",
                "TELEMETRY_FAILURE",
                summary,
            )

        try:
            if termination is not None:
                exit_code = process.terminate_group()
                process.close_after_termination()
            else:
                exit_code = process.finish()
        except ProcessLaunchError as error:
            exit_code = process.return_code
            process.close_after_termination()
            telemetry_error = error
            if isinstance(error, OutputQuotaExceeded):
                reason_code = "DISK_QUOTA_EXHAUSTED"
                summary = "allowlisted runtime output exceeded its hard disk quota"
            elif isinstance(error, DiskSpaceExhausted):
                reason_code = "DISK_SPACE_EXHAUSTED"
                summary = "runtime output could not be written because storage is full"
            else:
                reason_code = "RUNTIME_CLEANUP_FAILURE"
                summary = str(error)
            termination = (
                "infrastructure_error",
                reason_code,
                summary,
            )

        if process.reader_error is not None and termination is None:
            telemetry_error = process.reader_error
            termination = (
                "infrastructure_error",
                "LOG_CAPTURE_FAILURE",
                "runtime log capture failed",
            )

        post_seal_verified = False
        post_seal: Any = None
        try:
            # Re-checking after process-group cleanup detects candidate code that
            # tried to alter the sealed worktree while it was executing.
            post_seal = self.seal_verifier.verify(request, workspace)
            post_seal_verified = True
        except ExecutionAuthorizationError as error:
            termination = (
                "contract_error",
                "WORKSPACE_SEAL_CHANGED",
                str(error),
            )

        try:
            outputs = manager.capture_outputs(resolved.expected_artifacts)
        except BaseException as error:
            outputs = CapturedOutputs(None, None, ("artifact_capture",))
            if termination is None:
                telemetry_error = error
                quota_error = (
                    isinstance(error, OSError)
                    and getattr(error, "errno", None) == errno.ENOSPC
                )
                termination = (
                    "infrastructure_error",
                    (
                        "DISK_SPACE_EXHAUSTED"
                        if quota_error
                        else "ARTIFACT_CAPTURE_FAILURE"
                    ),
                    "artifact capture ran out of disk space"
                    if quota_error
                    else "expected artifact capture failed",
                )

        result_prediction_artifact = outputs.prediction_artifact
        if submission_artifact is not None:
            result_prediction_artifact = submission_artifact[1]

        if (
            post_seal_verified
            and termination is None
            and exit_code == 0
            and not outputs.missing_required_roles
            and outputs.prediction_artifact is not None
        ):
            receipt_sha256 = _receipt_sha256(post_seal) or _receipt_sha256(
                initial_seal
            )
            try:
                if receipt_sha256 is None:
                    raise ExecutionAuthorizationError(
                        "verified receipt hash is required for output sealing"
                    )
                manager.write_execution_seal(
                    request=request,
                    command=resolved,
                    prediction_artifact=outputs.prediction_artifact,
                    receipt_sha256=receipt_sha256,
                )
            except BaseException as error:
                telemetry_error = error
                termination = (
                    "infrastructure_error",
                    "EXECUTION_SEAL_WRITE_FAILURE",
                    "trusted execution seal could not be written",
                )

        log_tail = process.recent_output_tail()
        if termination is None:
            if exit_code is None:
                raise AssertionError("finished process has no exit code")
            outcome, error_class, result_summary = _normalize_exit(
                exit_code,
                log_tail,
                outputs.missing_required_roles,
                process.runtime_state,
            )
        else:
            outcome, error_class, result_summary = termination
        if telemetry_error is not None:
            result_summary = result_summary or type(telemetry_error).__name__

        log_artifact = manager.log_reference()
        telemetry_artifact = manager.telemetry_reference()
        resource_delta = tracker.finish()
        fingerprint = (
            None
            if outcome == "success"
            else error_fingerprint(error_class, log_tail or result_summary or outcome)
        )
        return self.model_factory(
            "RunResult",
            run_id=identity.run_id,
            experiment_id=identity.experiment_id,
            attempt=identity.attempt,
            fidelity=identity.fidelity,
            patch_commit_sha=identity.patch_commit_sha,
            outcome=outcome,
            exit_code=exit_code,
            error_class=None if outcome == "success" else error_class,
            error_fingerprint=fingerprint,
            error_summary=(
                None
                if outcome == "success"
                else _safe_summary(result_summary or error_class or outcome)
            ),
            log_artifact=log_artifact,
            telemetry_artifact=telemetry_artifact,
            checkpoint_artifact=outputs.checkpoint_artifact,
            prediction_artifact=result_prediction_artifact,
            resource_delta=resource_delta,
        )

    def _result_without_process(
        self,
        request: Any,
        manager: RunArtifactManager,
        tracker: ResourceTracker,
        *,
        outcome: str,
        error_class: str,
        summary: str,
    ) -> Any:
        identity = _request_identity(request)
        safe_summary = _safe_summary(summary)
        log_artifact = manager.write_launch_failure(safe_summary)
        if not manager.telemetry_path.exists():
            with TelemetryJournal(manager.telemetry_path):
                pass
        telemetry_artifact = manager.telemetry_reference()
        return self.model_factory(
            "RunResult",
            run_id=identity.run_id,
            experiment_id=identity.experiment_id,
            attempt=identity.attempt,
            fidelity=identity.fidelity,
            patch_commit_sha=identity.patch_commit_sha,
            outcome=outcome,
            exit_code=None,
            error_class=error_class,
            error_fingerprint=error_fingerprint(error_class, safe_summary),
            error_summary=safe_summary,
            log_artifact=log_artifact,
            telemetry_artifact=telemetry_artifact,
            checkpoint_artifact=None,
            prediction_artifact=None,
            resource_delta=tracker.finish(),
        )

    def _submission_artifact(
        self, request: Any, command_id: str
    ) -> Optional[Tuple[Path, Any]]:
        if command_id != "submission_check":
            return None
        resolver = self.submission_artifact_resolver
        if resolver is None:
            raise ExecutionAuthorizationError(
                "submission_check requires a controller-owned prior prediction resolver"
            )
        try:
            artifact_ref = resolver.resolve(request)
            resolved = Path(self.artifacts.verify(artifact_ref))
        except Exception as error:
            raise ExecutionAuthorizationError(
                "submission_check prior prediction failed artifact verification"
            ) from error
        if resolved.is_symlink() or not resolved.is_file():
            raise ExecutionAuthorizationError(
                "submission_check prior prediction is not a regular artifact"
            )
        return resolved, artifact_ref


class FakeExecutionRunner:
    """Deterministic shared-model test double for Person 2 routing tests."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Tuple[Any, HealthObserver]] = []

    async def run(self, request: Any, observer: HealthObserver) -> Any:
        self.calls.append((request, observer))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@dataclass(frozen=True)
class _RequestIdentity:
    run_id: str
    experiment_id: str
    attempt: int
    fidelity: str
    command_id: str
    patch_commit_sha: str
    seed: int
    network_enabled: bool


def _validate_request(request: Any, policy: RunnerPolicy) -> _RequestIdentity:
    identity = _request_identity(request)
    identifier_pattern = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
    for name in ("run_id", "experiment_id", "fidelity", "command_id"):
        if not identifier_pattern.fullmatch(str(getattr(identity, name))):
            raise InvalidRunRequest("invalid {0}".format(name))
    if identity.attempt < 1:
        raise InvalidRunRequest("attempt must be at least one")
    if not isinstance(identity.seed, int) or isinstance(identity.seed, bool):
        raise InvalidRunRequest("seed must be an integer")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", identity.patch_commit_sha):
        raise InvalidRunRequest("patch_commit_sha must be a lowercase Git hash")
    receipt = str(_field(request, "patch_receipt_id"))
    if not receipt or len(receipt) > 256:
        raise InvalidRunRequest("patch_receipt_id is required")
    manifest_hash = str(_field(request, "data_manifest_sha256"))
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise InvalidRunRequest("data_manifest_sha256 must be a sha256 hash")
    timeout = float(_field(request, "timeout_seconds"))
    memory = int(_field(request, "memory_limit_mb"))
    gpu_memory = int(_field(request, "gpu_memory_limit_mb"))
    if not math.isfinite(timeout) or timeout <= 0 or timeout > policy.max_timeout_seconds:
        raise InvalidRunRequest("timeout is outside the approved range")
    if memory <= 0 or memory > policy.max_memory_limit_mb:
        raise InvalidRunRequest("memory limit is outside the approved range")
    if gpu_memory < 0 or gpu_memory > policy.max_gpu_memory_limit_mb:
        raise InvalidRunRequest("GPU memory limit is outside the approved range")
    network = _field(request, "network_enabled")
    if not isinstance(network, bool):
        raise InvalidRunRequest("network_enabled must be boolean")
    return identity


def _request_identity(request: Any) -> _RequestIdentity:
    return _RequestIdentity(
        run_id=str(_field(request, "run_id")),
        experiment_id=str(_field(request, "experiment_id")),
        attempt=int(_field(request, "attempt")),
        fidelity=str(_enum_value(_field(request, "fidelity"))),
        command_id=str(_enum_value(_field(request, "command_id"))),
        patch_commit_sha=str(_field(request, "patch_commit_sha")),
        seed=int(_field(request, "seed")),
        network_enabled=_field(request, "network_enabled"),
    )


def _resource_limits(request: Any, policy: RunnerPolicy) -> ResourceLimits:
    return ResourceLimits(
        wall_time_seconds=float(_field(request, "timeout_seconds")),
        memory_limit_mb=int(_field(request, "memory_limit_mb")),
        gpu_memory_limit_mb=int(_field(request, "gpu_memory_limit_mb")),
        disk_free_floor_mb=policy.disk_free_floor_mb,
        max_log_bytes=policy.max_log_bytes,
        max_open_files=policy.max_open_files,
        max_processes=policy.max_processes,
        termination_grace_seconds=policy.termination_grace_seconds,
    )


def _canonical_workspace(value: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise InvalidRunRequest("resolved workspace must be an absolute path")
    current = candidate
    while True:
        if current.is_symlink():
            raise InvalidRunRequest("resolved workspace contains a symbolic link")
        if current == current.parent:
            break
        current = current.parent
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise InvalidRunRequest("resolved workspace does not exist") from error
    if candidate != resolved or not resolved.is_dir():
        raise InvalidRunRequest("resolved workspace must be a canonical directory")
    return resolved


def _limits_summary(limits: ResourceLimits) -> Mapping[str, object]:
    return {
        "wall_time_seconds": limits.wall_time_seconds,
        "memory_limit_mb": limits.memory_limit_mb,
        "gpu_memory_limit_mb": limits.gpu_memory_limit_mb,
        "disk_free_floor_mb": limits.disk_free_floor_mb,
        "max_log_bytes": limits.max_log_bytes,
        "max_open_files": limits.max_open_files,
        "max_processes": limits.max_processes,
        "termination_grace_seconds": limits.termination_grace_seconds,
    }


def _require_fresh_execution_paths(manager: RunArtifactManager, expected: Tuple[Any, ...]) -> None:
    for path in (
        manager.log_path,
        manager.telemetry_path,
        manager.execution_seal_path,
    ):
        if path.exists() or path.is_symlink():
            raise SandboxPolicyError("execution artifact path already exists")
    for item in expected:
        candidate = manager.output_directory / item.relative_path
        output_root = manager.output_directory.resolve(strict=True)
        try:
            candidate.resolve(strict=False).relative_to(output_root)
        except ValueError as error:
            raise SandboxPolicyError("expected output escapes output root") from error
        current = candidate
        while current != output_root:
            if current.is_symlink():
                raise SandboxPolicyError("expected output path contains a symlink")
            current = current.parent
        if candidate.exists() or candidate.is_symlink():
            raise SandboxPolicyError("expected output path already exists")
        candidate.parent.mkdir(parents=True, exist_ok=True)


def _normalize_exit(
    exit_code: int,
    log_tail: str,
    missing_roles: Tuple[str, ...],
    runtime_state: Mapping[str, Any],
) -> Tuple[str, Optional[str], Optional[str]]:
    runtime_exit = runtime_state.get("ExitCode")
    if isinstance(runtime_exit, int) and runtime_exit != exit_code:
        return (
            "infrastructure_error",
            "RUNTIME_EXIT_MISMATCH",
            "container state did not match the launcher exit code",
        )
    if runtime_state.get("OOMKilled") is True:
        return "oom", "OUT_OF_MEMORY", "container runtime reported an OOM kill"
    if exit_code == 0 and missing_roles:
        return (
            "interface_error",
            "MISSING_EXPECTED_ARTIFACT",
            "missing required output roles: {0}".format(", ".join(missing_roles)),
        )
    if exit_code == 0:
        return "success", None, None
    lower = log_tail.lower()
    if any(
        marker in lower
        for marker in (
            "no space left",
            "enospc",
            "disk quota exceeded",
            "storage is full",
        )
    ):
        return (
            "infrastructure_error",
            "DISK_QUOTA_EXHAUSTED",
            _tail_summary(log_tail, "runtime storage quota was exhausted"),
        )
    if any(
        marker in lower
        for marker in (
            "out of memory",
            "memoryerror",
            "cuda error: out of memory",
            "cuda out of memory",
            "cannot allocate memory",
        )
    ):
        return "oom", "OUT_OF_MEMORY", _tail_summary(log_tail, "process ran out of memory")
    if any(marker in lower for marker in ("nan", "inf loss", "non-finite")):
        return (
            "numerical_error",
            "NUMERICAL_ERROR",
            _tail_summary(log_tail, "non-finite numerical state"),
        )
    if any(
        marker in lower
        for marker in (
            "missing interface",
            "attributeerror",
            "modulenotfounderror",
            "importerror",
        )
    ):
        return (
            "interface_error",
            "INTERFACE_ERROR",
            _tail_summary(log_tail, "required interface was unavailable"),
        )
    if exit_code in {-getattr(signal, "SIGXCPU", signal.SIGTERM)}:
        return "timeout", "CPU_TIMEOUT", "CPU-time limit exceeded"
    # Windows does not expose SIGKILL; a forcibly terminated child is still
    # represented as a timeout/termination outcome by the runner.
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    if exit_code in {-sigkill, -signal.SIGTERM}:
        return (
            "infrastructure_error",
            "UNEXPECTED_PROCESS_SIGNAL",
            "process exited after an unexpected signal",
        )
    return "code_error", "CANDIDATE_CODE_ERROR", _tail_summary(
        log_tail, "candidate command exited nonzero"
    )


def _observer_outcome(reason_code: str) -> str:
    upper = reason_code.upper()
    if "HANG" in upper or "HEARTBEAT" in upper:
        return "hang"
    if any(marker in upper for marker in ("NUMERICAL", "LOSS", "GRADIENT", "NONFINITE")):
        return "numerical_error"
    if "OOM" in upper or "MEMORY_LIMIT" in upper:
        return "oom"
    if "DISK" in upper or "QUOTA" in upper or "STORAGE" in upper:
        return "infrastructure_error"
    if "CONTRACT" in upper or "POLICY" in upper:
        return "contract_error"
    if "CANCEL" in upper or "EMERGENCY" in upper:
        return "cancelled"
    return "cancelled"


def _observed_limit_summary(reason_code: str) -> str:
    return {
        "OBSERVED_CPU_MEMORY_LIMIT": "telemetry observed memory above the hard backend limit",
        "OBSERVED_GPU_MEMORY_LIMIT": "telemetry observed GPU memory above the hard backend limit",
        "OBSERVED_DISK_FREE_FLOOR": "artifact filesystem reached its safe free-space floor",
    }.get(reason_code, "telemetry observed an execution limit crossing")


def _receipt_sha256(seal: Any) -> Optional[str]:
    if seal is None:
        return None
    value = _field(seal, "receipt_sha256", None)
    if value is None:
        return None
    candidate = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", candidate) is None:
        return None
    return candidate


def error_fingerprint(error_class: Optional[str], evidence: str) -> str:
    normalized = _normalized_error_evidence(evidence)
    payload = "{0}\n{1}".format(error_class or "unknown", normalized)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_error_evidence(evidence: str) -> str:
    selected = []
    for line in redact_runtime_output(evidence).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            "traceback" in stripped.lower()
            or stripped.startswith("File ")
            or re.search(r"(?:Error|Exception|Failure|OOM|timeout)", stripped, re.I)
        ):
            selected.append(stripped)
    if not selected:
        selected = [line.strip() for line in evidence.splitlines() if line.strip()][-3:]
    normalized = "\n".join(selected[-12:])
    normalized = re.sub(r'File "[^"]+"', 'File "<path>"', normalized)
    normalized = re.sub(r", line \d+", ", line <n>", normalized)
    normalized = re.sub(r"0x[0-9a-fA-F]+", "0x<addr>", normalized)
    normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ][^\s]+", "<timestamp>", normalized)
    return normalized[:4096]


def _safe_summary(summary: str, limit: int = 512) -> str:
    return redact_runtime_output(str(summary)).replace("\x00", "")[:limit]


# Traceback frames pointing into the candidate's own source. The exception
# line alone names the error but never where it came from, and the coding
# worker has no shell to reproduce the failure, so a bare "ValueError: left
# keys must be sorted" leaves it editing blind until its step budget is gone.
_CANDIDATE_FRAME_RE = re.compile(
    r'^\s*File "(?P<path>[^"]*solution[/\\][^"]+)", line (?P<line>\d+), in (?P<symbol>\S+)'
)
_TAIL_SUMMARY_LIMIT = 1024


def _tail_summary(log_tail: str, fallback: str) -> str:
    lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
    if not lines:
        return _safe_summary(fallback)
    # Keep the deepest candidate frames so the fault is locatable, then the
    # final exception line. Only frames inside the candidate's own files are
    # included: interpreter and third-party frames add length without telling
    # the worker which of its own lines to change.
    frames = [
        "%s:%s in %s" % (
            match.group("path").replace("\\", "/").rsplit("/solution/", 1)[-1],
            match.group("line"),
            match.group("symbol"),
        )
        for line in lines
        for match in (_CANDIDATE_FRAME_RE.match(line),)
        if match
    ]
    if not frames:
        return _safe_summary(lines[-1], _TAIL_SUMMARY_LIMIT)
    return _safe_summary(
        "%s (candidate frames: %s)" % (lines[-1], " <- ".join(frames[-4:])),
        _TAIL_SUMMARY_LIMIT,
    )


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _field(value: Any, name: str, default: Any = ...) -> Any:
    if isinstance(value, Mapping):
        if default is ...:
            return value[name]
        return value.get(name, default)
    if default is ...:
        return getattr(value, name)
    return getattr(value, name, default)
