"""Fail-closed execution backends and launch specifications.

Production execution uses :class:`DockerSandbox`. It constructs one exact
Docker-compatible argv vector and relies on kernel/runtime controls for mount,
network, memory, process, and CPU isolation. The local process backend is
deliberately named and double-gated as test-only; POSIX resource limits are a
useful safety belt, but are not represented as containment guarantees.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

from tacorank.execution.commands import ResolvedCommand


class SandboxPolicyError(RuntimeError):
    """Raised before launch when sandbox policy cannot be satisfied."""


@dataclass(frozen=True)
class ResourceLimits:
    wall_time_seconds: float
    memory_limit_mb: int
    gpu_memory_limit_mb: int
    disk_free_floor_mb: int = 128
    max_log_bytes: int = 16 * 1024 * 1024
    max_open_files: int = 256
    max_processes: int = 128
    termination_grace_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.wall_time_seconds <= 0:
            raise SandboxPolicyError("wall timeout must be positive")
        if self.memory_limit_mb <= 0:
            raise SandboxPolicyError("memory limit must be positive")
        if self.gpu_memory_limit_mb < 0:
            raise SandboxPolicyError("GPU memory limit must be non-negative")
        if self.disk_free_floor_mb < 0:
            raise SandboxPolicyError("disk floor must be non-negative")
        if self.max_log_bytes <= 0:
            raise SandboxPolicyError("maximum log size must be positive")
        if self.max_open_files < 16:
            raise SandboxPolicyError("open-file limit is too small")
        if self.max_processes < 1:
            raise SandboxPolicyError("process limit must be positive")
        if self.termination_grace_seconds < 0:
            raise SandboxPolicyError("termination grace must be non-negative")


@dataclass(frozen=True)
class SandboxPolicy:
    """Controller-owned roots and the only permitted network posture."""

    allowed_workspace_roots: Tuple[Path, ...]
    allowed_artifact_roots: Tuple[Path, ...]
    allowed_read_only_roots: Tuple[Path, ...] = ()
    inherit_environment: Tuple[str, ...] = (
        "PATH",
        "LANG",
        "LC_ALL",
        "TZ",
        "SYSTEMROOT",
    )
    allow_network: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_workspace_roots:
            raise SandboxPolicyError("at least one workspace root is required")
        if not self.allowed_artifact_roots:
            raise SandboxPolicyError("at least one artifact root is required")
        for variable in self.inherit_environment:
            if _credential_shaped(variable):
                raise SandboxPolicyError(
                    "credential-shaped environment variables cannot be inherited"
                )


@dataclass(frozen=True)
class SandboxConfig:
    workspace: Path
    artifact_directory: Path
    temporary_directory: Path
    network_enabled: bool
    limits: ResourceLimits
    fidelity: str = ""
    data_manifest_sha256: str = ""


@dataclass(frozen=True)
class OutputQuotaProof:
    """Controller-verified hard bound for the exact writable output mount."""

    artifact_directory: Path
    enforced_max_bytes: int
    mechanism: str

    def __post_init__(self) -> None:
        if not Path(self.artifact_directory).is_absolute():
            raise SandboxPolicyError("quota proof path must be absolute")
        if (
            isinstance(self.enforced_max_bytes, bool)
            or not isinstance(self.enforced_max_bytes, int)
            or self.enforced_max_bytes < 1
        ):
            raise SandboxPolicyError("quota proof limit must be positive")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.mechanism):
            raise SandboxPolicyError("quota proof mechanism is invalid")


@dataclass(frozen=True)
class ImageEnvironmentProof:
    """Exact pinned-image environment inspected before container launch."""

    image: str
    environment_sha256: str
    environment_keys: Tuple[str, ...]


class OutputQuotaVerifier(Protocol):
    """Deployment capability that proves a candidate-writable hard disk cap."""

    production_capable: bool

    def verify(
        self, artifact_directory: Path, configured_max_bytes: int
    ) -> OutputQuotaProof:
        ...


class DedicatedFilesystemQuotaVerifier:
    """Prove the output directory is a capacity-bounded mount point.

    Deployments can mount a dedicated filesystem of the approved size at each
    attempt's ``outputs/`` directory.  Filesystem capacity is then a kernel-
    enforced upper bound even if candidate code ignores cooperative limits.
    Ordinary directories and oversized/shared filesystem roots fail closed.
    """

    production_capable = True

    def verify(
        self, artifact_directory: Path, configured_max_bytes: int
    ) -> OutputQuotaProof:
        path = Path(artifact_directory)
        _reject_symlink_components(path, "quota-backed output directory")
        resolved = path.resolve(strict=True)
        if resolved != path or not resolved.is_dir():
            raise SandboxPolicyError(
                "quota-backed output directory must be canonical"
            )
        if not os.path.ismount(str(resolved)):
            raise SandboxPolicyError(
                "hard output disk quota requires a dedicated filesystem mount"
            )
        try:
            filesystem = os.statvfs(str(resolved))
        except OSError as error:
            raise SandboxPolicyError(
                "hard output disk quota capacity cannot be inspected"
            ) from error
        fragment_size = filesystem.f_frsize or filesystem.f_bsize
        capacity_bytes = int(fragment_size) * int(filesystem.f_blocks)
        if capacity_bytes < 1 or capacity_bytes > configured_max_bytes:
            raise SandboxPolicyError(
                "output filesystem capacity exceeds the reviewed hard quota"
            )
        return OutputQuotaProof(
            artifact_directory=resolved,
            enforced_max_bytes=capacity_bytes,
            mechanism="dedicated_filesystem_capacity",
        )


@dataclass(frozen=True)
class ContainerReadOnlyMount:
    """A controller-selected contract/data mount exposed read-only."""

    source: Path
    target: str
    purpose: str = "candidate_data"

    def __post_init__(self) -> None:
        source = Path(self.source)
        if not source.is_absolute():
            raise SandboxPolicyError("container mount source must be absolute")
        _validate_container_path(self.target, "container mount target")
        if "," in str(source) or "," in self.target:
            raise SandboxPolicyError("container mount paths cannot contain commas")
        if self.purpose not in {
            "contract",
            "candidate_data",
            "hidden_inference_data",
            "evaluator_labels",
            "verified_prediction",
        }:
            raise SandboxPolicyError("unsupported container mount purpose")
        target = PurePosixPath(self.target)
        for reserved in (
            PurePosixPath("/workspace"),
            PurePosixPath("/artifacts"),
            PurePosixPath("/tmp"),
        ):
            if target == reserved or reserved in target.parents:
                raise SandboxPolicyError(
                    "container mount target overlaps a reserved path"
                )
        expected_root = (
            PurePosixPath("/contracts")
            if self.purpose == "contract"
            else PurePosixPath("/inputs")
        )
        if target != expected_root and expected_root not in target.parents:
            raise SandboxPolicyError(
                "container mount target does not match its declared purpose"
            )


@dataclass(frozen=True)
class ContainerMountPolicy:
    """Exact data view approved for one command/fidelity/manifest identity."""

    command_id: str
    fidelity: str
    data_manifest_sha256: str
    mounts: Tuple[ContainerReadOnlyMount, ...] = ()

    def __post_init__(self) -> None:
        identifier = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
        if not identifier.fullmatch(self.command_id):
            raise SandboxPolicyError("invalid mount-policy command_id")
        if not identifier.fullmatch(self.fidelity):
            raise SandboxPolicyError("invalid mount-policy fidelity")
        if not re.fullmatch(r"[0-9a-f]{64}", self.data_manifest_sha256):
            raise SandboxPolicyError("mount-policy manifest must be a sha256 hash")


@dataclass(frozen=True)
class IsolationGuarantees:
    """Properties actually enforced by the selected launch backend."""

    filesystem_containment: bool
    network_containment: bool
    memory_limit: bool
    process_limit: bool
    cpu_limit: bool
    gpu_memory_limit: bool
    trusted_local_only: bool = False
    disk_limit: bool = False


@dataclass(frozen=True)
class RuntimeCleanupSpec:
    """Exact shell-free commands used to remove and verify a runtime object."""

    terminate_argv: Tuple[str, ...]
    inspect_argv: Tuple[str, ...]
    healthcheck_argv: Tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    state_argv: Optional[Tuple[str, ...]] = None
    output_extraction: Optional["RuntimeOutputExtractionSpec"] = None
    completion_argv: Optional[Tuple[str, ...]] = None
    release_argv: Optional[Tuple[str, ...]] = None
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class RuntimeOutputExtractionSpec:
    """Stream a live container's bounded tmpfs outputs into trusted storage."""

    argv: Tuple[str, ...]
    destination: Path
    allowed_relative_paths: Tuple[str, ...]
    max_bytes: int
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class RuntimeMetricsSpec:
    """Exact runtime command for candidate-container CPU/RSS telemetry."""

    argv: Tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    timeout_seconds: float = 2.0


@dataclass(frozen=True)
class LaunchSpec:
    argv: Tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    preexec_fn: Optional[Callable[[], None]]
    start_new_session: bool
    guarantees: IsolationGuarantees
    runtime_cleanup: Optional[RuntimeCleanupSpec] = None
    runtime_metrics: Optional[RuntimeMetricsSpec] = None
    output_quota: Optional[OutputQuotaProof] = None
    image_environment: Optional[ImageEnvironmentProof] = None


class ExecutionSandbox(Protocol):
    """Backend contract consumed by :class:`ExecutionRunner`."""

    def prepare(
        self, command: ResolvedCommand, configuration: SandboxConfig
    ) -> LaunchSpec:
        ...


class TrustedLocalProcessSandbox:
    """Unsafe local backend for deterministic unit tests only.

    This backend does not isolate the filesystem or network, and its portable
    rlimit wrapper is not a hard memory/GPU boundary. Construction therefore
    requires an explicit acknowledgement, and the runner independently rejects
    it unless ``RunnerPolicy.allow_trusted_local_backend`` is enabled.
    """

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        allow_unsafe_for_tests: bool = False,
    ) -> None:
        if not allow_unsafe_for_tests:
            raise SandboxPolicyError(
                "trusted local backend is test-only; explicit acknowledgement required"
            )
        self.policy = policy

    def prepare(
        self, command: ResolvedCommand, configuration: SandboxConfig
    ) -> LaunchSpec:
        workspace, artifact_directory, temporary_directory, cwd = _validated_paths(
            self.policy, command, configuration
        )
        del workspace, artifact_directory
        _validate_network_policy(self.policy, command, configuration)
        executable = Path(command.argv[0]).resolve(strict=True)
        if not executable.is_file() or not os.access(str(executable), os.X_OK):
            raise SandboxPolicyError("reviewed executable is unavailable")

        environment = _minimal_host_environment(
            self.policy, command.environment, temporary_directory
        )
        argv = (str(executable),) + tuple(command.argv[1:])
        if os.name == "posix":
            argv = _rlimit_wrapper(argv, configuration.limits)
        return LaunchSpec(
            argv=argv,
            cwd=cwd,
            environment=environment,
            preexec_fn=None,
            start_new_session=True,
            guarantees=IsolationGuarantees(
                filesystem_containment=False,
                network_containment=False,
                memory_limit=False,
                process_limit=False,
                cpu_limit=False,
                gpu_memory_limit=False,
                trusted_local_only=True,
            ),
        )


class DockerSandbox:
    """Build a production Docker launch with hard CPU/RAM/PID isolation.

    The image must be pinned by digest and already present (``--pull never``).
    Candidate code is mounted at ``/workspace`` read-only, while only the
    attempt's output subtree is writable at ``/artifacts``. Docker does not
    provide a portable per-container GPU-memory limit, so GPU commands are
    rejected before launch until a concrete hard GPU backend is added.
    """

    _IMAGE = re.compile(r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
    _NETWORK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    _USER = re.compile(r"^[0-9]+:[0-9]+$")

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        image: str,
        docker_executable: Optional[Path] = None,
        cpu_count: float = 1.0,
        tmpfs_size_mb: int = 256,
        container_user: Optional[str] = None,
        network_name: Optional[str] = None,
        mount_policies: Tuple[ContainerMountPolicy, ...] = (),
        output_quota_max_bytes: Optional[int] = None,
        output_quota_verifier: Optional[OutputQuotaVerifier] = None,
        image_environment_sha256: Optional[str] = None,
        docker_host: Optional[str] = None,
    ) -> None:
        if not self._IMAGE.fullmatch(image):
            raise SandboxPolicyError("container image must be pinned by sha256 digest")
        if not math.isfinite(cpu_count) or cpu_count <= 0:
            raise SandboxPolicyError("container CPU count must be positive")
        if tmpfs_size_mb <= 0:
            raise SandboxPolicyError("container tmpfs size must be positive")
        get_uid = getattr(os, "getuid", lambda: 65534)
        get_gid = getattr(os, "getgid", lambda: 65534)
        selected_user = container_user or "{0}:{1}".format(get_uid(), get_gid())
        if not self._USER.fullmatch(selected_user):
            raise SandboxPolicyError("container user must be a numeric uid:gid")
        if network_name is not None and not self._NETWORK.fullmatch(network_name):
            raise SandboxPolicyError("invalid Docker network name")
        if network_name is not None and not policy.allow_network:
            raise SandboxPolicyError(
                "Docker network configured while network is forbidden"
            )
        if output_quota_max_bytes is None and output_quota_verifier is not None:
            raise SandboxPolicyError(
                "hard output quota verifier requires a configured limit"
            )
        if output_quota_max_bytes is not None and (
            isinstance(output_quota_max_bytes, bool)
            or not isinstance(output_quota_max_bytes, int)
            or output_quota_max_bytes < 1
        ):
            raise SandboxPolicyError("hard output quota limit must be positive")
        if image_environment_sha256 is not None and re.fullmatch(
            r"[0-9a-f]{64}", image_environment_sha256
        ) is None:
            raise SandboxPolicyError(
                "reviewed image environment must be a sha256 hash"
            )

        self.policy = policy
        self.image = image
        self.docker_executable = (
            Path(docker_executable)
            if docker_executable is not None
            else Path(shutil.which("docker") or "/nonexistent/tacorank-docker")
        )
        self.cpu_count = cpu_count
        self.tmpfs_size_mb = tmpfs_size_mb
        self.container_user = selected_user
        self.network_name = network_name
        self.mount_policies = tuple(mount_policies)
        self.output_quota_max_bytes = output_quota_max_bytes
        self.output_quota_verifier = output_quota_verifier
        self.image_environment_sha256 = image_environment_sha256
        self.docker_host = _validated_docker_host(docker_host)
        identities = [
            (item.command_id, item.fidelity, item.data_manifest_sha256)
            for item in self.mount_policies
        ]
        if len(identities) != len(set(identities)):
            raise SandboxPolicyError("duplicate container mount policy")

    def preflight(self, working_directory: Path) -> ImageEnvironmentProof:
        """Verify the Docker daemon and exact image without launching candidate code."""

        directory = Path(working_directory).resolve(strict=True)
        if not directory.is_dir():
            raise SandboxPolicyError("Docker preflight directory is not a directory")
        docker = self._resolved_docker_executable()
        host_environment = dict(
            _minimal_host_environment(self.policy, {}, directory)
        )
        if self.docker_host is not None:
            host_environment["DOCKER_HOST"] = self.docker_host
        try:
            health = subprocess.run(
                [str(docker), "info", "--format", "{{.ServerVersion}}"],
                cwd=str(directory),
                env=dict(host_environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                shell=False,
                close_fds=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxPolicyError("Docker daemon preflight failed") from error
        if health.returncode != 0 or not health.stdout.strip():
            raise SandboxPolicyError("Docker daemon preflight failed")
        proof = self._verify_image_environment(docker, host_environment, directory)
        self._verify_runtime_probe(docker, host_environment, directory)
        return proof

    def prepare(
        self, command: ResolvedCommand, configuration: SandboxConfig
    ) -> LaunchSpec:
        workspace, artifact_directory, temporary_directory, cwd = _validated_paths(
            self.policy, command, configuration
        )
        _validate_network_policy(self.policy, command, configuration)
        output_quota = self._verify_output_quota(artifact_directory)
        docker = self._resolved_docker_executable()
        host_environment = dict(
            _minimal_host_environment(self.policy, {}, temporary_directory)
        )
        if self.docker_host is not None:
            host_environment["DOCKER_HOST"] = self.docker_host
        image_environment = self._verify_image_environment(
            docker, host_environment, artifact_directory
        )
        container_executable = command.container_executable
        if container_executable is None:
            raise SandboxPolicyError(
                "container command executable is not registered for this profile"
            )
        _validate_container_path(
            container_executable, "container command executable"
        )

        if command.gpu_count > 0:
            if configuration.limits.gpu_memory_limit_mb <= 0:
                raise SandboxPolicyError(
                    "GPU commands require a positive GPU memory limit"
                )
            raise SandboxPolicyError(
                "hard per-container GPU memory isolation is unavailable"
            )

        mount_policy = self._select_mount_policy(command, configuration)
        if "," in str(workspace) or "," in str(artifact_directory):
            raise SandboxPolicyError("Docker bind paths cannot contain commas")
        mounts = self._validated_read_only_mounts(
            command,
            configuration,
            mount_policy,
            artifact_directory,
        )
        translations = (
            (workspace, "/workspace"),
            (artifact_directory, "/artifacts"),
        ) + tuple((source, target) for source, target in mounts)
        container_cwd = _translate_host_path(cwd, translations)
        container_arguments = tuple(
            _translate_value(argument, translations) for argument in command.argv[1:]
        )
        container_environment = _container_environment(
            command.environment, translations
        )
        name = "tacorank-{0}".format(secrets.token_hex(12))
        memory_bytes = configuration.limits.memory_limit_mb * 1024 * 1024
        network = "none"
        if configuration.network_enabled:
            if self.network_name is None:
                raise SandboxPolicyError(
                    "network-enabled execution requires a reviewed Docker network"
                )
            network = self.network_name

        portable_tmpfs = self.output_quota_verifier is None
        runtime_user = "0:0" if portable_tmpfs else self.container_user
        supervisor_capabilities = (
            ["--cap-add", "SETUID", "--cap-add", "SETGID"]
            if portable_tmpfs
            else []
        )
        argv = [
            str(docker),
            "run",
            "--init",
            "--pull",
            "never",
            "--name",
            name,
            "--network",
            network,
            "--log-driver",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            *supervisor_capabilities,
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(configuration.limits.max_processes),
            "--memory",
            str(memory_bytes),
            "--memory-swap",
            str(memory_bytes),
            "--cpus",
            format(self.cpu_count, "g"),
            "--ulimit",
            "nofile={0}:{0}".format(configuration.limits.max_open_files),
            "--user",
            runtime_user,
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size={0}m".format(self.tmpfs_size_mb),
            "--mount",
            _bind_mount(workspace, "/workspace", read_only=True),
        ]
        if portable_tmpfs:
            argv.extend(
                (
                    "--tmpfs",
                    "/artifacts:rw,nosuid,nodev,noexec,mode=1777,size={0}".format(
                        output_quota.enforced_max_bytes
                    ),
                )
            )
        else:
            argv.extend(
                (
                    "--mount",
                    _bind_mount(artifact_directory, "/artifacts", read_only=False),
                )
            )
        for source, target in mounts:
            argv.extend(("--mount", _bind_mount(source, target, read_only=True)))
        argv.extend(("--workdir", container_cwd))
        for key in sorted(container_environment):
            argv.extend(
                ("--env", "{0}={1}".format(key, container_environment[key]))
            )
        # A tmpfs disappears when a container stops.  The reviewed supervisor
        # runs as root, drops only the candidate child to ``container_user``,
        # and keeps the container alive until the controller has copied the
        # allowlisted outputs.  Dedicated quota filesystems do not need this
        # handshake and retain the direct reviewed entrypoint.
        control_directory = "/tmp/{0}-control".format(name)
        if portable_tmpfs:
            child_uid, child_gid = self.container_user.split(":", 1)
            argv.extend(
                (
                    "--entrypoint",
                    container_executable,
                    self.image,
                    "-m",
                    "tacorank.execution.container_supervisor",
                    "run",
                    "--control-directory",
                    control_directory,
                    "--uid",
                    child_uid,
                    "--gid",
                    child_gid,
                    "--",
                )
            )
            argv.extend(container_arguments)
        else:
            # Force the reviewed executable instead of allowing an image-baked
            # ENTRYPOINT to intercept the symbolic command argv.
            argv.extend(("--entrypoint", container_executable, self.image))
            argv.extend(container_arguments)

        extraction = None
        if portable_tmpfs:
            extraction = RuntimeOutputExtractionSpec(
                argv=(str(docker), "cp", name + ":/artifacts/.", "-"),
                destination=artifact_directory,
                allowed_relative_paths=tuple(
                    sorted(item.relative_path for item in command.expected_artifacts)
                ),
                max_bytes=output_quota.enforced_max_bytes,
            )
        cleanup = RuntimeCleanupSpec(
            terminate_argv=(str(docker), "rm", "--force", name),
            inspect_argv=(
                str(docker),
                "container",
                "ls",
                "--all",
                "--quiet",
                "--filter",
                "name=^/{0}$".format(name),
                "--format",
                "{{.Names}}",
            ),
            healthcheck_argv=(
                str(docker),
                "info",
                "--format",
                "{{.ServerVersion}}",
            ),
            cwd=artifact_directory,
            environment=host_environment,
            state_argv=(
                str(docker),
                "container",
                "inspect",
                "--format",
                "{{json .State}}",
                name,
            ),
            output_extraction=extraction,
            completion_argv=(
                (
                    str(docker),
                    "exec",
                    "--user",
                    "0:0",
                    name,
                    container_executable,
                    "-m",
                    "tacorank.execution.container_supervisor",
                    "probe",
                    "--control-directory",
                    control_directory,
                )
                if portable_tmpfs
                else None
            ),
            release_argv=(
                (
                    str(docker),
                    "exec",
                    "--user",
                    "0:0",
                    name,
                    container_executable,
                    "-m",
                    "tacorank.execution.container_supervisor",
                    "release",
                    "--control-directory",
                    control_directory,
                )
                if portable_tmpfs
                else None
            ),
        )
        metrics = RuntimeMetricsSpec(
            argv=(
                str(docker),
                "stats",
                "--no-stream",
                "--format",
                "{{json .}}",
                name,
            ),
            cwd=artifact_directory,
            environment=host_environment,
        )
        return LaunchSpec(
            argv=tuple(argv),
            cwd=artifact_directory,
            environment=host_environment,
            preexec_fn=None,
            start_new_session=True,
            guarantees=IsolationGuarantees(
                filesystem_containment=True,
                network_containment=True,
                memory_limit=True,
                process_limit=True,
                cpu_limit=True,
                # No GPU device is exposed for CPU commands. GPU commands are
                # rejected above until a real per-container memory backend exists.
                gpu_memory_limit=True,
                disk_limit=True,
            ),
            runtime_cleanup=cleanup,
            runtime_metrics=metrics,
            output_quota=output_quota,
            image_environment=image_environment,
        )

    def _resolved_docker_executable(self) -> Path:
        docker_candidate = self.docker_executable
        if docker_candidate.is_symlink():
            raise SandboxPolicyError("Docker runtime executable cannot be a symlink")
        try:
            docker = docker_candidate.resolve(strict=True)
        except OSError as error:
            raise SandboxPolicyError(
                "Docker runtime executable is unavailable"
            ) from error
        if not docker.is_file() or not os.access(str(docker), os.X_OK):
            raise SandboxPolicyError("Docker runtime executable is unavailable")
        return docker

    def _verify_runtime_probe(
        self,
        docker: Path,
        host_environment: Mapping[str, str],
        working_directory: Path,
    ) -> None:
        quota = self.output_quota_max_bytes
        if quota is None:
            raise SandboxPolicyError(
                "production Docker execution requires a hard output disk quota limit"
            )
        name = "tacorank-preflight-{0}".format(secrets.token_hex(12))
        child_uid, child_gid = self.container_user.split(":", 1)
        command = [
            str(docker),
            "run",
            "--name",
            name,
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "32",
            "--memory",
            str(256 * 1024 * 1024),
            "--memory-swap",
            str(256 * 1024 * 1024),
            "--cpus",
            "1",
            "--user",
            "0:0",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=32m",
            "--tmpfs",
            "/artifacts:rw,nosuid,nodev,noexec,mode=1777,size={0}".format(quota),
            "--entrypoint",
            "/usr/local/bin/python3",
            self.image,
            "-m",
            "tacorank.execution.container_supervisor",
            "self-test",
            "--uid",
            child_uid,
            "--gid",
            child_gid,
        ]
        cleanup_completed: Optional[subprocess.CompletedProcess[str]] = None
        cleanup_error: Optional[BaseException] = None
        try:
            completed = subprocess.run(
                command,
                cwd=str(working_directory),
                env=dict(host_environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                shell=False,
                close_fds=True,
                timeout=30.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxPolicyError("Docker runtime capability probe failed") from error
        finally:
            try:
                cleanup_completed = subprocess.run(
                    [str(docker), "rm", "--force", name],
                    cwd=str(working_directory),
                    env=dict(host_environment),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    shell=False,
                    close_fds=True,
                    timeout=10.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_error = error
        if (
            cleanup_error is not None
            or cleanup_completed is None
            or cleanup_completed.returncode != 0
        ):
            raise SandboxPolicyError("Docker preflight container cleanup failed")
        if completed.returncode != 0:
            raise SandboxPolicyError("Docker runtime capability probe failed")
        try:
            payload = json.loads(completed.stdout)
            capacity = payload["capacity"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise SandboxPolicyError("Docker runtime capability probe is malformed") from error
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 1
            or capacity > quota
        ):
            raise SandboxPolicyError("Docker tmpfs quota probe does not match policy")

    def _verify_image_environment(
        self,
        docker: Path,
        host_environment: Mapping[str, str],
        working_directory: Path,
    ) -> ImageEnvironmentProof:
        expected = self.image_environment_sha256
        if expected is None:
            raise SandboxPolicyError(
                "production Docker execution requires reviewed image environment identity"
            )
        try:
            completed = subprocess.run(
                [
                    str(docker),
                    "image",
                    "inspect",
                    "--format",
                    "{{json .Config.Env}}",
                    self.image,
                ],
                cwd=str(working_directory),
                env=dict(host_environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                shell=False,
                close_fds=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxPolicyError(
                "pinned image environment could not be inspected"
            ) from error
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode != 0 or len(lines) != 1:
            raise SandboxPolicyError("pinned image environment could not be inspected")
        try:
            raw_environment = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise SandboxPolicyError("pinned image environment is malformed") from error
        if raw_environment is None:
            values = []
        elif isinstance(raw_environment, list) and all(
            isinstance(value, str) for value in raw_environment
        ):
            values = raw_environment
        else:
            raise SandboxPolicyError("pinned image environment is malformed")
        if len(values) > 256 or any(
            "=" not in value or "\x00" in value or len(value) > 4096
            for value in values
        ):
            raise SandboxPolicyError("pinned image environment is malformed")
        keys = [value.split("=", 1)[0] for value in values]
        if len(keys) != len(set(keys)) or any(
            not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) for key in keys
        ):
            raise SandboxPolicyError("pinned image environment is malformed")
        if any(_credential_shaped(key) for key in keys):
            raise SandboxPolicyError(
                "credential-shaped variables are forbidden in the pinned image"
            )
        payload = json.dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise SandboxPolicyError("pinned image environment identity mismatch")
        return ImageEnvironmentProof(self.image, actual, tuple(keys))

    def _verify_output_quota(self, artifact_directory: Path) -> OutputQuotaProof:
        verifier = self.output_quota_verifier
        configured_max_bytes = self.output_quota_max_bytes
        if configured_max_bytes is None:
            raise SandboxPolicyError(
                "production Docker execution requires a hard output disk quota limit"
            )
        if verifier is None:
            return OutputQuotaProof(
                artifact_directory=artifact_directory.resolve(strict=True),
                enforced_max_bytes=configured_max_bytes,
                mechanism="container_tmpfs",
            )
        if getattr(verifier, "production_capable", False) is not True:
            raise SandboxPolicyError(
                "output quota verifier is not a production enforcement capability"
            )
        try:
            proof = verifier.verify(artifact_directory, configured_max_bytes)
        except SandboxPolicyError:
            raise
        except (OSError, ValueError, TypeError) as error:
            raise SandboxPolicyError("hard output disk quota verification failed") from error
        expected = artifact_directory.resolve(strict=True)
        if (
            Path(proof.artifact_directory) != expected
            or proof.enforced_max_bytes > configured_max_bytes
        ):
            raise SandboxPolicyError("hard output disk quota proof does not match launch")
        return proof

    def _select_mount_policy(
        self, command: ResolvedCommand, configuration: SandboxConfig
    ) -> ContainerMountPolicy:
        matches = [
            item
            for item in self.mount_policies
            if item.command_id == command.command_id
            and item.fidelity == configuration.fidelity
            and item.data_manifest_sha256 == configuration.data_manifest_sha256
        ]
        if len(matches) != 1:
            raise SandboxPolicyError(
                "exact command/fidelity/data-manifest mount policy is required"
            )
        return matches[0]

    def _validated_read_only_mounts(
        self,
        command: ResolvedCommand,
        configuration: SandboxConfig,
        mount_policy: ContainerMountPolicy,
        artifact_directory: Path,
    ) -> Tuple[Tuple[Path, str], ...]:
        validated = []
        targets: list[PurePosixPath] = []
        for mount in mount_policy.mounts:
            if mount.purpose == "evaluator_labels":
                raise SandboxPolicyError(
                    "evaluator labels cannot be mounted into execution commands"
                )
            if mount.purpose == "hidden_inference_data" and not (
                command.command_id == "candidate_final_infer"
                and configuration.fidelity == "full"
            ):
                raise SandboxPolicyError(
                    "hidden inference data is restricted to candidate_final_infer"
                )
            if mount.purpose == "verified_prediction":
                if command.command_id != "submission_check":
                    raise SandboxPolicyError(
                        "verified prediction mounts are restricted to submission_check"
                    )
                source = _validate_read_only_source(
                    mount.source, self.policy.allowed_artifact_roots
                )
                if not source.is_file():
                    raise SandboxPolicyError(
                        "verified prediction mount must name one regular file"
                    )
            else:
                source = _validate_read_only_source(
                    mount.source, self.policy.allowed_read_only_roots
                )
            if mount.purpose != "verified_prediction" and (
                _paths_overlap(source, artifact_directory) or any(
                    _paths_overlap(source, Path(root).resolve(strict=True))
                    for root in self.policy.allowed_artifact_roots
                )
            ):
                raise SandboxPolicyError(
                    "candidate inputs cannot expose the attempt/artifact tree"
                )
            target_path = PurePosixPath(mount.target)
            if any(
                target_path == existing
                or existing in target_path.parents
                or target_path in existing.parents
                for existing in targets
            ):
                raise SandboxPolicyError("overlapping container mount target")
            targets.append(target_path)
            validated.append((source, mount.target))
        return tuple(validated)


def validate_launch_spec(
    command: ResolvedCommand,
    configuration: SandboxConfig,
    specification: LaunchSpec,
    *,
    allow_trusted_local: bool,
) -> None:
    """Reject a backend that cannot prove all requested hard boundaries."""

    if not specification.argv or not Path(specification.argv[0]).is_absolute():
        raise SandboxPolicyError("launch executable must be an absolute path")
    if any("\x00" in value for value in specification.argv):
        raise SandboxPolicyError("launch argv contains NUL")
    if specification.preexec_fn is not None:
        raise SandboxPolicyError("preexec_fn is forbidden in the threaded runner")
    if not specification.start_new_session:
        raise SandboxPolicyError("launch must create a new process session")

    guarantees = specification.guarantees
    if guarantees.trusted_local_only:
        if not allow_trusted_local:
            raise SandboxPolicyError(
                "trusted local backend is disabled by runner policy"
            )
        return
    required = {
        "filesystem containment": guarantees.filesystem_containment,
        "network containment": guarantees.network_containment,
        "memory limit": guarantees.memory_limit,
        "process limit": guarantees.process_limit,
        "CPU limit": guarantees.cpu_limit,
        "disk limit": guarantees.disk_limit,
    }
    if command.gpu_count > 0 or configuration.limits.gpu_memory_limit_mb > 0:
        required["GPU memory limit"] = guarantees.gpu_memory_limit
    missing = [name for name, enforced in required.items() if not enforced]
    if missing:
        raise SandboxPolicyError(
            "sandbox cannot prove hard isolation: {0}".format(
                ", ".join(sorted(missing))
            )
        )
    proof = specification.output_quota
    if proof is None:
        raise SandboxPolicyError("sandbox did not provide a hard output quota proof")
    expected_output = Path(configuration.artifact_directory).resolve(strict=True)
    if (
        Path(proof.artifact_directory) != expected_output
        or proof.enforced_max_bytes < 1
    ):
        raise SandboxPolicyError("sandbox output quota proof does not match launch")
    if (
        specification.runtime_cleanup is None
        or specification.runtime_cleanup.state_argv is None
    ):
        raise SandboxPolicyError(
            "production sandbox must expose authoritative runtime state and cleanup"
        )
    if specification.runtime_metrics is None:
        raise SandboxPolicyError(
            "production sandbox must expose exact candidate runtime telemetry"
        )
    image_environment = specification.image_environment
    if (
        image_environment is None
        or re.fullmatch(
            r"[0-9a-f]{64}", image_environment.environment_sha256
        )
        is None
    ):
        raise SandboxPolicyError(
            "production sandbox must attest the pinned image environment"
        )


def disk_free_mb(path: Path) -> int:
    return int(shutil.disk_usage(str(path)).free // (1024 * 1024))


def _validated_paths(
    policy: SandboxPolicy,
    command: ResolvedCommand,
    configuration: SandboxConfig,
) -> Tuple[Path, Path, Path, Path]:
    workspace = _validate_directory(
        configuration.workspace, policy.allowed_workspace_roots, "workspace"
    )
    artifact_directory = _validate_directory(
        configuration.artifact_directory,
        policy.allowed_artifact_roots,
        "artifact directory",
    )
    temporary_directory = _validate_directory(
        configuration.temporary_directory,
        (artifact_directory,),
        "temporary directory",
    )
    cwd = Path(command.cwd).resolve(strict=True)
    if cwd not in {workspace, artifact_directory}:
        raise SandboxPolicyError("command working directory is not approved")
    return workspace, artifact_directory, temporary_directory, cwd


def _validate_network_policy(
    policy: SandboxPolicy,
    command: ResolvedCommand,
    configuration: SandboxConfig,
) -> None:
    if configuration.network_enabled != command.network_enabled:
        raise SandboxPolicyError("network policy mismatch")
    if configuration.network_enabled and not policy.allow_network:
        raise SandboxPolicyError("network access is not approved")


def _minimal_host_environment(
    policy: SandboxPolicy,
    command_environment: Mapping[str, str],
    temporary_directory: Path,
) -> Mapping[str, str]:
    environment: Dict[str, str] = {}
    for key in policy.inherit_environment:
        value = os.environ.get(key)
        if value is not None and not _credential_shaped(key):
            environment[key] = value
    for key, value in command_environment.items():
        _validate_environment_value(key, value)
        environment[key] = value
    environment.update(
        {
            "HOME": str(temporary_directory),
            "TMPDIR": str(temporary_directory),
            "TMP": str(temporary_directory),
            "TEMP": str(temporary_directory),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": environment.get("PYTHONHASHSEED", "0"),
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    _remove_proxy_environment(environment)
    return environment


def _container_environment(
    command_environment: Mapping[str, str],
    translations: Tuple[Tuple[Path, str], ...],
) -> Mapping[str, str]:
    environment: Dict[str, str] = {}
    required_roots = {
        "TACORANK_CONTRACT_ROOT": "/contracts",
        "TACORANK_INPUT_ROOT": "/inputs",
        "TACORANK_ARTIFACT_ROOT": "/artifacts",
    }
    for key, value in command_environment.items():
        _validate_environment_value(key, value)
        if key in required_roots:
            source = Path(value)
            if not source.is_absolute():
                raise SandboxPolicyError(
                    "{0} must name a controller-mounted host root".format(key)
                )
            translated = _translate_host_path(source, translations)
            if translated != required_roots[key]:
                raise SandboxPolicyError(
                    "{0} must map to {1}".format(key, required_roots[key])
                )
            environment[key] = translated
        else:
            environment[key] = _translate_value(value, translations)
    environment.update(
        {
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": environment.get("PYTHONHASHSEED", "0"),
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    for proxy in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment[proxy] = ""
    return environment


def _remove_proxy_environment(environment: Dict[str, str]) -> None:
    for proxy in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment.pop(proxy, None)


def _validate_environment_value(key: str, value: str) -> None:
    if _credential_shaped(key):
        raise SandboxPolicyError(
            "credential-shaped command environment is forbidden"
        )
    if "\x00" in key or "\x00" in value:
        raise SandboxPolicyError("environment contains NUL")


def _validate_directory(
    path: Path, approved_roots: Tuple[Path, ...], label: str
) -> Path:
    candidate = Path(path)
    _reject_symlink_components(candidate, label)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise SandboxPolicyError("{0} is not a directory".format(label))
    for approved in approved_roots:
        approved_candidate = Path(approved)
        _reject_symlink_components(approved_candidate, "approved root")
        approved_path = approved_candidate.resolve(strict=True)
        try:
            resolved.relative_to(approved_path)
            return resolved
        except ValueError:
            continue
    raise SandboxPolicyError("{0} is outside approved roots".format(label))


def _validate_read_only_source(
    path: Path, approved_roots: Tuple[Path, ...]
) -> Path:
    if not approved_roots:
        raise SandboxPolicyError(
            "read-only mounts require an approved input root"
        )
    candidate = Path(path)
    _reject_symlink_components(candidate, "read-only mount source")
    resolved = candidate.resolve(strict=True)
    for approved in approved_roots:
        approved_candidate = Path(approved)
        _reject_symlink_components(
            approved_candidate, "approved read-only root"
        )
        approved_path = approved_candidate.resolve(strict=True)
        try:
            resolved.relative_to(approved_path)
            return resolved
        except ValueError:
            continue
    raise SandboxPolicyError(
        "read-only mount source is outside approved roots"
    )


def _bind_mount(source: Path, target: str, *, read_only: bool) -> str:
    value = "type=bind,src={0},dst={1},bind-propagation=rprivate".format(
        source, target
    )
    if read_only:
        value += ",readonly"
    return value


def _translate_host_path(
    value: Path, translations: Tuple[Tuple[Path, str], ...]
) -> str:
    resolved = Path(value).resolve(strict=True)
    for source, target in sorted(
        translations, key=lambda item: len(str(item[0])), reverse=True
    ):
        try:
            relative = resolved.relative_to(source)
        except ValueError:
            continue
        if not relative.parts:
            return target
        return str(PurePosixPath(target).joinpath(*relative.parts))
    raise SandboxPolicyError(
        "command path is not exposed inside the container"
    )


def _translate_value(
    value: str, translations: Tuple[Tuple[Path, str], ...]
) -> str:
    translated = value
    for source, target in sorted(
        translations, key=lambda item: len(str(item[0])), reverse=True
    ):
        pattern = re.compile(
            r"(?<![A-Za-z0-9_.-]){0}(?=$|[/\s'\";,):\]])".format(
                re.escape(str(source))
            )
        )
        translated = pattern.sub(target, translated)
    if "\x00" in translated:
        raise SandboxPolicyError("container command value contains NUL")
    return translated


def _validate_container_path(value: str, label: str) -> None:
    if "\x00" in value or "\\" in value:
        raise SandboxPolicyError(
            "{0} contains an invalid character".format(label)
        )
    path = PurePosixPath(value)
    if path == PurePosixPath("/") or not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise SandboxPolicyError(
            "{0} must be a normalized absolute path".format(label)
        )
    if path.as_posix() != value:
        raise SandboxPolicyError(
            "{0} must be a normalized absolute path".format(label)
        )


def _rlimit_wrapper(
    argv: Tuple[str, ...], limits: ResourceLimits
) -> Tuple[str, ...]:
    wrapper = Path(__file__).with_name("_limit_exec.py").resolve(strict=True)
    python = Path(sys.executable).resolve(strict=True)
    return (
        str(python),
        str(wrapper),
        "--memory-bytes",
        str(limits.memory_limit_mb * 1024 * 1024),
        "--cpu-seconds",
        str(max(1, int(math.ceil(limits.wall_time_seconds)) + 1)),
        "--open-files",
        str(limits.max_open_files),
        "--processes",
        str(limits.max_processes),
        "--",
    ) + argv


def _credential_shaped(key: str) -> bool:
    upper = key.upper()
    return any(
        marker in upper
        for marker in (
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "PASSWD",
            "API_KEY",
            "PRIVATE_KEY",
            "CREDENTIAL",
        )
    )


def _validated_docker_host(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("unix://") or "\x00" in value:
        raise SandboxPolicyError("Docker host must be a local Unix socket")
    path = Path(value[len("unix://") :])
    if not path.is_absolute() or path.is_symlink():
        raise SandboxPolicyError("Docker host must be a canonical local Unix socket")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise SandboxPolicyError("Docker host socket is unavailable") from error
    if resolved != path or not stat.S_ISSOCK(metadata.st_mode):
        raise SandboxPolicyError("Docker host must be a canonical local Unix socket")
    return "unix://" + str(resolved)


def _reject_symlink_components(path: Path, label: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise SandboxPolicyError(
                "{0} contains a symbolic link".format(label)
            )
        if current == current.parent:
            return
        current = current.parent


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False
