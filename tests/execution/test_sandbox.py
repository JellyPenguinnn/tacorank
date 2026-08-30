from __future__ import annotations

import os
import sys
import hashlib
import json
import socket
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from tacorank.execution.commands import (
    CommandContext,
    CommandProfile,
    CommandRegistry,
    ExpectedArtifact,
)
from tacorank.execution.sandbox import (
    ContainerMountPolicy,
    ContainerReadOnlyMount,
    DedicatedFilesystemQuotaVerifier,
    DockerSandbox,
    OutputQuotaProof,
    ResourceLimits,
    SandboxConfig,
    SandboxPolicy,
    SandboxPolicyError,
    TrustedLocalProcessSandbox,
    _probe_failure_detail,
    validate_launch_spec,
)


MANIFEST_SHA = "d" * 64
IMAGE = "registry.invalid/tacorank@sha256:" + "a" * 64
OUTPUT_QUOTA_BYTES = 8 * 1024 * 1024
IMAGE_ENVIRONMENT = ["LANG=C.UTF-8", "PATH=/usr/local/bin:/usr/bin"]
IMAGE_ENVIRONMENT_SHA256 = hashlib.sha256(
    json.dumps(IMAGE_ENVIRONMENT, separators=(",", ":")).encode("utf-8")
).hexdigest()


def test_probe_failure_detail_is_bounded_and_credential_redacted() -> None:
    detail = _probe_failure_detail(
        "Traceback (most recent call last):\n"
        "RuntimeError: api_key=secret-value Bearer abcdefghijklmnop "
        "sk-abcdefghijklmnop\n"
    )

    assert detail == (
        "RuntimeError: api_key=[REDACTED] Bearer [REDACTED] [REDACTED]"
    )
    assert len(_probe_failure_detail("ValueError: " + "x" * 1000)) == 512


def test_probe_failure_detail_prefers_python_exception() -> None:
    detail = _probe_failure_detail(
        "ModuleNotFoundError: No module named 'numpy'\n"
        "tacorank container supervisor: candidate identity self-test failed\n"
    )

    assert detail == "ModuleNotFoundError: No module named 'numpy'"


@pytest.fixture(autouse=True)
def _bounded_test_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the production verifier against deterministic kernel probes."""

    monkeypatch.setattr(os.path, "ismount", lambda path: True)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: SimpleNamespace(f_frsize=4096, f_bsize=4096, f_blocks=1024),
    )

    def inspect_image(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        assert argv[1:5] == ["image", "inspect", "--format", "{{json .Config.Env}}"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(IMAGE_ENVIRONMENT) + "\n",
            stderr="",
        )

    monkeypatch.setattr("tacorank.execution.sandbox.subprocess.run", inspect_image)


def _resolved(
    tmp_path: Path,
    *,
    command_id: str = "candidate_smoke",
    fidelity: str = "smoke",
    network_enabled: bool = False,
    gpu_count: int = 0,
    gpu_memory_limit_mb: int = 0,
):
    worktree = tmp_path / "worktree"
    artifacts = tmp_path / "artifacts" / "run_001" / "exp_0001" / "attempt_1" / "outputs"
    temporary = artifacts / "tmp"
    worktree.mkdir()
    temporary.mkdir(parents=True)
    registry = CommandRegistry(
        (
            CommandProfile(
                command_id,
                str(Path(sys.executable).resolve()),
                ("-c", "print('ok')", "{artifact_dir}/prediction.csv"),
                allowed_fidelities=(fidelity,),
                allow_network=network_enabled,
                gpu_count=gpu_count,
                container_executable="/usr/local/bin/python3",
            ),
        )
    )
    command = registry.resolve(
        command_id,
        CommandContext(
            tmp_path.resolve(),
            worktree.resolve(),
            artifacts.resolve(),
            "run_001",
            "exp_0001",
            1,
            fidelity,
            1,
        ),
        network_enabled=network_enabled,
    )
    config = SandboxConfig(
        workspace=worktree,
        artifact_directory=artifacts,
        temporary_directory=temporary,
        network_enabled=network_enabled,
        limits=ResourceLimits(5, 4096, gpu_memory_limit_mb),
        fidelity=fidelity,
        data_manifest_sha256=MANIFEST_SHA,
    )
    return command, config, worktree, artifacts


def _policy(worktree: Path, artifacts: Path, *read_only_roots: Path, allow_network: bool = False):
    return SandboxPolicy(
        allowed_workspace_roots=(worktree,),
        allowed_artifact_roots=(artifacts,),
        allowed_read_only_roots=tuple(read_only_roots),
        allow_network=allow_network,
    )


def _docker(
    policy: SandboxPolicy,
    command_id: str = "candidate_smoke",
    fidelity: str = "smoke",
    mounts: tuple[ContainerReadOnlyMount, ...] = (),
    network_name: Optional[str] = None,
) -> DockerSandbox:
    return DockerSandbox(
        policy,
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        cpu_count=1.5,
        network_name=network_name,
        mount_policies=(
            ContainerMountPolicy(
                command_id,
                fidelity,
                MANIFEST_SHA,
                mounts,
            ),
        ),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        output_quota_verifier=DedicatedFilesystemQuotaVerifier(),
        image_environment_sha256=IMAGE_ENVIRONMENT_SHA256,
    )


def test_trusted_local_backend_is_explicit_and_never_claims_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    policy = _policy(worktree, artifacts)
    with pytest.raises(SandboxPolicyError, match="test-only"):
        TrustedLocalProcessSandbox(policy)

    monkeypatch.setenv("API_TOKEN", "must-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    sandbox = TrustedLocalProcessSandbox(policy, allow_unsafe_for_tests=True)
    launch = sandbox.prepare(command, config)

    assert "API_TOKEN" not in launch.environment
    assert "HTTPS_PROXY" not in launch.environment
    assert launch.environment["HOME"].endswith("/outputs/tmp")
    assert launch.guarantees.trusted_local_only is True
    assert launch.guarantees.filesystem_containment is False
    assert launch.guarantees.network_containment is False
    with pytest.raises(SandboxPolicyError, match="disabled by runner policy"):
        validate_launch_spec(command, config, launch, allow_trusted_local=False)
    validate_launch_spec(command, config, launch, allow_trusted_local=True)


def test_docker_launch_enforces_hard_cpu_ram_pid_network_and_mount_boundaries(
    tmp_path: Path,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    contract_root = tmp_path / "contracts"
    data_root = tmp_path / "data"
    contract_root.mkdir()
    data_root.mkdir()
    contract = contract_root / "contract.json"
    data = data_root / "candidate.parquet"
    contract.write_text("{}", encoding="utf-8")
    data.write_bytes(b"data")
    mounts = (
        ContainerReadOnlyMount(contract, "/contracts/contract.json", "contract"),
        ContainerReadOnlyMount(data, "/inputs/candidate.parquet", "candidate_data"),
    )
    sandbox = _docker(
        _policy(worktree, artifacts, contract_root, data_root),
        mounts=mounts,
    )

    launch = sandbox.prepare(command, config)
    validate_launch_spec(command, config, launch, allow_trusted_local=False)
    argv = launch.argv

    assert argv[0] == str(Path(sys.executable).resolve())
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--log-driver") + 1] == "none"
    assert argv[argv.index("--entrypoint") + 1] == "/usr/local/bin/python3"
    assert "--read-only" in argv
    assert argv[argv.index("--memory") + 1] == str(4096 * 1024 * 1024)
    assert argv[argv.index("--memory-swap") + 1] == str(4096 * 1024 * 1024)
    assert argv[argv.index("--pids-limit") + 1] == "128"
    assert argv[argv.index("--cpus") + 1] == "1.5"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges:true"
    mount_values = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert any("dst=/workspace" in value and value.endswith(",readonly") for value in mount_values)
    output_mounts = [value for value in mount_values if "dst=/artifacts" in value]
    assert len(output_mounts) == 1
    assert "src={0}".format(artifacts) in output_mounts[0]
    assert "readonly" not in output_mounts[0]
    assert not any("src={0},".format(artifacts.parent) in value for value in mount_values)
    assert all(
        "readonly" in value
        for value in mount_values
        if "dst=/contracts" in value or "dst=/inputs" in value
    )
    assert "/artifacts/prediction.csv" in argv
    assert launch.runtime_cleanup is not None
    assert launch.runtime_cleanup.terminate_argv[-2:] == (
        "--force",
        argv[argv.index("--name") + 1],
    )
    assert launch.runtime_metrics is not None
    assert launch.runtime_metrics.argv[-1] == argv[argv.index("--name") + 1]
    assert launch.runtime_metrics.argv[1:5] == (
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
    )
    assert launch.output_quota == OutputQuotaProof(
        artifact_directory=artifacts,
        enforced_max_bytes=4 * 1024 * 1024,
        mechanism="dedicated_filesystem_capacity",
    )


def test_docker_requires_pinned_image_and_exact_manifest_mount_policy(
    tmp_path: Path,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    policy = _policy(worktree, artifacts)
    with pytest.raises(SandboxPolicyError, match="pinned"):
        DockerSandbox(
            policy,
            image="tacorank:latest",
            docker_executable=Path(sys.executable).resolve(),
        )
    sandbox = DockerSandbox(
        policy,
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        mount_policies=(
            ContainerMountPolicy("candidate_smoke", "smoke", "e" * 64),
        ),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        output_quota_verifier=DedicatedFilesystemQuotaVerifier(),
        image_environment_sha256=IMAGE_ENVIRONMENT_SHA256,
    )
    with pytest.raises(SandboxPolicyError, match="exact command/fidelity/data-manifest"):
        sandbox.prepare(command, config)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_docker_runtime_executable_symlink_is_rejected(tmp_path: Path) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    linked_runtime = tmp_path / "docker"
    linked_runtime.symlink_to(Path(sys.executable).resolve())
    sandbox = DockerSandbox(
        _policy(worktree, artifacts),
        image=IMAGE,
        docker_executable=linked_runtime,
        mount_policies=(
            ContainerMountPolicy("candidate_smoke", "smoke", MANIFEST_SHA),
        ),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        output_quota_verifier=DedicatedFilesystemQuotaVerifier(),
    )
    with pytest.raises(SandboxPolicyError, match="cannot be a symlink"):
        sandbox.prepare(command, config)


def test_network_enabled_requires_both_policy_and_named_docker_network(
    tmp_path: Path,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path, network_enabled=True)
    with pytest.raises(SandboxPolicyError, match="not approved"):
        _docker(_policy(worktree, artifacts)).prepare(command, config)

    policy = _policy(worktree, artifacts, allow_network=True)
    with pytest.raises(SandboxPolicyError, match="reviewed Docker network"):
        _docker(policy).prepare(command, config)
    launch = _docker(policy, network_name="tacorank-egress").prepare(command, config)
    assert launch.argv[launch.argv.index("--network") + 1] == "tacorank-egress"


def test_hidden_and_evaluator_data_views_are_fail_closed(tmp_path: Path) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    hidden = inputs / "hidden.parquet"
    labels = inputs / "labels.parquet"
    hidden.write_bytes(b"hidden")
    labels.write_bytes(b"labels")
    policy = _policy(worktree, artifacts, inputs)

    hidden_sandbox = _docker(
        policy,
        mounts=(
            ContainerReadOnlyMount(hidden, "/inputs/hidden.parquet", "hidden_inference_data"),
        ),
    )
    with pytest.raises(SandboxPolicyError, match="candidate_final_infer"):
        hidden_sandbox.prepare(command, config)

    labels_sandbox = _docker(
        policy,
        mounts=(
            ContainerReadOnlyMount(labels, "/inputs/labels.parquet", "evaluator_labels"),
        ),
    )
    with pytest.raises(SandboxPolicyError, match="evaluator labels"):
        labels_sandbox.prepare(command, config)


def test_pipeline_root_environment_requires_exact_scoped_mounts(
    tmp_path: Path,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    contract_root = tmp_path / "contracts"
    input_root = tmp_path / "candidate-input"
    contract_root.mkdir()
    input_root.mkdir()
    command = replace(
        command,
        environment={
            "TACORANK_CONTRACT_ROOT": str(contract_root),
            "TACORANK_INPUT_ROOT": str(input_root),
            "TACORANK_ARTIFACT_ROOT": str(artifacts),
            "TACORANK_CANDIDATE_ENTRYPOINT": "solution.candidate:run",
        },
    )
    policy = _policy(worktree, artifacts, contract_root, input_root)
    incomplete = _docker(
        policy,
        mounts=(
            ContainerReadOnlyMount(contract_root, "/contracts", "contract"),
        ),
    )
    with pytest.raises(SandboxPolicyError, match="not exposed inside the container"):
        incomplete.prepare(command, config)

    launch = _docker(
        policy,
        mounts=(
            ContainerReadOnlyMount(contract_root, "/contracts", "contract"),
            ContainerReadOnlyMount(input_root, "/inputs", "candidate_data"),
        ),
    ).prepare(command, config)
    environment_arguments = {
        launch.argv[index + 1]
        for index, value in enumerate(launch.argv)
        if value == "--env"
    }
    assert "TACORANK_CONTRACT_ROOT=/contracts" in environment_arguments
    assert "TACORANK_INPUT_ROOT=/inputs" in environment_arguments
    assert "TACORANK_ARTIFACT_ROOT=/artifacts" in environment_arguments


def test_submission_mounts_only_the_verified_prior_prediction_read_only(
    tmp_path: Path,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    artifact_root = tmp_path / "artifacts"
    prior = artifact_root / "run_001" / "exp_0001" / "attempt_1" / "outputs" / "prior.csv"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text("row_id,score\n", encoding="utf-8")
    contract_root = tmp_path / "contracts"
    contract_root.mkdir()
    command = replace(
        command,
        command_id="submission_check",
        argv=(command.argv[0], "-c", "print('ok')", str(prior)),
        environment={
            "TACORANK_CONTRACT_ROOT": str(contract_root),
            "TACORANK_ARTIFACT_ROOT": str(artifacts),
            "TACORANK_VERIFIED_PREDICTION_PATH": str(prior),
            "TACORANK_SUBMISSION_CHECK_ENTRYPOINT": "approved.submission:check",
        },
    )
    config = replace(config, fidelity="full")
    sandbox = _docker(
        SandboxPolicy(
            allowed_workspace_roots=(worktree,),
            allowed_artifact_roots=(artifact_root,),
            allowed_read_only_roots=(contract_root,),
        ),
        command_id="submission_check",
        fidelity="full",
        mounts=(
            ContainerReadOnlyMount(contract_root, "/contracts", "contract"),
            ContainerReadOnlyMount(
                prior,
                "/inputs/submission/predictions.csv",
                "verified_prediction",
            ),
        ),
    )

    launch = sandbox.prepare(command, config)
    mount_values = [
        launch.argv[index + 1]
        for index, value in enumerate(launch.argv)
        if value == "--mount"
    ]
    prior_mount = next(
        value for value in mount_values if "dst=/inputs/submission/predictions.csv" in value
    )
    assert "readonly" in prior_mount
    assert "/inputs/submission/predictions.csv" in launch.argv
    assert any(
        value == "TACORANK_VERIFIED_PREDICTION_PATH=/inputs/submission/predictions.csv"
        for index, value in enumerate(launch.argv)
        if index > 0 and launch.argv[index - 1] == "--env"
    )
    assert not any("src={0},".format(artifacts.parent) in value for value in mount_values)


def test_final_inference_can_receive_exact_hidden_view(tmp_path: Path) -> None:
    command, config, worktree, artifacts = _resolved(
        tmp_path,
        command_id="candidate_final_infer",
        fidelity="full",
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    hidden = inputs / "hidden.parquet"
    hidden.write_bytes(b"hidden")
    mount = ContainerReadOnlyMount(
        hidden,
        "/inputs/hidden.parquet",
        "hidden_inference_data",
    )
    launch = _docker(
        _policy(worktree, artifacts, inputs),
        command_id="candidate_final_infer",
        fidelity="full",
        mounts=(mount,),
    ).prepare(command, config)
    assert any("dst=/inputs/hidden.parquet" in value for value in launch.argv)


def test_gpu_command_is_rejected_before_launch_without_hard_memory_backend(
    tmp_path: Path,
) -> None:
    command, config, worktree, artifacts = _resolved(
        tmp_path,
        gpu_count=1,
        gpu_memory_limit_mb=4096,
    )
    with pytest.raises(SandboxPolicyError, match="hard per-container GPU memory"):
        _docker(_policy(worktree, artifacts)).prepare(command, config)


def test_docker_fails_closed_without_production_output_quota(
    tmp_path: Path,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    policy = _policy(worktree, artifacts)
    sandbox = DockerSandbox(
        policy,
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        mount_policies=(
            ContainerMountPolicy("candidate_smoke", "smoke", MANIFEST_SHA),
        ),
    )
    with pytest.raises(SandboxPolicyError, match="hard output disk quota limit"):
        sandbox.prepare(command, config)

    class TestOnlyAttestation:
        production_capable = False

        @staticmethod
        def verify(path: Path, configured_max_bytes: int) -> OutputQuotaProof:
            return OutputQuotaProof(path, configured_max_bytes, "test_attestation")

    sandbox = DockerSandbox(
        policy,
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        mount_policies=(
            ContainerMountPolicy("candidate_smoke", "smoke", MANIFEST_SHA),
        ),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        output_quota_verifier=TestOnlyAttestation(),
    )
    with pytest.raises(SandboxPolicyError, match="not a production enforcement"):
        sandbox.prepare(command, config)


def test_docker_portable_quota_uses_bounded_tmpfs_and_output_copy(
    tmp_path: Path,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    command = replace(
        command,
        expected_artifacts=(
            ExpectedArtifact(
                role="prediction",
                relative_path="prediction.csv",
                kind="predictions",
                content_type="text/csv",
            ),
        ),
    )
    sandbox = DockerSandbox(
        _policy(worktree, artifacts),
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        mount_policies=(
            ContainerMountPolicy("candidate_smoke", "smoke", MANIFEST_SHA),
        ),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        image_environment_sha256=IMAGE_ENVIRONMENT_SHA256,
    )

    launch = sandbox.prepare(command, config)

    mount_values = [
        launch.argv[index + 1]
        for index, value in enumerate(launch.argv)
        if value == "--mount"
    ]
    tmpfs_values = [
        launch.argv[index + 1]
        for index, value in enumerate(launch.argv)
        if value == "--tmpfs"
    ]
    assert not any("dst=/artifacts" in value for value in mount_values)
    assert any(
        value
        == "/artifacts:rw,nosuid,nodev,noexec,mode=1777,size=%d"
        % OUTPUT_QUOTA_BYTES
        for value in tmpfs_values
    )
    assert launch.output_quota == OutputQuotaProof(
        artifact_directory=artifacts,
        enforced_max_bytes=OUTPUT_QUOTA_BYTES,
        mechanism="container_tmpfs",
    )
    assert launch.runtime_cleanup is not None
    assert launch.argv[launch.argv.index("--user") + 1] == "0:0"
    cap_adds = [
        launch.argv[index + 1]
        for index, value in enumerate(launch.argv)
        if value == "--cap-add"
    ]
    assert cap_adds == ["SETUID", "SETGID"]
    assert "tacorank.execution.container_supervisor" in launch.argv
    extraction = launch.runtime_cleanup.output_extraction
    assert extraction is not None
    assert extraction.argv[1:4] == ("exec", "--user", sandbox.container_user)
    assert "tacorank.execution.container_supervisor" in extraction.argv
    assert "export" in extraction.argv
    assert extraction.argv[-2:] == ("--allowed-output", "prediction.csv")
    assert launch.runtime_cleanup.completion_argv is not None
    assert launch.runtime_cleanup.release_argv is not None
    assert launch.runtime_cleanup.completion_argv[1:4] == (
        "exec",
        "--user",
        "0:0",
    )


def test_docker_preflight_launches_and_cleans_exact_capability_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    del command, config
    calls = []
    environments = []
    socket_root = Path(tempfile.mkdtemp(prefix="tr-docker-"))
    socket_path = socket_root / "docker.sock"
    docker_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    docker_socket.bind(str(socket_path))
    socket_path = socket_path.resolve(strict=True)
    socket_root = socket_path.parent

    def docker_cli(argv: list[str], **kwargs: object):
        calls.append(argv)
        environments.append(kwargs.get("env"))
        if argv[1] == "info":
            return subprocess.CompletedProcess(argv, 0, stdout="29.0\n", stderr="")
        if argv[1] == "image":
            return subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps(IMAGE_ENVIRONMENT) + "\n", stderr=""
            )
        if argv[1] == "run":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps({"capacity": OUTPUT_QUOTA_BYTES}) + "\n",
                stderr="",
            )
        if argv[1] == "rm":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr("tacorank.execution.sandbox.subprocess.run", docker_cli)
    sandbox = DockerSandbox(
        _policy(worktree, artifacts),
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        docker_host="unix://" + str(socket_path),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        image_environment_sha256=IMAGE_ENVIRONMENT_SHA256,
    )

    proof = sandbox.preflight(artifacts)

    assert proof.environment_sha256 == IMAGE_ENVIRONMENT_SHA256
    assert any(argv[1] == "run" and "--read-only" in argv for argv in calls)
    runtime_probe = next(argv for argv in calls if argv[1] == "run")
    assert "tacorank.execution.container_supervisor" in runtime_probe
    assert "self-test" in runtime_probe
    assert runtime_probe[runtime_probe.index("--user") + 1] == "0:0"
    assert any(argv[1:3] == ["rm", "--force"] for argv in calls)
    assert all(
        environment["DOCKER_HOST"] == "unix://" + str(socket_path)
        for environment in environments
    )
    docker_socket.close()
    socket_path.unlink()
    socket_root.rmdir()


def test_docker_preflight_reports_sanitized_runtime_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, artifacts = _resolved(tmp_path)
    socket_root = Path(tempfile.mkdtemp(prefix="tr-docker-"))
    socket_path = socket_root / "docker.sock"
    docker_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    docker_socket.bind(str(socket_path))
    socket_path = socket_path.resolve(strict=True)

    def docker_cli(argv: list[str], **kwargs: object):
        if argv[1] == "info":
            return subprocess.CompletedProcess(argv, 0, stdout="29.0\n", stderr="")
        if argv[1] == "image":
            return subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps(IMAGE_ENVIRONMENT) + "\n", stderr=""
            )
        if argv[1] == "run":
            assert kwargs["stderr"] == subprocess.PIPE
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr=(
                    "ModuleNotFoundError: No module named 'numpy'\n"
                    "tacorank container supervisor: candidate identity self-test failed\n"
                ),
            )
        if argv[1] == "rm":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr("tacorank.execution.sandbox.subprocess.run", docker_cli)
    sandbox = DockerSandbox(
        _policy(Path(tmp_path / "worktree"), artifacts),
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        docker_host="unix://" + str(socket_path),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        image_environment_sha256=IMAGE_ENVIRONMENT_SHA256,
    )

    with pytest.raises(SandboxPolicyError, match="ModuleNotFoundError"):
        sandbox.preflight(artifacts)

    docker_socket.close()
    socket_path.unlink()
    socket_root.rmdir()


def test_docker_fails_closed_without_exact_safe_image_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    policy = _policy(worktree, artifacts)
    without_attestation = DockerSandbox(
        policy,
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        mount_policies=(
            ContainerMountPolicy("candidate_smoke", "smoke", MANIFEST_SHA),
        ),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        output_quota_verifier=DedicatedFilesystemQuotaVerifier(),
    )
    with pytest.raises(SandboxPolicyError, match="reviewed image environment"):
        without_attestation.prepare(command, config)

    mismatch = DockerSandbox(
        policy,
        image=IMAGE,
        docker_executable=Path(sys.executable).resolve(),
        mount_policies=(
            ContainerMountPolicy("candidate_smoke", "smoke", MANIFEST_SHA),
        ),
        output_quota_max_bytes=OUTPUT_QUOTA_BYTES,
        output_quota_verifier=DedicatedFilesystemQuotaVerifier(),
        image_environment_sha256="0" * 64,
    )
    with pytest.raises(SandboxPolicyError, match="identity mismatch"):
        mismatch.prepare(command, config)

    def credential_environment(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='["API_TOKEN=forbidden"]\n',
            stderr="",
        )

    monkeypatch.setattr("tacorank.execution.sandbox.subprocess.run", credential_environment)
    with pytest.raises(SandboxPolicyError, match="credential-shaped"):
        _docker(policy).prepare(command, config)


def test_dedicated_filesystem_quota_rejects_ordinary_or_oversized_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    sandbox = _docker(_policy(worktree, artifacts))

    monkeypatch.setattr(os.path, "ismount", lambda path: False)
    with pytest.raises(SandboxPolicyError, match="dedicated filesystem mount"):
        sandbox.prepare(command, config)

    monkeypatch.setattr(os.path, "ismount", lambda path: True)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: SimpleNamespace(
            f_frsize=4096,
            f_bsize=4096,
            f_blocks=(OUTPUT_QUOTA_BYTES // 4096) + 1,
        ),
    )
    with pytest.raises(SandboxPolicyError, match="exceeds the reviewed hard quota"):
        sandbox.prepare(command, config)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_backends_reject_symlinked_workspace(tmp_path: Path) -> None:
    command, config, worktree, artifacts = _resolved(tmp_path)
    real_workspace = tmp_path / "real-workspace"
    real_workspace.mkdir()
    worktree.rmdir()
    worktree.symlink_to(real_workspace, target_is_directory=True)
    sandbox = TrustedLocalProcessSandbox(
        _policy(tmp_path, artifacts),
        allow_unsafe_for_tests=True,
    )
    with pytest.raises(SandboxPolicyError, match="symbolic link"):
        sandbox.prepare(command, config)
