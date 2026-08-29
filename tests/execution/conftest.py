from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import pytest

from tacorank.execution.commands import (
    CommandProfile,
    CommandRegistry,
    ExpectedArtifact,
)
from tacorank.execution.runner import ExecutionRunner, RunnerPolicy
from tacorank.execution.sandbox import SandboxPolicy, TrustedLocalProcessSandbox

from .store import TestArtifactStore


class StubModel(SimpleNamespace):
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return dict(self.__dict__)


def stub_model_factory(model_name: str, **fields: Any) -> StubModel:
    return StubModel(_model_name=model_name, **fields)


class RecordingSealVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, request: Any, workspace: Path) -> StubModel:
        assert workspace.is_dir()
        assert request.patch_receipt_id
        self.calls += 1
        return StubModel(receipt_sha256="c" * 64)

    def acquire_lease(
        self,
        request: Any,
        workspace: Path,
        *,
        timeout_seconds: float,
    ) -> Any:
        del request, workspace, timeout_seconds
        return nullcontext()


class ContinuingObserver:
    def __init__(self) -> None:
        self.samples: list[Any] = []

    def observe(self, sample: Any) -> StubModel:
        self.samples.append(sample)
        return StubModel(action="continue", reason_code=None, summary=None)


def request(
    *,
    command_id: str = "candidate_smoke",
    attempt: int = 1,
    fidelity: str = "smoke",
    timeout_seconds: float = 5.0,
    memory_limit_mb: int = 4096,
    gpu_memory_limit_mb: int = 0,
    network_enabled: bool = False,
) -> StubModel:
    return StubModel(
        run_id="run_001",
        experiment_id="exp_0001",
        attempt=attempt,
        fidelity=fidelity,
        command_id=command_id,
        patch_commit_sha="a" * 40,
        patch_receipt_id="receipt_001",
        seed=7,
        data_manifest_sha256="b" * 64,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
        gpu_memory_limit_mb=gpu_memory_limit_mb,
        network_enabled=network_enabled,
    )


@pytest.fixture
def execution_layout(tmp_path: Path) -> SimpleNamespace:
    repository = tmp_path / "repository"
    workspace_root = repository / "worktrees"
    workspace = workspace_root / "run_001" / "exp_0001"
    artifact_root = repository / "artifacts"
    workspace.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    store = TestArtifactStore(
        repository,
        artifact_root,
        model_factory=stub_model_factory,
    )
    sandbox = TrustedLocalProcessSandbox(
        SandboxPolicy(
            allowed_workspace_roots=(workspace_root,),
            allowed_artifact_roots=(artifact_root,),
        ),
        allow_unsafe_for_tests=True,
    )
    return SimpleNamespace(
        repository=repository,
        workspace=workspace,
        artifact_root=artifact_root,
        store=store,
        sandbox=sandbox,
    )


def command_registry(
    code: str,
    *,
    expected: Tuple[ExpectedArtifact, ...] = (),
    allow_network: bool = False,
    extra_arguments: Tuple[str, ...] = (),
) -> CommandRegistry:
    profile = CommandProfile(
        command_id="candidate_smoke",
        executable=str(Path(sys.executable).resolve()),
        arguments=("-c", code) + extra_arguments,
        allowed_fidelities=("smoke",),
        allow_network=allow_network,
        expected_artifacts=expected,
    )
    return CommandRegistry((profile,))


def build_runner(
    layout: SimpleNamespace,
    registry: CommandRegistry,
    *,
    seal_verifier: Optional[RecordingSealVerifier] = None,
    process_launcher: Any = None,
    telemetry_collector_factory: Any = None,
    submission_artifact_resolver: Any = None,
    artifact_store: Any = None,
    model_factory: Any = stub_model_factory,
    interval: float = 0.02,
    disk_floor_mb: int = 0,
) -> tuple[ExecutionRunner, RecordingSealVerifier]:
    seal = seal_verifier or RecordingSealVerifier()
    keyword_arguments = {}
    if process_launcher is not None:
        keyword_arguments["process_launcher"] = process_launcher
    if telemetry_collector_factory is not None:
        keyword_arguments["telemetry_collector_factory"] = telemetry_collector_factory
    if submission_artifact_resolver is not None:
        keyword_arguments["submission_artifact_resolver"] = (
            submission_artifact_resolver
        )
    runner = ExecutionRunner(
        repository_root=layout.repository,
        artifacts=artifact_store or layout.store,
        commands=registry,
        sandbox=layout.sandbox,
        workspace_resolver=lambda run_id, experiment_id: layout.workspace,
        seal_verifier=seal,
        policy=RunnerPolicy(
            telemetry_interval_seconds=interval,
            termination_grace_seconds=0.15,
            disk_free_floor_mb=disk_floor_mb,
            allow_trusted_local_backend=True,
        ),
        model_factory=model_factory,
        **keyword_arguments,
    )
    return runner, seal
