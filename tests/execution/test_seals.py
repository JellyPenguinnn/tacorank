from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tacorank.execution import (
    CommandProfile,
    CommandRegistry,
    ExecutionAuthorizationError,
    ExecutionRunner,
    ReceiptArtifactBinding,
    RunnerPolicy,
    SandboxPolicy,
    SealedExecutionVerifier,
    TrustedLocalProcessSandbox,
)
from tacorank.git import WorktreeManager, commit_staged_patch, stage_and_capture
from tacorank.safety import (
    ProtectedManifest,
    ReceiptIdentity,
    ReceiptStore,
    SharedSchemaFactories,
)

from .conftest import ContinuingObserver, StubModel, stub_model_factory
from .store import TestArtifactStore


DATA_MANIFEST_SHA = "d" * 64


class MappingReceiptResolver:
    def __init__(self, receipt_id: str, binding: ReceiptArtifactBinding) -> None:
        self.receipt_id = receipt_id
        self.binding = binding

    def resolve(self, request: Any) -> ReceiptArtifactBinding:
        if request.patch_receipt_id != self.receipt_id:
            raise KeyError("receipt was not recorded by Person 2")
        return self.binding


def _factory(name: str):
    def build(**fields: Any) -> StubModel:
        return StubModel(_model_name=name, **fields)

    return build


FACTORIES = SharedSchemaFactories(
    check_result=_factory("CheckResult"),
    violation=_factory("Violation"),
    patch_check_result=_factory("PatchCheckResult"),
    output_check_result=_factory("OutputCheckResult"),
    artifact_ref=_factory("ArtifactRef"),
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


def _sealed_layout(tmp_path: Path) -> SimpleNamespace:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    (repository / "contract").mkdir()
    (repository / "contract" / "COMPETITION.md").write_text(
        "sealed contract\n", encoding="utf-8"
    )
    (repository / "solution").mkdir()
    (repository / "solution" / "model.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    _git(repository, "add", "--all")
    _git(
        repository,
        "-c",
        "user.name=TacoRank Test",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-m",
        "baseline",
    )
    base_commit = _git(repository, "rev-parse", "HEAD")
    manifest = ProtectedManifest.capture(
        repository,
        ("contract",),
        contract_paths=("contract",),
        data_manifest_sha256=DATA_MANIFEST_SHA,
        require_minimum=False,
    )
    worktrees = WorktreeManager(repository, tmp_path / "worktrees")
    record = worktrees.create("run_001", "exp_0001", base_commit)
    (record.path / "solution" / "model.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    staged = stage_and_capture(record.path, base_commit)
    sealed = commit_staged_patch(record.path, staged, message="candidate patch")
    assert sealed.patch_commit_sha is not None

    receipts = ReceiptStore(repository, FACTORIES)
    written = receipts.write(
        ReceiptIdentity(
            run_id="run_001",
            experiment_id="exp_0001",
            attempt=1,
            patch_commit_sha=sealed.patch_commit_sha,
            diff_sha256=sealed.diff_sha256,
            contract_sha256=manifest.contract_sha256,
            protected_manifest_sha256=manifest.manifest_sha256,
            data_manifest_sha256=DATA_MANIFEST_SHA,
            experiment_root_commit_sha=base_commit,
            cumulative_diff_sha256=sealed.diff_sha256,
        ),
        ({"name": "gate_a", "status": "pass", "summary": "accepted"},),
    )
    binding = ReceiptArtifactBinding(
        written.artifact_ref,
        patch_attempt=1,
        experiment_root_commit_sha=base_commit,
    )
    resolver = MappingReceiptResolver(written.receipt_id, binding)
    verifier = SealedExecutionVerifier(
        worktrees=worktrees,
        receipts=receipts,
        protected_manifest=manifest,
        receipt_artifact_resolver=resolver,
    )
    run_request = StubModel(
        run_id="run_001",
        experiment_id="exp_0001",
        # Execution attempt 2 intentionally reuses the Gate A attempt-1 seal.
        attempt=2,
        fidelity="smoke",
        command_id="candidate_smoke",
        patch_commit_sha=sealed.patch_commit_sha,
        patch_receipt_id=written.receipt_id,
        seed=7,
        data_manifest_sha256=DATA_MANIFEST_SHA,
        timeout_seconds=5.0,
        memory_limit_mb=4096,
        gpu_memory_limit_mb=0,
        network_enabled=False,
    )
    return SimpleNamespace(
        repository=repository,
        worktrees=worktrees,
        workspace=record.path,
        manifest=manifest,
        receipts=receipts,
        written=written,
        resolver=resolver,
        verifier=verifier,
        request=run_request,
        sealed=sealed,
        base_commit=base_commit,
    )


def test_concrete_verifier_accepts_exact_seal_and_same_commit_retry(
    tmp_path: Path,
) -> None:
    layout = _sealed_layout(tmp_path)

    layout.verifier.verify(layout.request, layout.workspace)


def test_concrete_verifier_rejects_receipt_artifact_substitution(
    tmp_path: Path,
) -> None:
    layout = _sealed_layout(tmp_path)
    fields = dict(layout.written.artifact_ref.__dict__)
    fields["sha256"] = "0" * 64
    substituted = StubModel(**fields)
    layout.resolver.binding = ReceiptArtifactBinding(
        substituted,
        patch_attempt=1,
        experiment_root_commit_sha=layout.base_commit,
    )

    with pytest.raises(ExecutionAuthorizationError, match="rejected"):
        layout.verifier.verify(layout.request, layout.workspace)


def test_concrete_verifier_rejects_commit_and_diff_substitution(
    tmp_path: Path,
) -> None:
    layout = _sealed_layout(tmp_path)
    previous = layout.sealed.patch_commit_sha
    (layout.workspace / "solution" / "model.py").write_text(
        "VALUE = 3\n", encoding="utf-8"
    )
    staged = stage_and_capture(layout.workspace, previous)
    replacement = commit_staged_patch(
        layout.workspace, staged, message="substituted patch"
    )
    layout.request.patch_commit_sha = replacement.patch_commit_sha

    with pytest.raises(ExecutionAuthorizationError, match="rejected"):
        layout.verifier.verify(layout.request, layout.workspace)


def test_concrete_verifier_rejects_data_manifest_substitution(
    tmp_path: Path,
) -> None:
    layout = _sealed_layout(tmp_path)
    layout.request.data_manifest_sha256 = "e" * 64

    with pytest.raises(ExecutionAuthorizationError, match="DATA_MANIFEST_MISMATCH"):
        layout.verifier.verify(layout.request, layout.workspace)


def test_concrete_verifier_rejects_experiment_root_substitution(
    tmp_path: Path,
) -> None:
    layout = _sealed_layout(tmp_path)
    layout.resolver.binding = ReceiptArtifactBinding(
        layout.written.artifact_ref,
        patch_attempt=1,
        experiment_root_commit_sha=layout.sealed.patch_commit_sha,
    )

    with pytest.raises(ExecutionAuthorizationError, match="rejected"):
        layout.verifier.verify(layout.request, layout.workspace)


def test_runner_postcheck_detects_worktree_mutation(tmp_path: Path) -> None:
    layout = _sealed_layout(tmp_path)
    artifact_root = layout.repository / "artifacts"
    artifacts = TestArtifactStore(
        layout.repository,
        artifact_root,
        model_factory=stub_model_factory,
    )
    command = CommandProfile(
        command_id="candidate_smoke",
        executable=str(Path(sys.executable).resolve()),
        arguments=(
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('tampered\\n')",
            "{worktree}/solution/model.py",
        ),
        allowed_fidelities=("smoke",),
    )
    sandbox = TrustedLocalProcessSandbox(
        SandboxPolicy(
            allowed_workspace_roots=(layout.worktrees.worktree_root,),
            allowed_artifact_roots=(artifact_root,),
        ),
        allow_unsafe_for_tests=True,
    )
    runner = ExecutionRunner(
        repository_root=layout.repository,
        artifacts=artifacts,
        commands=CommandRegistry((command,)),
        sandbox=sandbox,
        workspace_resolver=lambda run_id, experiment_id: layout.workspace,
        seal_verifier=layout.verifier,
        policy=RunnerPolicy(
            telemetry_interval_seconds=0.02,
            termination_grace_seconds=0.1,
            disk_free_floor_mb=0,
            allow_trusted_local_backend=True,
        ),
        model_factory=stub_model_factory,
    )

    result = runner.run_sync(layout.request, ContinuingObserver())

    assert result.outcome == "contract_error"
    assert result.error_class == "WORKSPACE_SEAL_CHANGED"
    assert len(result.error_fingerprint) == hashlib.sha256().digest_size * 2
