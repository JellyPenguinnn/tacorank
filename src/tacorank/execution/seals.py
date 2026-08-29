"""Concrete verification of the exact Gate A-authorized execution identity."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from tacorank.git import (
    GitOperationError,
    WorktreeLease,
    WorktreeManager,
    WorktreeRecord,
    capture_commit_patch,
    experiment_branch,
    require_ancestor,
    resolve_commit,
)
from tacorank.safety import (
    ProtectedManifest,
    ProtectedManifestError,
    ReceiptIdentity,
    ReceiptStore,
)

from tacorank.execution.runner import ExecutionAuthorizationError


class ReceiptArtifactResolver(Protocol):
    """Resolve Person 2's recorded receipt artifact for one RunRequest."""

    def resolve(self, request: Any) -> "ReceiptArtifactBinding":
        ...


@dataclass(frozen=True)
class ReceiptArtifactBinding:
    """Person 2's accepted Gate A receipt and its coding-attempt identity.

    ``RunRequest.attempt`` counts executions and may increase when retrying the
    same commit.  ``patch_attempt`` is therefore resolved from the accepted
    PatchCheckResult/event rather than equated with the execution attempt.
    """

    artifact_ref: Any
    patch_attempt: int
    experiment_root_commit_sha: str

    def __post_init__(self) -> None:
        _positive_integer(self.patch_attempt, "patch_attempt")
        if not isinstance(self.experiment_root_commit_sha, str):
            raise ValueError("experiment_root_commit_sha must be a Git object ID")


@dataclass(frozen=True)
class VerifiedExecutionSeal:
    """Receipt evidence returned only after every seal check succeeds."""

    receipt_id: str
    receipt_sha256: str
    receipt_path: str
    patch_attempt: int


class SealedExecutionVerifier:
    """Bind a RunRequest to receipt bytes, Git state, and protected content.

    The resolver supplies the shared ``ArtifactRef`` and its coding-attempt
    number from Person 2's accepted PatchCheckResult.  Every security-relevant
    identity is independently recomputed or taken from the controller-owned
    frozen manifest before ``ReceiptStore.verify`` checks the hash-bound bytes.
    """

    def __init__(
        self,
        *,
        worktrees: WorktreeManager,
        receipts: ReceiptStore,
        protected_manifest: ProtectedManifest,
        receipt_artifact_resolver: ReceiptArtifactResolver,
    ) -> None:
        self.worktrees = worktrees
        self.receipts = receipts
        self.protected_manifest = protected_manifest
        self.receipt_artifact_resolver = receipt_artifact_resolver

    def acquire_lease(
        self,
        request: Any,
        workspace: Path,
        *,
        timeout_seconds: float,
    ) -> WorktreeLease:
        """Hold the shared coding/execution lease for the complete run."""

        run_id = str(_field(request, "run_id"))
        experiment_id = str(_field(request, "experiment_id"))
        record = WorktreeRecord(
            repository=self.worktrees.repository,
            path=Path(workspace),
            branch=experiment_branch(run_id, experiment_id),
            run_id=run_id,
            experiment_id=experiment_id,
            commit_sha=str(_field(request, "patch_commit_sha")),
        )
        try:
            return self.worktrees.acquire_lease(
                record,
                timeout_seconds=timeout_seconds,
            )
        except (GitOperationError, OSError, ValueError, TypeError) as error:
            code = getattr(error, "code", type(error).__name__)
            raise ExecutionAuthorizationError(
                "sealed execution lease rejected: {0}".format(code)
            ) from error

    def verify(self, request: Any, workspace: Path) -> VerifiedExecutionSeal:
        try:
            return self._verify(request, workspace)
        except ExecutionAuthorizationError:
            raise
        except (
            GitOperationError,
            ProtectedManifestError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            code = getattr(error, "code", type(error).__name__)
            raise ExecutionAuthorizationError(
                "sealed execution identity rejected: {0}".format(code)
            ) from error

    def _verify(self, request: Any, workspace: Path) -> VerifiedExecutionSeal:
        run_id = str(_field(request, "run_id"))
        experiment_id = str(_field(request, "experiment_id"))
        _positive_integer(_field(request, "attempt"), "attempt")
        patch_commit_sha = str(_field(request, "patch_commit_sha"))
        receipt_id = str(_field(request, "patch_receipt_id"))
        data_manifest_sha256 = str(_field(request, "data_manifest_sha256"))

        expected_workspace = self.worktrees.path_for(run_id, experiment_id)
        actual_workspace = Path(workspace)
        if actual_workspace.is_symlink():
            raise ExecutionAuthorizationError(
                "sealed execution identity rejected: WORKTREE_PATH_SYMLINK"
            )
        actual_workspace = actual_workspace.resolve(strict=True)
        if actual_workspace != expected_workspace:
            raise ExecutionAuthorizationError(
                "sealed execution identity rejected: WORKTREE_IDENTITY_MISMATCH"
            )

        # attach() proves this is the registered experiment branch, at the exact
        # requested commit, with no tracked or untracked modifications.
        self.worktrees.attach(
            run_id,
            experiment_id,
            patch_commit_sha,
            require_clean=True,
        )

        base_commit_sha = _single_parent(actual_workspace, patch_commit_sha)
        sealed_patch = capture_commit_patch(
            actual_workspace,
            base_commit_sha,
            patch_commit_sha,
        )

        if data_manifest_sha256 != self.protected_manifest.data_manifest_sha256:
            raise ExecutionAuthorizationError(
                "sealed execution identity rejected: DATA_MANIFEST_MISMATCH"
            )
        manifest_verification = self.protected_manifest.verify(actual_workspace)
        if (
            not manifest_verification.valid
            or manifest_verification.current_contract_sha256
            != self.protected_manifest.contract_sha256
        ):
            raise ExecutionAuthorizationError(
                "sealed execution identity rejected: PROTECTED_STATE_MISMATCH"
            )

        binding = self.receipt_artifact_resolver.resolve(request)
        if binding is None or binding.artifact_ref is None:
            raise ExecutionAuthorizationError(
                "sealed execution identity rejected: RECEIPT_ARTIFACT_MISSING"
            )
        patch_attempt = _positive_integer(binding.patch_attempt, "patch_attempt")
        experiment_root_commit_sha = resolve_commit(
            actual_workspace, binding.experiment_root_commit_sha
        )
        require_ancestor(actual_workspace, experiment_root_commit_sha, base_commit_sha)
        require_ancestor(actual_workspace, experiment_root_commit_sha, patch_commit_sha)
        cumulative_diff_sha256 = hashlib.sha256(
            _cumulative_diff(
                actual_workspace,
                experiment_root_commit_sha,
                patch_commit_sha,
            )
        ).hexdigest()
        identity = ReceiptIdentity(
            run_id=run_id,
            experiment_id=experiment_id,
            attempt=patch_attempt,
            patch_commit_sha=patch_commit_sha,
            diff_sha256=sealed_patch.diff_sha256,
            contract_sha256=self.protected_manifest.contract_sha256,
            protected_manifest_sha256=self.protected_manifest.manifest_sha256,
            data_manifest_sha256=data_manifest_sha256,
            experiment_root_commit_sha=experiment_root_commit_sha,
            cumulative_diff_sha256=cumulative_diff_sha256,
        )
        self.receipts.verify(
            binding.artifact_ref,
            identity,
            receipt_id=receipt_id,
        )
        return VerifiedExecutionSeal(
            receipt_id=receipt_id,
            receipt_sha256=str(_field(binding.artifact_ref, "sha256")),
            receipt_path=str(_field(binding.artifact_ref, "path")),
            patch_attempt=patch_attempt,
        )


def _single_parent(workspace: Path, patch_commit_sha: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "protocol.file.allow=never",
            "-C",
            str(workspace),
            "rev-list",
            "--parents",
            "-n",
            "1",
            patch_commit_sha,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
        shell=False,
    )
    if completed.returncode != 0:
        raise ExecutionAuthorizationError(
            "sealed execution identity rejected: GIT_PARENT_LOOKUP_FAILED"
        )
    try:
        line = completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ExecutionAuthorizationError(
            "sealed execution identity rejected: MALFORMED_GIT_OUTPUT"
        ) from error
    fields = line.split(" ")
    if len(fields) != 2 or fields[0] != patch_commit_sha:
        raise ExecutionAuthorizationError(
            "sealed execution identity rejected: PATCH_PARENT_MISMATCH"
        )
    return fields[1]


def _cumulative_diff(
    workspace: Path,
    experiment_root_commit_sha: str,
    patch_commit_sha: str,
) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "protocol.file.allow=never",
            "-C",
            str(workspace),
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--no-relative",
            "--no-indent-heuristic",
            "--diff-algorithm=myers",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "--no-renames",
            experiment_root_commit_sha,
            patch_commit_sha,
            "--",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
        shell=False,
    )
    if completed.returncode != 0:
        raise ExecutionAuthorizationError(
            "sealed execution identity rejected: CUMULATIVE_DIFF_FAILED"
        )
    return completed.stdout


def _git_environment() -> Mapping[str, str]:
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError("request is missing {0}".format(name))
        return value[name]
    if not hasattr(value, name):
        raise ValueError("request is missing {0}".format(name))
    return getattr(value, name)


def _positive_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("{0} must be a positive integer".format(field))
    return value
