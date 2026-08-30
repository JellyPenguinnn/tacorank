from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import tacorank.git.worktrees as worktrees_module

from tacorank.git.patches import (
    capture_commit_range,
    capture_commit_patch,
    commit_staged_patch,
    stage_and_capture,
    write_artifact,
)
from tacorank.git.refs import (
    GitOperationError,
    experiment_branch,
    is_ancestor,
    resolve_commit,
)
from tacorank.git.worktrees import WorktreeManager


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "solution").mkdir()
    (root / "solution" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "solution/model.py")
    _git(root, "commit", "-q", "-m", "base")
    return root, _git(root, "rev-parse", "HEAD")


def test_ref_names_and_object_ids_are_fail_closed(repository: tuple[Path, str]) -> None:
    root, base = repository
    assert experiment_branch("run_01", "exp-01") == "experiment/run_01/exp-01"
    assert resolve_commit(root, base) == base

    for invalid in ("../run", "run/other", ".hidden", "run..id", "run.lock"):
        with pytest.raises(GitOperationError) as failure:
            experiment_branch(invalid, "exp")
        assert failure.value.code == "INVALID_IDENTIFIER"
    with pytest.raises(GitOperationError) as failure:
        resolve_commit(root, "HEAD")
    assert failure.value.code == "INVALID_OBJECT_ID"


def test_worktree_patch_commit_and_safe_removal(
    repository: tuple[Path, str], tmp_path: Path
) -> None:
    root, base = repository
    manager = WorktreeManager(root, tmp_path / "worktrees")
    record = manager.create("run1", "exp1", base)
    assert record.path == manager.path_for("run1", "exp1")
    assert manager.verify(record, expected_commit_sha=base, require_clean=True) == base

    changed = record.path / "solution" / "model.py"
    changed.write_text("VALUE = 2\n", encoding="utf-8")
    added = record.path / "solution" / "new.py"
    added.write_text("ENABLED = True\n", encoding="utf-8")

    staged = stage_and_capture(record.path, base)
    assert staged.patch_commit_sha is None
    assert staged.changed_files == ("solution/model.py", "solution/new.py")
    assert staged.diff_sha256 == hashlib.sha256(staged.diff).hexdigest()
    assert b"VALUE = 2" in staged.diff

    sealed = commit_staged_patch(record.path, staged, message="candidate patch")
    assert sealed.patch_commit_sha is not None
    assert is_ancestor(root, base, sealed.patch_commit_sha)
    assert sealed.diff == staged.diff
    assert capture_commit_patch(record.path, base, sealed.patch_commit_sha) == sealed

    with pytest.raises(GitOperationError) as failure:
        manager.remove(
            record,
            expected_commit_sha=sealed.patch_commit_sha,
            terminal_or_safe_checkpoint=False,
        )
    assert failure.value.code == "UNSAFE_WORKTREE_REMOVAL"

    manager.remove(
        record,
        expected_commit_sha=sealed.patch_commit_sha,
        terminal_or_safe_checkpoint=True,
    )
    assert not record.path.exists()
    assert _git(root, "rev-parse", record.branch) == sealed.patch_commit_sha

    repair_record = manager.create(
        "run1", "exp1", base, reuse_existing_branch=True
    )
    assert repair_record.commit_sha == sealed.patch_commit_sha
    manager.remove(
        repair_record,
        expected_commit_sha=sealed.patch_commit_sha,
        terminal_or_safe_checkpoint=True,
    )


def test_patch_substitution_is_rejected(
    repository: tuple[Path, str], tmp_path: Path
) -> None:
    root, base = repository
    manager = WorktreeManager(root, tmp_path / "worktrees")
    record = manager.create("run1", "exp2", base)
    candidate = record.path / "solution" / "model.py"
    candidate.write_text("VALUE = 2\n", encoding="utf-8")
    captured = stage_and_capture(record.path, base)

    candidate.write_text("VALUE = 999\n", encoding="utf-8")
    _git(record.path, "add", "--all", "--", ".")
    with pytest.raises(GitOperationError) as failure:
        commit_staged_patch(record.path, captured, message="substituted")
    assert failure.value.code == "PATCH_SUBSTITUTION"


def test_dirty_or_wrong_commit_worktree_is_rejected(
    repository: tuple[Path, str], tmp_path: Path
) -> None:
    root, base = repository
    manager = WorktreeManager(root, tmp_path / "worktrees")
    record = manager.create("run1", "exp3", base)
    dotgit = record.path / ".git"
    original_dotgit = dotgit.read_bytes()
    dotgit.write_text("gitdir: /tmp/unreviewed\n", encoding="utf-8")
    with pytest.raises(GitOperationError) as failure:
        manager.verify(record, expected_commit_sha=base, require_clean=True)
    assert failure.value.code == "WORKTREE_GIT_ADMIN_INVALID"
    dotgit.write_bytes(original_dotgit)
    (record.path / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(GitOperationError) as failure:
        manager.verify(record, expected_commit_sha=base, require_clean=True)
    assert failure.value.code == "WORKTREE_DIRTY"


def test_ignored_worktree_change_is_rejected(
    repository: tuple[Path, str], tmp_path: Path
) -> None:
    root, _ = repository
    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-m", "ignore generated file")
    base = _git(root, "rev-parse", "HEAD").strip()
    manager = WorktreeManager(root, tmp_path / "worktrees")
    record = manager.create("run1", "ignored", base)
    (record.path / "ignored.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(GitOperationError) as failure:
        manager.verify(record, expected_commit_sha=base, require_clean=True)

    assert failure.value.code == "WORKTREE_DIRTY"


def test_worktree_lease_is_exclusive_bounded_and_stale_safe(
    repository: tuple[Path, str], tmp_path: Path
) -> None:
    root, base = repository
    manager = WorktreeManager(root, tmp_path / "worktrees")
    record = manager.create("lease", "exp1", base)

    def contend() -> None:
        with manager.acquire_lease(record, timeout_seconds=0.1):
            raise AssertionError("contender unexpectedly acquired the held lease")

    with manager.acquire_lease(record, timeout_seconds=1) as lease:
        assert lease.lock_path.parent.name == ".tacorank-locks"
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(contend)
            with pytest.raises(GitOperationError) as failure:
                future.result(timeout=2)
            assert failure.value.code == "WORKTREE_LEASE_TIMEOUT"

    # The diagnostic file persists, but kernel/thread ownership is released;
    # process death has the same stale-safe flock release behavior.
    with manager.acquire_lease(record, timeout_seconds=1):
        assert True


def test_artifact_writes_are_confined_and_immutable(
    repository: tuple[Path, str], tmp_path: Path
) -> None:
    root, _ = repository
    content = b"exact diff bytes\n"
    artifact = write_artifact(
        root,
        "artifacts/run1/exp1/attempt_1/patch.diff",
        content,
        content_type="text/x-diff",
    )
    assert artifact.sha256 == hashlib.sha256(content).hexdigest()
    assert (root / artifact.path).read_bytes() == content
    assert write_artifact(
        root,
        artifact.path,
        content,
        content_type="text/x-diff",
    ) == artifact

    with pytest.raises(GitOperationError) as failure:
        write_artifact(root, "../outside", b"bad", content_type="text/plain")
    assert failure.value.code == "INVALID_RELATIVE_PATH"

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = root / "artifacts" / "link"
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(GitOperationError) as failure:
        write_artifact(root, "artifacts/link/leak", b"bad", content_type="text/plain")
    assert failure.value.code == "ARTIFACT_PATH_ESCAPE"


def test_cumulative_commit_range_preserves_direct_child_semantics(
    repository: tuple[Path, str], tmp_path: Path
) -> None:
    root, base = repository
    manager = WorktreeManager(root, tmp_path / "worktrees")
    record = manager.create("range", "exp1", base)
    candidate = record.path / "solution" / "model.py"
    candidate.write_text("VALUE = 2\n", encoding="utf-8")
    first = commit_staged_patch(
        record.path, stage_and_capture(record.path, base), message="first"
    )
    candidate.write_text("VALUE = 3\n", encoding="utf-8")
    second = commit_staged_patch(
        record.path,
        stage_and_capture(record.path, first.patch_commit_sha),
        message="second",
    )

    cumulative = capture_commit_range(record.path, base, second.patch_commit_sha)
    assert cumulative.base_commit_sha == base
    assert cumulative.patch_commit_sha == second.patch_commit_sha
    assert cumulative.changed_files == ("solution/model.py",)
    assert b"VALUE = 3" in cumulative.diff
    with pytest.raises(GitOperationError) as failure:
        capture_commit_patch(record.path, base, second.patch_commit_sha)
    assert failure.value.code == "PATCH_PARENT_MISMATCH"


def _repository_with_local_submodule(tmp_path: Path) -> tuple[Path, str, str]:
    upstream = tmp_path / "official-upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q")
    _git(upstream, "config", "user.name", "Test User")
    _git(upstream, "config", "user.email", "test@example.invalid")
    (upstream / "payload.txt").write_text("reviewed\n", encoding="utf-8")
    _git(upstream, "add", "payload.txt")
    _git(upstream, "commit", "-q", "-m", "reviewed submodule")
    submodule_sha = _git(upstream, "rev-parse", "HEAD")

    root = tmp_path / "superproject"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "solution").mkdir()
    (root / "solution" / ".gitkeep").write_text("", encoding="utf-8")
    _git(root, "add", "solution/.gitkeep")
    _git(root, "commit", "-q", "-m", "base")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--name",
        "official-kit",
        str(upstream),
        "official-kit",
    )
    modules = (root / ".gitmodules").read_text(encoding="utf-8")
    (root / ".gitmodules").write_text(
        modules.replace(str(upstream), "https://example.invalid/must-not-fetch.git"),
        encoding="utf-8",
    )
    _git(root, "add", ".gitmodules", "official-kit")
    _git(root, "commit", "-q", "-m", "reviewed official kit")
    return root, _git(root, "rev-parse", "HEAD"), submodule_sha


def test_submodule_initialization_uses_only_allowlisted_local_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base, submodule_sha = _repository_with_local_submodule(tmp_path)
    unconfigured = WorktreeManager(root, tmp_path / "unconfigured-worktrees")
    with pytest.raises(GitOperationError) as failure:
        unconfigured.create("run1", "unreviewed", base)
    assert failure.value.code == "SUBMODULE_POLICY_MISMATCH"

    manager = WorktreeManager(
        root,
        tmp_path / "worktrees",
        required_submodules=("official-kit",),
    )
    calls: list[tuple[str, ...]] = []
    original_git = worktrees_module._git

    def recording_git(
        repository: Path,
        args: tuple[str, ...],
        **kwargs: object,
    ) -> object:
        calls.append(tuple(args))
        return original_git(repository, args, **kwargs)

    monkeypatch.setattr(worktrees_module, "_git", recording_git)
    record = manager.create("run1", "reviewed", base)
    submodule = record.path / "official-kit"
    assert (submodule / "payload.txt").read_text(encoding="utf-8") == "reviewed\n"
    assert _git(submodule, "rev-parse", "HEAD") == submodule_sha
    assert manager.verify(record, expected_commit_sha=base, require_clean=True) == base
    update_calls = [
        args for args in calls if "submodule" in args and "update" in args
    ]
    assert len(update_calls) == 1
    update_args = update_calls[0]
    command_index = update_args.index("submodule")
    assert update_args[command_index : command_index + 2] == ("submodule", "update")
    assert "--no-fetch" in update_args
    assert update_args[-2:] == ("--", "official-kit")

    submodule_dotgit = submodule / ".git"
    original_submodule_dotgit = submodule_dotgit.read_bytes()
    submodule_dotgit.write_text("gitdir: /tmp/unreviewed\n", encoding="utf-8")
    with pytest.raises(GitOperationError) as failure:
        manager.verify(record, expected_commit_sha=base, require_clean=False)
    assert failure.value.code == "WORKTREE_GIT_ADMIN_INVALID"
    submodule_dotgit.write_bytes(original_submodule_dotgit)

    (submodule / "payload.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(GitOperationError) as failure:
        manager.verify(record, expected_commit_sha=base, require_clean=False)
    assert failure.value.code == "SUBMODULE_DIRTY"
    _git(submodule, "checkout", "--", "payload.txt")
    manager.remove(
        record,
        expected_commit_sha=base,
        terminal_or_safe_checkpoint=True,
    )


def test_submodule_policy_rejects_behavior_options_and_candidate_ref_updates(
    tmp_path: Path,
) -> None:
    root, base, _ = _repository_with_local_submodule(tmp_path)
    modules = (root / ".gitmodules").read_text(encoding="utf-8")
    (root / ".gitmodules").write_text(
        modules + "\tupdate = !unreviewed-command\n", encoding="utf-8"
    )
    _git(root, "add", ".gitmodules")
    _git(root, "commit", "-q", "-m", "unreviewed submodule behavior")
    unsafe = _git(root, "rev-parse", "HEAD")
    manager = WorktreeManager(
        root,
        tmp_path / "unsafe-worktrees",
        required_submodules=("official-kit",),
    )
    with pytest.raises(GitOperationError) as failure:
        manager.create("run2", "unsafe", unsafe)
    assert failure.value.code == "SUBMODULE_POLICY_INVALID"

    (root / ".gitmodules").write_text(
        "[DEFAULT]\n\tignored = unreviewed\n" + modules,
        encoding="utf-8",
    )
    _git(root, "add", ".gitmodules")
    _git(root, "commit", "-q", "-m", "unreviewed submodule defaults")
    unsafe_defaults = _git(root, "rev-parse", "HEAD")
    with pytest.raises(GitOperationError) as failure:
        manager.create("run2", "unsafe-defaults", unsafe_defaults)
    assert failure.value.code == "SUBMODULE_POLICY_INVALID"

    # The exact reviewed commit remains usable, but a candidate cannot advance
    # its gitlink even when the submodule path itself is present in the tree.
    record = manager.create("run2", "gitlink", base)
    submodule = record.path / "official-kit"
    (submodule / "payload.txt").write_text("candidate ref\n", encoding="utf-8")
    _git(submodule, "config", "user.name", "Test User")
    _git(submodule, "config", "user.email", "test@example.invalid")
    _git(submodule, "add", "payload.txt")
    _git(submodule, "commit", "-q", "-m", "candidate submodule commit")
    with pytest.raises(GitOperationError) as failure:
        stage_and_capture(record.path, base)
    assert failure.value.code == "SUBMODULE_UPDATE_FORBIDDEN"
