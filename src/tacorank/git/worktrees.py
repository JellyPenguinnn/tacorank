"""Deterministic lifecycle management for disposable experiment worktrees."""

from __future__ import annotations

import configparser
import errno
import hashlib
import math
import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Sequence, Tuple

from .patches import validate_relative_path

from .refs import (
    GitOperationError,
    _decode_line,
    _git,
    branch_tip,
    experiment_branch,
    require_ancestor,
    resolve_commit,
    validate_identifier,
    validated_repository,
)

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

try:  # Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None


@dataclass(frozen=True)
class WorktreeRecord:
    """Controller-owned identity for one experiment worktree."""

    repository: Path
    path: Path
    branch: str
    run_id: str
    experiment_id: str
    commit_sha: str


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: Dict[str, threading.Lock] = {}


def _lock_descriptor(descriptor: int, *, nonblocking: bool) -> None:
    if _fcntl is not None:
        flags = _fcntl.LOCK_EX | (_fcntl.LOCK_NB if nonblocking else 0)
        _fcntl.flock(descriptor, flags)
        return
    if _msvcrt is not None:
        os.lseek(descriptor, 0, os.SEEK_END)
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        mode = _msvcrt.LK_NBLCK if nonblocking else _msvcrt.LK_LOCK
        _msvcrt.locking(descriptor, mode, 1)
        return
    raise OSError("no supported file-locking backend is available")


def _unlock_descriptor(descriptor: int) -> None:
    if _fcntl is not None:
        _fcntl.flock(descriptor, _fcntl.LOCK_UN)
    elif _msvcrt is not None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        _msvcrt.locking(descriptor, _msvcrt.LK_UNLCK, 1)


class WorktreeLease:
    """Exclusive stale-safe lease for one canonical experiment worktree."""

    def __init__(
        self,
        *,
        descriptor: int,
        thread_lock: threading.Lock,
        lock_path: Path,
    ) -> None:
        self._descriptor: Optional[int] = descriptor
        self._thread_lock = thread_lock
        self.lock_path = lock_path

    def __enter__(self) -> "WorktreeLease":
        if self._descriptor is None:
            raise GitOperationError("WORKTREE_LEASE_INVALID", "lease is already released")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            _unlock_descriptor(descriptor)
        finally:
            os.close(descriptor)
            self._thread_lock.release()


class WorktreeManager:
    """Create, verify, and safely remove controller-owned Git worktrees.

    The manager never recursively deletes a path.  Removal is delegated to
    ``git worktree remove`` only after identity, cleanliness, branch, and commit
    checks succeed.
    """

    def __init__(
        self,
        repository: Path,
        worktree_root: Path,
        *,
        required_submodules: Sequence[str] = (),
    ) -> None:
        self.repository = validated_repository(Path(repository))
        requested_root = Path(worktree_root).expanduser()
        requested_root.mkdir(parents=True, exist_ok=True)
        self.worktree_root = requested_root.resolve(strict=True)
        if self.worktree_root == self.repository or _is_within(
            self.worktree_root, self.repository
        ):
            raise GitOperationError(
                "INVALID_WORKTREE_ROOT",
                "managed worktrees must be outside the source repository",
            )
        try:
            normalized_submodules = tuple(
                validate_relative_path(path) for path in required_submodules
            )
        except (TypeError, ValueError) as exc:
            raise GitOperationError(
                "INVALID_SUBMODULE_POLICY", "required_submodules contains an invalid path"
            ) from exc
        if len(normalized_submodules) != len(set(normalized_submodules)):
            raise GitOperationError(
                "INVALID_SUBMODULE_POLICY", "required_submodules contains duplicates"
            )
        self.required_submodules = tuple(sorted(normalized_submodules))
        locks_root = self.worktree_root / ".tacorank-locks"
        locks_root.mkdir(mode=0o700, exist_ok=True)
        if locks_root.is_symlink() or locks_root.resolve(strict=True) != locks_root:
            raise GitOperationError(
                "WORKTREE_LEASE_ROOT_INVALID", "worktree lease root is not canonical"
            )
        os.chmod(locks_root, 0o700)
        self._locks_root = locks_root

    def preflight(self, baseline_commit_sha: str) -> str:
        """Verify the baseline and required submodules without creating a worktree."""

        baseline = resolve_commit(self.repository, baseline_commit_sha)
        self._submodule_declarations(baseline)
        self._verify_submodules(self.repository, baseline)
        return baseline

    def path_for(self, run_id: str, experiment_id: str) -> Path:
        """Return the deterministic path for a validated experiment identity."""

        run = validate_identifier(run_id, "run_id")
        experiment = validate_identifier(experiment_id, "experiment_id")
        path = self.worktree_root / run / experiment
        if not _is_within(path, self.worktree_root):
            raise GitOperationError("WORKTREE_PATH_ESCAPE", "worktree path escaped its root")
        return path

    def acquire_lease(
        self,
        record: WorktreeRecord,
        *,
        timeout_seconds: float,
    ) -> WorktreeLease:
        """Acquire the shared coding/execution lease within a hard deadline."""

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > 300
        ):
            raise GitOperationError(
                "WORKTREE_LEASE_INVALID", "lease timeout must be in (0, 300] seconds"
            )
        expected_path = self.path_for(record.run_id, record.experiment_id)
        try:
            actual_path = Path(record.path).resolve(strict=True)
        except OSError as exc:
            raise GitOperationError(
                "WORKTREE_LEASE_INVALID", "lease worktree does not exist"
            ) from exc
        if (
            actual_path != expected_path
            or Path(record.repository).resolve(strict=True) != self.repository
            or record.branch != experiment_branch(record.run_id, record.experiment_id)
        ):
            raise GitOperationError(
                "WORKTREE_IDENTITY_MISMATCH", "cannot lease an unexpected worktree"
            )
        key = hashlib.sha256(str(expected_path).encode("utf-8")).hexdigest()
        lock_path = self._locks_root / f"{key}.lock"
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(str(lock_path), threading.Lock())
        deadline = time.monotonic() + float(timeout_seconds)
        if not thread_lock.acquire(timeout=float(timeout_seconds)):
            raise GitOperationError(
                "WORKTREE_LEASE_TIMEOUT", "timed out waiting for the in-process lease"
            )
        descriptor: Optional[int] = None
        try:
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            elif lock_path.is_symlink():
                raise GitOperationError(
                    "WORKTREE_LEASE_INVALID", "lease path must not be a symlink"
                )
            descriptor = os.open(lock_path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (
                    hasattr(os, "getuid")
                    and metadata.st_uid != os.getuid()
                )
            ):
                raise GitOperationError(
                    "WORKTREE_LEASE_INVALID", "lease file is not a private regular file"
                )
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            while True:
                try:
                    _lock_descriptor(descriptor, nonblocking=True)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise GitOperationError(
                            "WORKTREE_LEASE_TIMEOUT",
                            "timed out waiting for the cross-process worktree lease",
                        ) from exc
                    time.sleep(min(0.02, remaining))
            os.ftruncate(descriptor, 0)
            identity = (
                f"pid={os.getpid()}\npath_sha256={key}\n"
                f"branch={record.branch}\n"
            ).encode("ascii")
            os.write(descriptor, identity)
            os.fsync(descriptor)
            return WorktreeLease(
                descriptor=descriptor,
                thread_lock=thread_lock,
                lock_path=lock_path,
            )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            thread_lock.release()
            raise
    def create(
        self,
        run_id: str,
        experiment_id: str,
        parent_commit_sha: str,
        *,
        reuse_existing_branch: bool = False,
    ) -> WorktreeRecord:
        """Create an experiment branch/worktree from an exact parent commit.

        ``reuse_existing_branch`` is reserved for repair attempts.  When set,
        the existing branch tip must descend from the originally declared
        parent.  Initial attempts fail if the experiment branch already exists.
        """

        parent = resolve_commit(self.repository, parent_commit_sha)
        branch = experiment_branch(run_id, experiment_id)
        target = self.path_for(run_id, experiment_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.resolve(strict=True) != target.parent:
            raise GitOperationError(
                "WORKTREE_PATH_ESCAPE",
                "worktree parent contains a symlink or non-canonical component",
            )
        if target.exists() or target.is_symlink():
            raise GitOperationError(
                "WORKTREE_PATH_EXISTS", f"worktree target already exists: {target}"
            )

        tip = branch_tip(self.repository, branch)
        created_branch = False
        args: Tuple[str, ...]
        if tip is None:
            if reuse_existing_branch:
                raise GitOperationError(
                    "MISSING_EXPERIMENT_BRANCH",
                    f"repair branch does not exist: {branch}",
                )
            args = ("worktree", "add", "-b", branch, str(target), parent)
            expected_commit = parent
            created_branch = True
        else:
            if not reuse_existing_branch:
                raise GitOperationError(
                    "EXPERIMENT_BRANCH_EXISTS",
                    f"experiment branch already exists: {branch}",
                )
            require_ancestor(self.repository, parent, tip)
            args = ("worktree", "add", str(target), branch)
            expected_commit = tip

        try:
            _git(self.repository, args)
            record = WorktreeRecord(
                repository=self.repository,
                path=target.resolve(strict=True),
                branch=branch,
                run_id=run_id,
                experiment_id=experiment_id,
                commit_sha=expected_commit,
            )
            self._initialize_submodules(record, expected_commit)
            self.verify(record, expected_commit_sha=expected_commit, require_clean=True)
            return record
        except Exception:
            self._rollback_failed_creation(target, branch, created_branch)
            raise

    def attach(
        self,
        run_id: str,
        experiment_id: str,
        expected_commit_sha: str,
        *,
        require_clean: bool = True,
    ) -> WorktreeRecord:
        """Reconstruct and verify the identity of an existing managed worktree."""

        expected = resolve_commit(self.repository, expected_commit_sha)
        record = WorktreeRecord(
            repository=self.repository,
            path=self.path_for(run_id, experiment_id),
            branch=experiment_branch(run_id, experiment_id),
            run_id=run_id,
            experiment_id=experiment_id,
            commit_sha=expected,
        )
        self.verify(
            record,
            expected_commit_sha=expected,
            require_clean=require_clean,
        )
        return record

    def verify(
        self,
        record: WorktreeRecord,
        *,
        expected_commit_sha: str,
        require_clean: bool,
    ) -> str:
        """Verify path ownership, branch identity, commit, and optional cleanliness."""

        actual_path, expected = self._verify_identity_and_head(
            record, expected_commit_sha=expected_commit_sha
        )
        self._verify_submodules(actual_path, expected)
        if require_clean:
            status = _git(
                actual_path,
                (
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                    "--ignored=matching",
                ),
            ).stdout
            if status:
                raise GitOperationError("WORKTREE_DIRTY", "experiment worktree is not clean")
        return expected

    def discard_uncommitted_changes(
        self,
        record: WorktreeRecord,
        *,
        expected_commit_sha: str,
    ) -> None:
        """Restore a leased disposable worktree to its exact clean commit.

        This is intentionally narrower than a general reset primitive.  The
        caller must hold the worktree lease, and identity, branch, HEAD, and
        every required submodule are verified before any cleanup occurs.
        """

        actual_path, expected = self._verify_identity_and_head(
            record, expected_commit_sha=expected_commit_sha
        )
        self._verify_submodules(actual_path, expected, require_clean=False)
        for path in self.required_submodules:
            target = actual_path.joinpath(*PurePosixPath(path).parts)
            submodule_commit = self._gitlink_sha(expected, path)
            _git(target, ("reset", "--hard", submodule_commit))
            _git(target, ("clean", "-ffdx", "--", "."))
        _git(actual_path, ("reset", "--hard", expected))
        _git(actual_path, ("clean", "-ffdx", "--", "."))
        self.verify(record, expected_commit_sha=expected, require_clean=True)

    def _verify_identity_and_head(
        self,
        record: WorktreeRecord,
        *,
        expected_commit_sha: str,
    ) -> Tuple[Path, str]:
        """Bind a record to one registered branch and immutable HEAD."""

        expected_path = self.path_for(record.run_id, record.experiment_id)
        actual_path = Path(record.path).resolve(strict=True)
        if actual_path != expected_path:
            raise GitOperationError("WORKTREE_IDENTITY_MISMATCH", "unexpected worktree path")
        if Path(record.repository).resolve(strict=True) != self.repository:
            raise GitOperationError(
                "WORKTREE_IDENTITY_MISMATCH", "unexpected source repository"
            )
        expected_branch = experiment_branch(record.run_id, record.experiment_id)
        if record.branch != expected_branch:
            raise GitOperationError("WORKTREE_IDENTITY_MISMATCH", "unexpected branch")
        self._verify_git_admin_file(actual_path / ".git", primary_worktree=True)
        if actual_path not in self.registered_worktrees():
            raise GitOperationError(
                "UNREGISTERED_WORKTREE", f"Git does not register worktree {actual_path}"
            )

        expected = resolve_commit(self.repository, expected_commit_sha)
        current = _decode_line(
            _git(actual_path, ("rev-parse", "--verify", "HEAD^{commit}")).stdout,
            "worktree HEAD",
        )
        if current != expected:
            raise GitOperationError(
                "WORKTREE_COMMIT_MISMATCH",
                f"worktree HEAD {current} does not match expected {expected}",
            )
        checked_out_branch = _decode_line(
            _git(actual_path, ("symbolic-ref", "--short", "HEAD")).stdout,
            "worktree branch",
        )
        if checked_out_branch != expected_branch:
            raise GitOperationError(
                "WORKTREE_BRANCH_MISMATCH",
                f"worktree branch {checked_out_branch!r} is not {expected_branch!r}",
            )
        return actual_path, current

    def remove(
        self,
        record: WorktreeRecord,
        *,
        expected_commit_sha: str,
        terminal_or_safe_checkpoint: bool,
    ) -> None:
        """Remove a clean, exact worktree while preserving its evidence branch."""

        if not terminal_or_safe_checkpoint:
            raise GitOperationError(
                "UNSAFE_WORKTREE_REMOVAL",
                "worktree removal requires a terminal state or safe checkpoint",
            )
        self.verify(
            record,
            expected_commit_sha=expected_commit_sha,
            require_clean=True,
        )
        for path in self.required_submodules:
            _git(
                Path(record.path),
                ("submodule", "deinit", "--force", "--", path),
            )
        remove_args = ["worktree", "remove"]
        if self.required_submodules:
            # Git requires --force even after a clean submodule is deinitialized.
            # Identity, commit and cleanliness were verified immediately above.
            remove_args.append("--force")
        remove_args.append(str(record.path))
        _git(self.repository, tuple(remove_args))
        if Path(record.path).exists() or Path(record.path).is_symlink():
            raise GitOperationError(
                "WORKTREE_REMOVAL_FAILED", "Git left the managed worktree path behind"
            )

    def registered_worktrees(self) -> Tuple[Path, ...]:
        """Return canonical worktree paths registered by Git."""

        output = _git(self.repository, ("worktree", "list", "--porcelain")).stdout
        paths = []
        for raw_line in output.splitlines():
            if not raw_line.startswith(b"worktree "):
                continue
            raw_path = raw_line[len(b"worktree ") :]
            try:
                value = raw_path.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise GitOperationError(
                    "MALFORMED_GIT_OUTPUT", "non-UTF-8 worktree path"
                ) from exc
            paths.append(Path(value).resolve(strict=False))
        return tuple(paths)

    def _initialize_submodules(
        self, record: WorktreeRecord, commit_sha: str
    ) -> None:
        declarations = self._submodule_declarations(commit_sha)
        for path in sorted(declarations):
            name = declarations[path]
            source = self._local_submodule_gitdir(name)
            if _git(
                self.repository,
                ("--git-dir", str(source), "cat-file", "-e", f"{self._gitlink_sha(commit_sha, path)}^{{commit}}"),
                check=False,
            ).returncode != 0:
                raise GitOperationError(
                    "SUBMODULE_OBJECT_MISSING",
                    f"required local submodule commit is unavailable for {path}",
                )
            _git(
                record.path,
                (
                    "-c",
                    "protocol.file.allow=always",
                    "-c",
                    f"submodule.{name}.url={source}",
                    "submodule",
                    "update",
                    "--init",
                    "--checkout",
                    "--no-fetch",
                    "--reference",
                    str(source),
                    "--",
                    path,
                ),
            )
        self._verify_submodules(record.path, commit_sha)

    def _verify_submodules(
        self,
        worktree: Path,
        commit_sha: str,
        *,
        require_clean: bool = True,
    ) -> None:
        declarations = self._submodule_declarations(commit_sha)
        for path in sorted(declarations):
            target = worktree.joinpath(*PurePosixPath(path).parts)
            if target.is_symlink() or not target.is_dir():
                raise GitOperationError(
                    "SUBMODULE_WORKTREE_INVALID",
                    f"required submodule is not an initialized directory: {path}",
                )
            resolved = target.resolve(strict=True)
            if not _is_within(resolved, worktree.resolve(strict=True)):
                raise GitOperationError(
                    "SUBMODULE_PATH_ESCAPE", f"submodule escaped the worktree: {path}"
                )
            self._verify_git_admin_file(target / ".git", primary_worktree=False)
            expected = self._gitlink_sha(commit_sha, path)
            actual = _decode_line(
                _git(target, ("rev-parse", "--verify", "HEAD^{commit}")).stdout,
                "submodule HEAD",
            )
            if actual != expected:
                raise GitOperationError(
                    "SUBMODULE_COMMIT_MISMATCH",
                    f"submodule {path} is not at the exact reviewed gitlink commit",
                )
            if require_clean:
                status = _git(
                    target,
                    (
                        "status",
                        "--porcelain=v1",
                        "-z",
                        "--untracked-files=all",
                        "--ignored=matching",
                    ),
                ).stdout
                if status:
                    raise GitOperationError(
                        "SUBMODULE_DIRTY", f"submodule worktree is dirty: {path}"
                    )

    def _submodule_declarations(self, commit_sha: str) -> Dict[str, str]:
        commit = resolve_commit(self.repository, commit_sha)
        tree = _git(self.repository, ("ls-tree", "-r", "-z", commit)).stdout
        gitlinks: Dict[str, str] = {}
        for entry in tree.split(b"\x00"):
            if not entry:
                continue
            try:
                metadata, raw_path = entry.split(b"\t", 1)
                mode, object_type, object_id = metadata.decode("ascii").split(" ")
                path = validate_relative_path(raw_path.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise GitOperationError(
                    "MALFORMED_GIT_OUTPUT", "invalid Git tree entry"
                ) from exc
            if mode == "160000":
                if object_type != "commit":
                    raise GitOperationError(
                        "MALFORMED_GIT_OUTPUT", "gitlink does not name a commit"
                    )
                gitlinks[path] = object_id
        if tuple(sorted(gitlinks)) != self.required_submodules:
            raise GitOperationError(
                "SUBMODULE_POLICY_MISMATCH",
                "commit gitlinks differ from the configured submodule allowlist",
            )
        if not gitlinks:
            return {}

        raw_modules = _git(
            self.repository, ("show", f"{commit}:.gitmodules")
        ).stdout
        try:
            modules_text = raw_modules.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitOperationError(
                "SUBMODULE_POLICY_INVALID", ".gitmodules is not UTF-8"
            ) from exc
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        try:
            parser.read_string(modules_text)
        except configparser.Error as exc:
            raise GitOperationError(
                "SUBMODULE_POLICY_INVALID", ".gitmodules could not be parsed"
            ) from exc
        if parser.defaults():
            raise GitOperationError(
                "SUBMODULE_POLICY_INVALID", ".gitmodules cannot define default options"
            )
        declarations: Dict[str, str] = {}
        for section in parser.sections():
            match = re.fullmatch(r'submodule "([^"\r\n]+)"', section)
            if match is None or set(parser.options(section)) != {"path", "url"}:
                raise GitOperationError(
                    "SUBMODULE_POLICY_INVALID", "unreviewed .gitmodules section"
                )
            name = validate_relative_path(match.group(1))
            path = validate_relative_path(parser.get(section, "path"))
            url = parser.get(section, "url")
            if (
                not isinstance(url, str)
                or not url.strip()
                or "\x00" in url
                or any(ord(character) < 32 for character in url)
            ):
                raise GitOperationError(
                    "SUBMODULE_POLICY_INVALID", "submodule URL metadata is invalid"
                )
            if path in declarations:
                raise GitOperationError(
                    "SUBMODULE_POLICY_INVALID", "duplicate submodule path"
                )
            declarations[path] = name
        if set(declarations) != set(gitlinks):
            raise GitOperationError(
                "SUBMODULE_POLICY_MISMATCH",
                ".gitmodules paths differ from exact commit gitlinks",
            )
        return declarations

    def _gitlink_sha(self, commit_sha: str, path: str) -> str:
        line = _decode_line(
            _git(
                self.repository,
                ("ls-tree", resolve_commit(self.repository, commit_sha), "--", path),
            ).stdout,
            "gitlink",
        )
        try:
            metadata, returned_path = line.split("\t", 1)
            mode, object_type, object_id = metadata.split(" ")
        except ValueError as exc:
            raise GitOperationError("MALFORMED_GIT_OUTPUT", "invalid gitlink") from exc
        if (
            mode != "160000"
            or object_type != "commit"
            or returned_path != path
            or not re.fullmatch(r"[0-9a-f]{40,64}", object_id)
        ):
            raise GitOperationError("MALFORMED_GIT_OUTPUT", "invalid gitlink")
        return object_id

    def _local_submodule_gitdir(self, name: str) -> Path:
        raw_common = _decode_line(
            _git(self.repository, ("rev-parse", "--git-common-dir")).stdout,
            "Git common directory",
        )
        common = Path(raw_common)
        if not common.is_absolute():
            common = self.repository / common
        common = common.resolve(strict=True)
        try:
            modules_root = (common / "modules").resolve(strict=True)
        except FileNotFoundError as exc:
            raise GitOperationError(
                "SUBMODULE_OBJECT_MISSING", "local submodule object root is missing"
            ) from exc
        requested = modules_root.joinpath(*PurePosixPath(name).parts)
        try:
            source = requested.resolve(strict=True)
        except FileNotFoundError as exc:
            raise GitOperationError(
                "SUBMODULE_OBJECT_MISSING", f"local submodule object store is missing: {name}"
            ) from exc
        if (
            requested.is_symlink()
            or not source.is_dir()
            or not _is_within(source, modules_root)
        ):
            raise GitOperationError(
                "SUBMODULE_PATH_ESCAPE", "local submodule object store escaped its root"
            )
        return source

    def _verify_git_admin_file(
        self,
        dotgit: Path,
        *,
        primary_worktree: bool,
    ) -> Path:
        """Bind a writable checkout to its controller-owned Git admin directory."""

        try:
            metadata = dotgit.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (
                    hasattr(os, "getuid")
                    and metadata.st_uid != os.getuid()
                )
                or metadata.st_size > 4096
            ):
                raise GitOperationError(
                    "WORKTREE_GIT_ADMIN_INVALID",
                    "worktree .git must be a private bounded regular file",
                )
            raw = dotgit.read_bytes()
            text = raw.decode("utf-8")
        except GitOperationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise GitOperationError(
                "WORKTREE_GIT_ADMIN_INVALID", "worktree .git could not be verified"
            ) from exc
        if not text.endswith("\n") or text.count("\n") != 1 or not text.startswith("gitdir: "):
            raise GitOperationError(
                "WORKTREE_GIT_ADMIN_INVALID", "worktree .git has an invalid format"
            )
        raw_target = text[len("gitdir: ") : -1]
        target_path = Path(raw_target)
        if not target_path.is_absolute():
            target_path = dotgit.parent / target_path
        try:
            target = target_path.resolve(strict=True)
            common = self._common_git_dir()
        except OSError as exc:
            raise GitOperationError(
                "WORKTREE_GIT_ADMIN_INVALID", "worktree Git admin target is unavailable"
            ) from exc
        if not target.is_dir() or not _is_within(target, common):
            raise GitOperationError(
                "WORKTREE_GIT_ADMIN_INVALID",
                "worktree Git admin target escaped the controller repository",
            )
        worktrees_root = common / "worktrees"
        if primary_worktree and target.parent != worktrees_root:
            raise GitOperationError(
                "WORKTREE_GIT_ADMIN_INVALID",
                "primary worktree Git admin target is not controller registered",
            )
        if not primary_worktree:
            return target
        backlink = target / "gitdir"
        try:
            if backlink.is_symlink() or not backlink.is_file() or backlink.stat().st_size > 4096:
                raise GitOperationError(
                    "WORKTREE_GIT_ADMIN_INVALID", "Git admin backlink is invalid"
                )
            backlink_path = Path(backlink.read_text(encoding="utf-8").strip())
            if not backlink_path.is_absolute():
                backlink_path = target / backlink_path
            backlink_path = backlink_path.resolve(strict=True)
            canonical_dotgit = dotgit.resolve(strict=True)
        except GitOperationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise GitOperationError(
                "WORKTREE_GIT_ADMIN_INVALID", "Git admin backlink could not be verified"
            ) from exc
        if backlink_path != canonical_dotgit:
            raise GitOperationError(
                "WORKTREE_GIT_ADMIN_INVALID", "Git admin backlink names another worktree"
            )
        return target

    def _common_git_dir(self) -> Path:
        raw_common = _decode_line(
            _git(self.repository, ("rev-parse", "--git-common-dir")).stdout,
            "Git common directory",
        )
        common = Path(raw_common)
        if not common.is_absolute():
            common = self.repository / common
        return common.resolve(strict=True)

    def _rollback_failed_creation(
        self, target: Path, branch: str, created_branch: bool
    ) -> None:
        """Best-effort rollback limited to resources created by ``create``."""

        registered = False
        try:
            registered = target.resolve(strict=True) in self.registered_worktrees()
        except (FileNotFoundError, GitOperationError):
            registered = False
        if registered:
            for path in self.required_submodules:
                _git(
                    target,
                    ("submodule", "deinit", "--force", "--", path),
                    check=False,
                )
            _git(
                self.repository,
                ("worktree", "remove", "--force", str(target)),
                check=False,
            )
        if created_branch and branch_tip(self.repository, branch) is not None:
            _git(self.repository, ("branch", "-D", branch), check=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root
