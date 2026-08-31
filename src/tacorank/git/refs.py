"""Validated Git references and ancestry checks for experiment branches.

This module deliberately exposes a small, argv-only Git surface.  Callers pass
resolved repository paths and immutable commit object IDs; arbitrary revision
expressions and shell command strings are not accepted.
"""

from __future__ import annotations

import os
import re
import selectors
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class GitOperationError(RuntimeError):
    """A safe, classified failure from a bounded Git operation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def validated_repository(path: Path) -> Path:
    """Return the canonical repository root, rejecting non-repositories."""

    candidate = Path(path).resolve(strict=True)
    result = _git(candidate, ("rev-parse", "--show-toplevel"))
    root_text = _decode_line(result.stdout, "repository root")
    root = Path(root_text).resolve(strict=True)
    if root != candidate:
        raise GitOperationError(
            "NOT_REPOSITORY_ROOT",
            f"expected a Git repository root, got {candidate}",
        )
    return root


def validate_identifier(value: str, field: str) -> str:
    """Validate a run/experiment identifier before using it in a ref or path."""

    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise GitOperationError(
            "INVALID_IDENTIFIER",
            f"{field} must match {_IDENTIFIER_RE.pattern}",
        )
    if value in {".", ".."} or value.endswith(".lock") or ".." in value:
        raise GitOperationError("INVALID_IDENTIFIER", f"invalid {field}")
    return value


def experiment_branch(run_id: str, experiment_id: str) -> str:
    """Return the only branch namespace used for a candidate experiment."""

    run = validate_identifier(run_id, "run_id")
    experiment = validate_identifier(experiment_id, "experiment_id")
    return f"experiment/{run}/{experiment}"


def best_ref(run_id: str) -> str:
    """Return Person 2's derived best-candidate ref name."""

    return f"best/{validate_identifier(run_id, 'run_id')}"


def validate_object_id(value: str, field: str = "commit_sha") -> str:
    """Accept only a full lowercase Git object ID, never a rev expression."""

    if not isinstance(value, str) or not _OBJECT_ID_RE.fullmatch(value):
        raise GitOperationError(
            "INVALID_OBJECT_ID",
            f"{field} must be a full lowercase Git object ID",
        )
    return value


def resolve_commit(repository: Path, object_id: str) -> str:
    """Resolve and verify that *object_id* names a commit object exactly."""

    repo = Path(repository).resolve(strict=True)
    expected = validate_object_id(object_id)
    result = _git(repo, ("rev-parse", "--verify", f"{expected}^{{commit}}"))
    resolved = _decode_line(result.stdout, "commit object ID")
    if resolved != expected:
        raise GitOperationError(
            "OBJECT_ID_MISMATCH",
            f"Git resolved {expected} to a different object ID",
        )
    return resolved


def read_blob_at_commit(
    repository: Path,
    commit_sha: str,
    relative_path: str,
    *,
    max_bytes: int = 4 * 1024 * 1024,
) -> bytes:
    """Read one bounded repository file from an exact immutable commit.

    The revision and path are validated separately before being joined into
    Git's ``<commit>:<path>`` object syntax. This keeps callers from deriving
    integrity identities from a mutable checkout while preserving the module's
    argv-only Git boundary.
    """

    if not isinstance(relative_path, str) or not relative_path:
        raise GitOperationError("INVALID_BLOB_PATH", "blob path must be non-empty")
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or relative_path != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise GitOperationError(
            "INVALID_BLOB_PATH", "blob path must be a normalized relative path"
        )
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
    ):
        raise GitOperationError("INVALID_GIT_BOUND", "invalid blob size bound")

    repo = Path(repository).resolve(strict=True)
    commit = resolve_commit(repo, commit_sha)
    result = _git(
        repo,
        ("cat-file", "blob", "%s:%s" % (commit, relative_path)),
        max_stdout_bytes=max_bytes,
    )
    return result.stdout


def branch_tip(repository: Path, branch: str) -> Optional[str]:
    """Return a local branch tip, or ``None`` when the branch is absent."""

    _validate_ref_name(branch)
    repo = Path(repository).resolve(strict=True)
    ref = f"refs/heads/{branch}"
    result = _git(
        repo,
        ("for-each-ref", "--format=%(objectname)", "--count=1", ref),
    )
    if not result.stdout.strip():
        return None
    tip = _decode_line(result.stdout, "branch tip")
    return validate_object_id(tip, "branch tip")


def is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    """Return whether two exact commits have the requested ancestry."""

    repo = Path(repository).resolve(strict=True)
    older = resolve_commit(repo, ancestor)
    newer = resolve_commit(repo, descendant)
    result = _git(repo, ("merge-base", "--is-ancestor", older, newer), check=False)
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    _raise_git_failure(("merge-base", "--is-ancestor", older, newer), result)
    raise AssertionError("unreachable")


def require_ancestor(repository: Path, ancestor: str, descendant: str) -> None:
    """Raise when *descendant* does not descend from *ancestor*."""

    if not is_ancestor(repository, ancestor, descendant):
        raise GitOperationError(
            "ANCESTRY_MISMATCH",
            f"commit {descendant} does not descend from {ancestor}",
        )


def update_best_ref(
    repository: Path,
    run_id: str,
    commit_sha: str,
    *,
    expected_old_sha: Optional[str] = None,
) -> None:
    """Atomically update Person 2's derived ``best/<run_id>`` pointer.

    Supplying ``expected_old_sha`` provides compare-and-swap behavior.  This
    function is intentionally not used by the coding adapter.
    """

    repo = Path(repository).resolve(strict=True)
    target = resolve_commit(repo, commit_sha)
    ref = f"refs/heads/{best_ref(run_id)}"
    args = ["update-ref", ref, target]
    if expected_old_sha is not None:
        args.append(resolve_commit(repo, expected_old_sha))
    _git(repo, tuple(args))


def _validate_ref_name(ref: str) -> None:
    if not isinstance(ref, str) or not ref:
        raise GitOperationError("INVALID_REF", "Git ref must be a non-empty string")
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", ref],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        env=_git_environment(),
    )
    if result.returncode != 0:
        raise GitOperationError("INVALID_REF", f"invalid Git branch name: {ref!r}")


@dataclass(frozen=True)
class _RawGitResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def _git(
    repository: Path,
    args: Sequence[str],
    *,
    check: bool = True,
    input_bytes: Optional[bytes] = None,
    max_stdout_bytes: Optional[int] = None,
) -> _RawGitResult:
    """Run Git without a shell and with stable non-interactive behavior."""

    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "protocol.file.allow=never",
        "-C",
        str(repository),
        *args,
    ]
    if max_stdout_bytes is None:
        result = subprocess.run(
            command,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            env=_git_environment(),
        )
        raw = _RawGitResult(result.returncode, result.stdout, result.stderr)
    else:
        if (
            isinstance(max_stdout_bytes, bool)
            or not isinstance(max_stdout_bytes, int)
            or max_stdout_bytes < 1
            or input_bytes is not None
        ):
            raise GitOperationError("INVALID_GIT_BOUND", "invalid bounded Git output request")
        raw = _run_git_bounded(command, max_stdout_bytes=max_stdout_bytes)
    if check and raw.returncode != 0:
        _raise_git_failure(args, raw)
    return raw


def _run_git_bounded(command: Sequence[str], *, max_stdout_bytes: int) -> _RawGitResult:
    """Capture local Git output without an unbounded in-memory pipe."""

    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=_git_environment(),
        start_new_session=True,
        close_fds=True,
    )
    if process.stdout is None or process.stderr is None:
        raise AssertionError("bounded Git pipes were not created")
    if os.name == "nt":
        # ``selectors.SelectSelector`` cannot poll Windows pipe handles. Read
        # each stream with a strict cap on a worker thread instead.
        def read_limited(stream, limit: int) -> bytes:
            return stream.read(limit + 1)

        with ThreadPoolExecutor(max_workers=2) as pool:
            stdout_future = pool.submit(read_limited, process.stdout, max_stdout_bytes)
            stderr_future = pool.submit(read_limited, process.stderr, 64 * 1024)
            try:
                stdout = stdout_future.result(timeout=60.0)
                stderr = stderr_future.result(timeout=60.0)
            except FutureTimeout as exc:
                _kill_git_process(process)
                raise GitOperationError(
                    "GIT_COMMAND_TIMEOUT", "bounded Git command timed out"
                ) from exc
        returncode = process.wait()
        if len(stdout) > max_stdout_bytes:
            raise GitOperationError("PATCH_TOO_LARGE", "bounded Git output exceeded its limit")
        if len(stderr) > 64 * 1024:
            raise GitOperationError("GIT_OUTPUT_LIMIT", "bounded Git output exceeded its limit")
        return _RawGitResult(returncode, stdout, stderr)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    stdout = bytearray()
    stderr = bytearray()
    deadline = time.monotonic() + 60.0
    try:
        while selector.get_map():
            if time.monotonic() >= deadline:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise GitOperationError("GIT_COMMAND_TIMEOUT", "bounded Git command timed out")
            for key, _ in selector.select(timeout=0.05):
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                destination = stdout if key.data == "stdout" else stderr
                destination.extend(chunk)
                limit = max_stdout_bytes if key.data == "stdout" else 64 * 1024
                if len(destination) > limit:
                    _kill_git_process(process)
                    code = "PATCH_TOO_LARGE" if key.data == "stdout" else "GIT_OUTPUT_LIMIT"
                    raise GitOperationError(code, "bounded Git output exceeded its limit")
        return _RawGitResult(process.wait(), bytes(stdout), bytes(stderr))
    finally:
        selector.close()
        if process.poll() is None:
            _kill_git_process(process)


def _kill_git_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError, AttributeError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
    process.wait()


def _raise_git_failure(args: Sequence[str], result: _RawGitResult) -> None:
    detail = result.stderr.decode("utf-8", errors="replace").strip()
    if len(detail) > 600:
        detail = detail[:600] + "..."
    operation = args[0] if args else "unknown"
    raise GitOperationError(
        "GIT_COMMAND_FAILED",
        f"git {operation} failed with exit code {result.returncode}: {detail}",
    )


def _decode_line(data: bytes, label: str) -> str:
    try:
        value = data.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise GitOperationError("MALFORMED_GIT_OUTPUT", f"non-ASCII {label}") from exc
    if not value or "\n" in value or "\r" in value:
        raise GitOperationError("MALFORMED_GIT_OUTPUT", f"invalid {label}")
    return value


def _git_environment() -> dict[str, str]:
    """Return a credential-free environment for local-only Git mechanics."""

    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
