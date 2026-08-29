"""Exact Git patch capture and single-parent commit sealing."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple

from .refs import GitOperationError, _decode_line, _git, require_ancestor, resolve_commit


@dataclass(frozen=True)
class NormalizedPatch:
    """Canonical bytes and identity for one candidate change."""

    base_commit_sha: str
    diff: bytes
    diff_sha256: str
    changed_files: Tuple[str, ...]
    patch_commit_sha: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.diff


@dataclass(frozen=True)
class WrittenArtifact:
    """Hash-addressed artifact metadata before shared-schema conversion."""

    path: str
    sha256: str
    size_bytes: int
    content_type: str


def stage_and_capture(
    worktree: Path,
    base_commit_sha: str,
    *,
    max_diff_bytes: Optional[int] = None,
) -> NormalizedPatch:
    """Stage the full worktree change and capture its exact canonical diff.

    Staging occurs only after verifying that ``HEAD`` is the supplied immutable
    parent.  It brings tracked, untracked, and deleted files into one snapshot.
    """

    root = _worktree_root(worktree)
    base = _require_exact_head(root, base_commit_sha)
    _preflight_changed_file_bytes(root, max_diff_bytes)
    _git(root, ("add", "--all", "--", "."))
    return _capture_cached(root, base, max_diff_bytes=max_diff_bytes)


def capture_commit_patch(
    worktree: Path,
    base_commit_sha: str,
    patch_commit_sha: str,
    *,
    max_diff_bytes: Optional[int] = None,
) -> NormalizedPatch:
    """Capture a sealed direct child commit and verify its parent relationship."""

    root = _worktree_root(worktree)
    base = resolve_commit(root, base_commit_sha)
    patch_commit = resolve_commit(root, patch_commit_sha)
    parent_line = _decode_line(
        _git(root, ("rev-list", "--parents", "-n", "1", patch_commit)).stdout,
        "commit parents",
    )
    parents = parent_line.split(" ")
    if parents != [patch_commit, base]:
        raise GitOperationError(
            "PATCH_PARENT_MISMATCH",
            "patch commit must have exactly the declared base as its parent",
        )
    diff = _diff_bytes(
        root, base, patch_commit, cached=False, max_diff_bytes=max_diff_bytes
    )
    changed_files = _changed_files(root, base, patch_commit, cached=False)
    _reject_submodule_changes(root, base, patch_commit, cached=False)
    return NormalizedPatch(
        base_commit_sha=base,
        patch_commit_sha=patch_commit,
        diff=diff,
        diff_sha256=hashlib.sha256(diff).hexdigest(),
        changed_files=changed_files,
    )


def capture_commit_range(
    repository: Path,
    root_commit_sha: str,
    current_commit_sha: str,
    *,
    max_diff_bytes: Optional[int] = None,
) -> NormalizedPatch:
    """Capture the canonical cumulative patch across a verified commit range.

    Unlike :func:`capture_commit_patch`, this helper intentionally permits
    intermediate commits while requiring the current commit to descend from
    the declared experiment root.
    """

    root = _worktree_root(repository)
    base = resolve_commit(root, root_commit_sha)
    current = resolve_commit(root, current_commit_sha)
    require_ancestor(root, base, current)
    diff = _diff_bytes(
        root, base, current, cached=False, max_diff_bytes=max_diff_bytes
    )
    changed_files = _changed_files(root, base, current, cached=False)
    _reject_submodule_changes(root, base, current, cached=False)
    return NormalizedPatch(
        base_commit_sha=base,
        patch_commit_sha=current,
        diff=diff,
        diff_sha256=hashlib.sha256(diff).hexdigest(),
        changed_files=changed_files,
    )


def commit_staged_patch(
    worktree: Path,
    expected_patch: NormalizedPatch,
    *,
    message: str,
    author_name: str = "TacoRank Coding Worker",
    author_email: str = "tacorank@invalid",
) -> NormalizedPatch:
    """Commit only the already captured bytes and return their sealed identity."""

    root = _worktree_root(worktree)
    _require_exact_head(root, expected_patch.base_commit_sha)
    if expected_patch.patch_commit_sha is not None:
        raise GitOperationError("PATCH_ALREADY_COMMITTED", "patch already has a commit")
    if expected_patch.is_empty:
        raise GitOperationError("EMPTY_PATCH", "must-patch task produced no diff")
    if not isinstance(message, str) or not message.strip() or "\x00" in message:
        raise GitOperationError("INVALID_COMMIT_MESSAGE", "invalid commit message")

    current = _capture_cached(
        root,
        expected_patch.base_commit_sha,
        max_diff_bytes=max(len(expected_patch.diff) * 2, len(expected_patch.diff) + 4096),
    )
    if current.diff != expected_patch.diff or current.changed_files != expected_patch.changed_files:
        raise GitOperationError(
            "PATCH_SUBSTITUTION",
            "staged patch changed after it was captured",
        )

    args = (
        "-c",
        f"user.name={author_name}",
        "-c",
        f"user.email={author_email}",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--no-gpg-sign",
        "--no-verify",
        "-m",
        message.strip(),
    )
    _git(root, args)
    commit_sha = _decode_line(
        _git(root, ("rev-parse", "--verify", "HEAD^{commit}")).stdout,
        "patch commit",
    )
    sealed = capture_commit_patch(
        root,
        expected_patch.base_commit_sha,
        commit_sha,
        max_diff_bytes=len(expected_patch.diff),
    )
    if (
        sealed.diff != expected_patch.diff
        or sealed.diff_sha256 != expected_patch.diff_sha256
        or sealed.changed_files != expected_patch.changed_files
    ):
        raise GitOperationError(
            "PATCH_SUBSTITUTION",
            "committed patch bytes do not match the verified staged bytes",
        )
    status = _git(
        root, ("status", "--porcelain=v1", "-z", "--untracked-files=all")
    ).stdout
    if status:
        raise GitOperationError("WORKTREE_DIRTY", "worktree remained dirty after commit")
    return sealed


def write_artifact(
    repository_root: Path,
    relative_path: str,
    content: bytes,
    *,
    content_type: str,
) -> WrittenArtifact:
    """Write immutable artifact bytes below an explicit repository root."""

    root = Path(repository_root).resolve(strict=True)
    normalized = validate_relative_path(relative_path)
    destination = root.joinpath(*PurePosixPath(normalized).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = destination.parent.resolve(strict=True)
    if not _is_within_or_equal(resolved_parent, root):
        raise GitOperationError("ARTIFACT_PATH_ESCAPE", "artifact parent escaped its root")
    destination = resolved_parent / destination.name
    digest = hashlib.sha256(content).hexdigest()

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise GitOperationError(
                "ARTIFACT_PATH_COLLISION", "artifact destination is not a regular file"
            )
        existing = destination.read_bytes()
        if existing != content:
            raise GitOperationError(
                "ARTIFACT_PATH_COLLISION", "artifact path already contains different bytes"
            )
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(destination, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            raise

    return WrittenArtifact(
        path=normalized,
        sha256=digest,
        size_bytes=len(content),
        content_type=content_type,
    )


def validate_relative_path(path: str) -> str:
    """Validate and normalize a repository-relative POSIX path."""

    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise GitOperationError("INVALID_RELATIVE_PATH", "invalid relative path")
    if any(ord(character) < 32 for character in path):
        raise GitOperationError("INVALID_RELATIVE_PATH", "control character in path")
    value = PurePosixPath(path)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise GitOperationError("INVALID_RELATIVE_PATH", "path is not normalized and relative")
    normalized = value.as_posix()
    if normalized != path:
        raise GitOperationError("INVALID_RELATIVE_PATH", "path is not normalized")
    return normalized


def _capture_cached(
    root: Path,
    base: str,
    *,
    max_diff_bytes: Optional[int] = None,
) -> NormalizedPatch:
    diff = _diff_bytes(
        root, base, None, cached=True, max_diff_bytes=max_diff_bytes
    )
    changed_files = _changed_files(root, base, None, cached=True)
    _reject_submodule_changes(root, base, None, cached=True)
    return NormalizedPatch(
        base_commit_sha=base,
        diff=diff,
        diff_sha256=hashlib.sha256(diff).hexdigest(),
        changed_files=changed_files,
    )


def _diff_bytes(
    root: Path,
    base: str,
    target: Optional[str],
    *,
    cached: bool,
    max_diff_bytes: Optional[int] = None,
) -> bytes:
    args = [
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
    ]
    if cached:
        args.append("--cached")
        args.append(base)
    else:
        if target is None:
            raise AssertionError("target is required for a committed diff")
        args.extend((base, target))
    args.append("--")
    return _git(root, tuple(args), max_stdout_bytes=max_diff_bytes).stdout


def _preflight_changed_file_bytes(root: Path, max_diff_bytes: Optional[int]) -> None:
    if max_diff_bytes is None:
        return
    if (
        isinstance(max_diff_bytes, bool)
        or not isinstance(max_diff_bytes, int)
        or max_diff_bytes < 1
    ):
        raise GitOperationError("INVALID_GIT_BOUND", "max_diff_bytes must be positive")
    status = _git(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        max_stdout_bytes=max(max_diff_bytes, 64 * 1024),
    ).stdout
    records = status.split(b"\x00")
    total = 0
    index = 0
    while index < len(records):
        entry = records[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2:3] != b" ":
            raise GitOperationError("MALFORMED_GIT_OUTPUT", "invalid porcelain status")
        candidates = [entry[3:]]
        if b"R" in entry[:2] or b"C" in entry[:2]:
            if index >= len(records) or not records[index]:
                raise GitOperationError("MALFORMED_GIT_OUTPUT", "rename source is missing")
            candidates.append(records[index])
            index += 1
        for raw_path in candidates:
            try:
                relative = validate_relative_path(raw_path.decode("utf-8"))
                metadata = root.joinpath(*PurePosixPath(relative).parts).lstat()
            except FileNotFoundError:
                continue
            except (UnicodeDecodeError, OSError) as exc:
                raise GitOperationError(
                    "MALFORMED_GIT_OUTPUT", "changed path could not be inspected"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                raise GitOperationError(
                    "UNSUPPORTED_WORKTREE_ENTRY", "candidate created a special file"
                )
            total += metadata.st_size
            if total > max_diff_bytes:
                raise GitOperationError(
                    "PATCH_TOO_LARGE", "changed file bytes exceed the patch limit"
                )


def _changed_files(
    root: Path, base: str, target: Optional[str], *, cached: bool
) -> Tuple[str, ...]:
    args = ["diff", "--name-only", "-z", "--no-renames"]
    if cached:
        args.extend(("--cached", base))
    else:
        if target is None:
            raise AssertionError("target is required for committed paths")
        args.extend((base, target))
    args.append("--")
    output = _git(root, tuple(args)).stdout
    raw_paths = output.split(b"\x00")
    if raw_paths and raw_paths[-1] == b"":
        raw_paths.pop()
    paths = []
    for raw_path in raw_paths:
        try:
            decoded = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitOperationError(
                "NON_UTF8_PATH", "candidate patch contains a non-UTF-8 path"
            ) from exc
        paths.append(validate_relative_path(decoded))
    if len(paths) != len(set(paths)):
        raise GitOperationError(
            "MALFORMED_DIFF_PATHS", "Git returned duplicate changed paths"
        )
    return tuple(sorted(paths))


def _reject_submodule_changes(
    root: Path, base: str, target: Optional[str], *, cached: bool
) -> None:
    args = ["diff", "--raw", "--no-abbrev", "-z", "--no-renames"]
    if cached:
        args.extend(("--cached", base))
    else:
        if target is None:
            raise AssertionError("target is required for committed raw diff")
        args.extend((base, target))
    args.append("--")
    entries = _git(root, tuple(args)).stdout.split(b"\x00")
    index = 0
    while index < len(entries):
        header = entries[index]
        index += 1
        if not header:
            continue
        if index >= len(entries):
            raise GitOperationError("MALFORMED_GIT_OUTPUT", "raw diff path is missing")
        index += 1  # no-renames guarantees exactly one following path
        try:
            fields = header[1:].decode("ascii").split(" ")
        except UnicodeDecodeError as exc:
            raise GitOperationError("MALFORMED_GIT_OUTPUT", "invalid raw diff") from exc
        if not header.startswith(b":") or len(fields) != 5:
            raise GitOperationError("MALFORMED_GIT_OUTPUT", "invalid raw diff")
        if "160000" in fields[:2]:
            raise GitOperationError(
                "SUBMODULE_UPDATE_FORBIDDEN",
                "candidate patches cannot add, remove, or move a submodule gitlink",
            )


def _require_exact_head(root: Path, expected_commit_sha: str) -> str:
    expected = resolve_commit(root, expected_commit_sha)
    actual = _decode_line(
        _git(root, ("rev-parse", "--verify", "HEAD^{commit}")).stdout,
        "worktree HEAD",
    )
    if actual != expected:
        raise GitOperationError(
            "WORKTREE_COMMIT_MISMATCH",
            f"worktree HEAD {actual} does not match declared base {expected}",
        )
    return expected


def _worktree_root(path: Path) -> Path:
    candidate = Path(path).resolve(strict=True)
    root = Path(
        _decode_line(
            _git(candidate, ("rev-parse", "--show-toplevel")).stdout,
            "worktree root",
        )
    ).resolve(strict=True)
    if candidate != root:
        raise GitOperationError("NOT_WORKTREE_ROOT", "expected exact worktree root")
    return root


def _is_within_or_equal(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
