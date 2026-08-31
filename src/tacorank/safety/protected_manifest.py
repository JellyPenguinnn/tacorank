"""Frozen protected-path manifests and deterministic content verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from .path_policy import normalize_policy_path, path_is_within


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# A containing directory satisfies a required child.  ``kuairand-starter-kit``
# is the official starter-kit Git submodule in the current repository layout.
MINIMUM_PROTECTED_PATHS = (
    "PROTECTED_PATHS.md",
    "contract",
    "runs",
    "src/tacorank/memory",
    "src/tacorank/orchestrator",
    "src/tacorank/safety",
    "kuairand-starter-kit/data.py",
    "kuairand-starter-kit/baseline.py",
    "kuairand-starter-kit/evaluate.py",
    "kuairand-starter-kit/submit.py",
    "kuairand-starter-kit/baseline_scores.json",
)


class ProtectedManifestError(ValueError):
    """Raised when a protected manifest is malformed or incomplete."""


@dataclass(frozen=True)
class ProtectedSnapshot:
    path: str
    kind: str
    sha256: str
    size_bytes: int

    def as_payload(self) -> dict:
        return {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ManifestVerification:
    valid: bool
    changed_paths: Tuple[str, ...]
    current_contract_sha256: str
    current_protected_content_sha256: str


@dataclass(frozen=True)
class _Gitlink:
    path: str
    commit_sha: str


@dataclass(frozen=True)
class ProtectedManifest:
    """A frozen path list plus the content identity captured at sealing time."""

    repository_root: Path
    protected_paths: Tuple[str, ...]
    snapshots: Tuple[ProtectedSnapshot, ...]
    manifest_sha256: str
    contract_paths: Tuple[str, ...]
    contract_sha256: str
    data_manifest_sha256: str
    protected_content_sha256: str

    @classmethod
    def capture(
        cls,
        repository_root: Path,
        protected_paths: Sequence[str],
        *,
        contract_paths: Sequence[str] = ("contract",),
        data_manifest_sha256: str,
        manifest_sha256: Optional[str] = None,
        require_minimum: bool = True,
    ) -> "ProtectedManifest":
        root = Path(repository_root).resolve(strict=True)
        normalized = _normalize_unique_paths(protected_paths)
        normalized_contract = _normalize_unique_paths(contract_paths)
        _validate_sha256(data_manifest_sha256, "data_manifest_sha256")
        if require_minimum:
            assert_minimum_protection(normalized)

        snapshots = tuple(_snapshot(root, path) for path in normalized)
        contract_snapshots = tuple(_snapshot(root, path) for path in normalized_contract)
        invalid_submodule_paths = sorted(
            snapshot.path
            for snapshot in (*snapshots, *contract_snapshots)
            if snapshot.kind in {
                "uninitialized_submodule",
                "missing_submodule_path",
            }
        )
        if invalid_submodule_paths:
            raise ProtectedManifestError(
                "cannot freeze protected paths from an uninitialized or incomplete submodule: {}".format(
                    ", ".join(invalid_submodule_paths)
                )
            )
        paths_payload = {
            "schema_version": "1.0",
            "protected_paths": list(normalized),
        }
        frozen_manifest_sha = manifest_sha256 or _hash_json(paths_payload)
        _validate_sha256(frozen_manifest_sha, "manifest_sha256")
        return cls(
            repository_root=root,
            protected_paths=normalized,
            snapshots=snapshots,
            manifest_sha256=frozen_manifest_sha,
            contract_paths=normalized_contract,
            contract_sha256=_hash_snapshots(contract_snapshots),
            data_manifest_sha256=data_manifest_sha256,
            protected_content_sha256=_hash_snapshots(snapshots),
        )

    @classmethod
    def from_markdown(
        cls,
        manifest_path: Path,
        repository_root: Path,
        *,
        contract_paths: Sequence[str] = ("contract",),
        data_manifest_sha256: str,
        expected_manifest_sha256: Optional[str] = None,
        require_minimum: bool = True,
    ) -> "ProtectedManifest":
        path = Path(manifest_path)
        raw = path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if expected_manifest_sha256 is not None:
            _validate_sha256(expected_manifest_sha256, "expected_manifest_sha256")
            if actual_sha != expected_manifest_sha256:
                raise ProtectedManifestError("protected manifest hash does not match")
        entries = parse_protected_paths_markdown(raw.decode("utf-8"))
        return cls.capture(
            repository_root,
            entries,
            contract_paths=contract_paths,
            data_manifest_sha256=data_manifest_sha256,
            manifest_sha256=actual_sha,
            require_minimum=require_minimum,
        )

    def verify(self, repository_root: Optional[Path] = None) -> ManifestVerification:
        root = (
            self.repository_root
            if repository_root is None
            else Path(repository_root).resolve(strict=True)
        )
        current = tuple(_snapshot(root, item.path) for item in self.snapshots)
        expected_by_path = {item.path: item for item in self.snapshots}
        changed = tuple(
            item.path
            for item in current
            if item != expected_by_path[item.path]
        )
        current_contract = tuple(_snapshot(root, path) for path in self.contract_paths)
        contract_sha = _hash_snapshots(current_contract)
        protected_sha = _hash_snapshots(current)
        return ManifestVerification(
            valid=(
                not changed
                and contract_sha == self.contract_sha256
                and protected_sha == self.protected_content_sha256
            ),
            changed_paths=changed,
            current_contract_sha256=contract_sha,
            current_protected_content_sha256=protected_sha,
        )


def parse_protected_paths_markdown(text: str) -> Tuple[str, ...]:
    """Extract path-only bullet items from ``PROTECTED_PATHS.md``.

    Accepted forms are ``- `path/` `` and ``- path/``.  Explanatory prose is
    intentionally not interpreted as policy.
    """

    entries = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("<!--"):
            continue
        match = re.match(r"^[-*]\s+(.+?)\s*$", stripped)
        if match is None:
            continue
        candidate = match.group(1)
        if candidate.startswith("`") and candidate.endswith("`"):
            candidate = candidate[1:-1]
        elif any(character.isspace() for character in candidate):
            continue
        try:
            normalized = normalize_policy_path(candidate.rstrip("/"))
        except ValueError as exc:
            raise ProtectedManifestError(
                "invalid protected path {!r}: {}".format(candidate, exc)
            )
        entries.append(normalized)
    if not entries:
        raise ProtectedManifestError("protected manifest contains no path entries")
    return tuple(sorted(set(entries)))


def assert_minimum_protection(protected_paths: Sequence[str]) -> None:
    normalized = _normalize_unique_paths(protected_paths)
    missing = []
    for required in MINIMUM_PROTECTED_PATHS:
        if not any(path_is_within(required, protected) for protected in normalized):
            missing.append(required)
    if missing:
        raise ProtectedManifestError(
            "protected manifest omits mandatory paths: {}".format(", ".join(missing))
        )


def _snapshot(repository_root: Path, relative_path: str) -> ProtectedSnapshot:
    normalized = normalize_policy_path(relative_path)
    target = repository_root.joinpath(*normalized.split("/"))
    gitlink = _gitlink_ancestor(repository_root, normalized)
    if gitlink is not None and not _gitlink_checkout_matches(repository_root, gitlink):
        return ProtectedSnapshot(
            normalized,
            "uninitialized_submodule",
            _hash_json(
                {
                    "gitlink_path": gitlink.path,
                    "gitlink_commit": gitlink.commit_sha,
                }
            ),
            0,
        )
    if target.is_symlink():
        raise ProtectedManifestError(
            "protected path cannot be a symbolic link: {}".format(normalized)
        )
    if not target.exists():
        if gitlink is not None:
            return ProtectedSnapshot(
                normalized,
                "missing_submodule_path",
                _hash_json(
                    {
                        "gitlink_path": gitlink.path,
                        "gitlink_commit": gitlink.commit_sha,
                    }
                ),
                0,
            )
        # Missing protected locations are still sealed: creating one changes the
        # snapshot from ``missing`` and therefore invalidates verification.
        return ProtectedSnapshot(normalized, "missing", hashlib.sha256(b"").hexdigest(), 0)
    if target.is_file():
        if gitlink is not None:
            submodule_root = repository_root.joinpath(*gitlink.path.split("/"))
            submodule_path = normalized[len(gitlink.path) :].lstrip("/")
            canonical = _hash_git_worktree_file(submodule_root, submodule_path)
            digest, size = canonical or _hash_file(target)
            digest = _hash_json(
                {
                    "gitlink_path": gitlink.path,
                    "gitlink_commit": gitlink.commit_sha,
                    "content_sha256": digest,
                }
            )
            return ProtectedSnapshot(normalized, "submodule_file", digest, size)
        tracked = _tracked_files_within(repository_root, normalized)
        if tracked == (normalized,):
            canonical = _hash_git_worktree_file(repository_root, normalized)
            if canonical is not None:
                digest, size = canonical
                return ProtectedSnapshot(normalized, "file", digest, size)
        digest, size = _hash_file(target)
        return ProtectedSnapshot(normalized, "file", digest, size)
    if not target.is_dir():
        raise ProtectedManifestError("unsupported protected path type: {}".format(normalized))

    entries = []
    total_size = 0
    tracked_files = (
        None
        if gitlink is not None
        else _tracked_files_within(repository_root, normalized)
    )
    if tracked_files:
        # Controller-owned append-only and ignored runtime files (for example,
        # run ledgers and bytecode) are not present in experiment worktrees.
        # Freeze the Git-indexed tree here; worktree cleanliness checks reject
        # both untracked and ignored candidate additions separately.
        for relative in tracked_files:
            child = repository_root.joinpath(*relative.split("/"))
            if child.is_symlink():
                raise ProtectedManifestError(
                    "protected tree contains symbolic link: {}".format(relative)
                )
            if not child.exists():
                entries.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                        "size_bytes": 0,
                        "state": "missing",
                    }
                )
                continue
            if not child.is_file():
                raise ProtectedManifestError(
                    "tracked protected path is not a regular file: {}".format(relative)
                )
            canonical = _hash_git_worktree_file(repository_root, relative)
            digest, size = canonical or _hash_file(child)
            total_size += size
            entries.append({"path": relative, "sha256": digest, "size_bytes": size})
        return ProtectedSnapshot(
            normalized,
            "tracked_directory",
            _hash_json(entries),
            total_size,
        )

    for directory, directory_names, file_names in os.walk(str(target), followlinks=False):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        if gitlink is not None and directory_path == target:
            directory_names[:] = [name for name in directory_names if name != ".git"]
            file_names[:] = [name for name in file_names if name != ".git"]
        for name in directory_names:
            child = directory_path / name
            if child.is_symlink():
                relative = child.relative_to(repository_root).as_posix()
                raise ProtectedManifestError(
                    "protected tree contains symbolic link: {}".format(relative)
                )
        for name in file_names:
            child = directory_path / name
            relative = child.relative_to(repository_root).as_posix()
            if child.is_symlink():
                raise ProtectedManifestError(
                    "protected tree contains symbolic link: {}".format(relative)
                )
            digest, size = _hash_file(child)
            total_size += size
            entries.append({"path": relative, "sha256": digest, "size_bytes": size})
    digest = _hash_json(entries)
    if gitlink is not None:
        digest = _hash_json(
            {
                "gitlink_path": gitlink.path,
                "gitlink_commit": gitlink.commit_sha,
                "tree_sha256": digest,
            }
        )
        return ProtectedSnapshot(normalized, "submodule_directory", digest, total_size)
    return ProtectedSnapshot(normalized, "directory", digest, total_size)


def _tracked_files_within(
    repository_root: Path,
    relative_path: str,
) -> Optional[Tuple[str, ...]]:
    encoded = _run_git(
        repository_root,
        ("ls-files", "-z", "--", relative_path),
    )
    if encoded is None:
        return None
    files = []
    for item in encoded.split(b"\x00"):
        if not item:
            continue
        try:
            decoded = item.decode("utf-8", errors="strict")
            normalized = normalize_policy_path(decoded)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProtectedManifestError(
                "Git returned an invalid protected path"
            ) from exc
        if path_is_within(normalized, relative_path):
            files.append(normalized)
    return tuple(sorted(set(files)))


def _gitlink_ancestor(repository_root: Path, relative_path: str) -> Optional[_Gitlink]:
    parts = relative_path.split("/")
    for length in range(1, len(parts) + 1):
        prefix = "/".join(parts[:length])
        completed = _run_git(
            repository_root,
            ("ls-files", "--stage", "-z", "--", prefix),
        )
        if completed is None:
            return None
        for record in completed.split(b"\x00"):
            if not record:
                continue
            try:
                metadata, encoded_path = record.split(b"\t", 1)
                mode, object_id, stage = metadata.split()
                indexed_path = encoded_path.decode("utf-8", errors="strict")
                commit_sha = object_id.decode("ascii", errors="strict")
            except (ValueError, UnicodeDecodeError):
                continue
            if indexed_path == prefix and mode == b"160000" and stage == b"0":
                return _Gitlink(prefix, commit_sha)
    return None


def _gitlink_checkout_matches(repository_root: Path, gitlink: _Gitlink) -> bool:
    checkout = repository_root.joinpath(*gitlink.path.split("/"))
    marker = checkout / ".git"
    if (
        checkout.is_symlink()
        or not checkout.is_dir()
        or marker.is_symlink()
        or not marker.exists()
    ):
        return False
    head = _run_git(checkout, ("rev-parse", "--verify", "HEAD^{commit}"))
    if head is None:
        return False
    try:
        decoded = head.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError:
        return False
    return decoded == gitlink.commit_sha


def _run_git(repository: Path, arguments: Sequence[str]) -> Optional[bytes]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "protocol.file.allow=never",
                "-C",
                str(repository),
                *arguments,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _hash_git_worktree_file(
    repository_root: Path,
    relative_path: str,
) -> Optional[Tuple[str, int]]:
    """Hash tracked content through Git's clean-filter identity.

    A Windows checkout may contain CRLF while a Docker or macOS worktree for
    the same index content contains LF.  Bind such non-binary, line-ending-only
    variants to the indexed bytes rather than to a platform-specific checkout
    representation.  Any other worktree edit is hashed as-is and therefore
    still fails verification.
    """

    try:
        normalized = normalize_policy_path(relative_path)
    except ValueError:
        return None
    indexed = _run_git(repository_root, ("show", ":" + normalized))
    if indexed is None:
        return None
    try:
        worktree = repository_root.joinpath(*normalized.split("/")).read_bytes()
    except OSError:
        return None
    canonical = worktree
    if worktree == indexed or _text_line_endings_equivalent(worktree, indexed):
        canonical = indexed
    return hashlib.sha256(canonical).hexdigest(), len(canonical)


def _text_line_endings_equivalent(left: bytes, right: bytes) -> bool:
    """Return whether two non-binary byte strings differ only by CRLF/LF."""

    if b"\x00" in left or b"\x00" in right:
        return False
    return left.replace(b"\r\n", b"\n") == right.replace(b"\r\n", b"\n")


def _hash_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _hash_snapshots(snapshots: Iterable[ProtectedSnapshot]) -> str:
    return _hash_json([snapshot.as_payload() for snapshot in snapshots])


def _hash_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_unique_paths(values: Sequence[str]) -> Tuple[str, ...]:
    if not values:
        raise ProtectedManifestError("path list cannot be empty")
    normalized = []
    for value in values:
        try:
            path = normalize_policy_path(value.rstrip("/"))
        except ValueError as exc:
            raise ProtectedManifestError("invalid path {!r}: {}".format(value, exc))
        if path not in normalized:
            normalized.append(path)
    return tuple(sorted(normalized))


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProtectedManifestError("{} must be a lowercase SHA-256".format(field_name))
