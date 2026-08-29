"""Execution-test artifact service; production ownership belongs to Person 2."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from tacorank.execution.interfaces import ModelFactory, sha256_file


class ExecutionTestArtifactError(RuntimeError):
    pass


class TestArtifactStore:
    """Small immutable filesystem adapter implementing ExecutionArtifactStore."""

    __test__ = False

    def __init__(
        self,
        repository_root: Path,
        artifact_root: Path,
        *,
        model_factory: ModelFactory,
    ) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)
        root = Path(artifact_root)
        if not root.is_absolute():
            root = self.repository_root / root
        _reject_symlink_components(root)
        self.artifact_root = root.resolve(strict=False)
        _within(self.artifact_root, self.repository_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(self.artifact_root)
        self.model_factory = model_factory

    def attempt_directory(
        self, run_id: str, experiment_id: str, attempt: int
    ) -> Path:
        if attempt < 1:
            raise ExecutionTestArtifactError("attempt must be at least one")
        path = self.artifact_root / _identifier(run_id) / _identifier(experiment_id)
        path = path / "attempt_{0}".format(attempt)
        _prepare_directory(path, self.artifact_root)
        return path

    def path_for(self, relative_path: str) -> Path:
        relative = _relative(relative_path)
        destination = self.artifact_root.joinpath(*relative.parts)
        _within(destination.resolve(strict=False), self.artifact_root)
        return destination

    def write_bytes(
        self,
        relative_path: str,
        data: bytes,
        *,
        kind: str,
        content_type: Optional[str] = None,
    ) -> Any:
        destination = self.path_for(relative_path)
        _prepare_directory(destination.parent, self.artifact_root)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(str(destination), flags, 0o600)
        except OSError as error:
            raise ExecutionTestArtifactError("artifact path is immutable once written") from error
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return self.reference(destination, kind=kind, content_type=content_type)

    def write_text(
        self,
        relative_path: str,
        text: str,
        *,
        kind: str,
        content_type: str = "text/plain; charset=utf-8",
    ) -> Any:
        return self.write_bytes(
            relative_path,
            text.encode("utf-8"),
            kind=kind,
            content_type=content_type,
        )

    def write_json(self, relative_path: str, value: Any, *, kind: str) -> Any:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=lambda item: item.__dict__,
        )
        return self.write_text(
            relative_path,
            payload + "\n",
            kind=kind,
            content_type="application/json",
        )

    def reference(
        self,
        path: Path,
        *,
        kind: str,
        content_type: Optional[str] = None,
    ) -> Any:
        candidate = Path(path)
        _reject_symlink_components(candidate)
        resolved = candidate.resolve(strict=True)
        _within(resolved, self.artifact_root)
        if not resolved.is_file():
            raise ExecutionTestArtifactError("artifact is not a regular file")
        digest = sha256_file(resolved)
        return self.model_factory(
            "ArtifactRef",
            artifact_id="sha256:{0}".format(digest),
            kind=kind,
            path=resolved.relative_to(self.repository_root).as_posix(),
            sha256=digest,
            size_bytes=resolved.stat().st_size,
            content_type=content_type,
        )

    def verify(self, artifact_ref: Any) -> Path:
        relative = _relative(str(getattr(artifact_ref, "path")))
        candidate = self.repository_root.joinpath(*relative.parts)
        _reject_symlink_components(candidate)
        resolved = candidate.resolve(strict=True)
        _within(resolved, self.artifact_root)
        if resolved.stat().st_size != int(getattr(artifact_ref, "size_bytes")):
            raise ExecutionTestArtifactError("artifact size mismatch")
        if sha256_file(resolved) != str(getattr(artifact_ref, "sha256")):
            raise ExecutionTestArtifactError("artifact sha256 mismatch")
        return resolved


def _identifier(value: str) -> str:
    candidate = str(value)
    if (
        not candidate
        or len(candidate) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in candidate)
    ):
        raise ExecutionTestArtifactError("invalid artifact identifier")
    return candidate


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ExecutionTestArtifactError("artifact path must be normalized")
    return path


def _within(candidate: Path, root: Path) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ExecutionTestArtifactError("artifact path escapes root") from error


def _prepare_directory(path: Path, root: Path) -> None:
    _within(path.resolve(strict=False), root)
    _reject_symlink_components(path)
    path.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path)


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ExecutionTestArtifactError("artifact path contains a symbolic link")
        if current == current.parent:
            return
        current = current.parent
