"""Narrow shared-model and artifact-store ports owned by execution.

Person 2 supplies the canonical artifact store and shared schema classes.  This
module defines only the behavior execution consumes and fails explicitly while
those shared integrations are unavailable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable


class SharedSchemaUnavailable(RuntimeError):
    """The canonical Person 2 shared model has not been integrated."""


ModelFactory = Callable[..., Any]


def default_model_factory(model_name: str, **fields: Any) -> Any:
    from tacorank import schemas

    model = getattr(schemas, model_name, None)
    if model is None:
        raise SharedSchemaUnavailable(
            "tacorank.schemas.{0} is required for this integration".format(
                model_name
            )
        )
    return model(**fields)


def model_to_mapping(model: Any) -> Mapping[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if isinstance(model, Mapping):
        return model
    values = getattr(model, "__dict__", None)
    if values is None:
        raise TypeError("model cannot be serialized as a mapping")
    return values


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@runtime_checkable
class ExecutionArtifactStore(Protocol):
    """Subset of Person 2's immutable artifact service used by execution."""

    artifact_root: Path

    def attempt_directory(
        self, run_id: str, experiment_id: str, attempt: int
    ) -> Path:
        ...

    def write_text(
        self,
        relative_path: str,
        text: str,
        *,
        kind: str,
        content_type: str = "text/plain; charset=utf-8",
    ) -> Any:
        ...

    def write_json(self, relative_path: str, value: Any, *, kind: str) -> Any:
        ...

    def reference(
        self,
        path: Path,
        *,
        kind: str,
        content_type: Optional[str] = None,
    ) -> Any:
        ...

    def verify(self, artifact_ref: Any) -> Path:
        """Resolve and hash-verify an immutable ArtifactRef."""

        ...


class CanonicalArtifactStoreAdapter:
    """Adapt Person 2's canonical ``ArtifactStore`` to the execution port."""

    def __init__(self, store: Any, artifact_root: str = "artifacts") -> None:
        repository_root = Path(getattr(store, "repository_root")).resolve(strict=True)
        root_relative = _normalized_relative(artifact_root)
        root = repository_root.joinpath(*root_relative.parts)
        _reject_symlink_components(root)
        root.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(root)
        self.store = store
        self.repository_root = repository_root
        self.artifact_root = root.resolve(strict=True)
        _require_within(self.artifact_root, repository_root)

    def attempt_directory(
        self, run_id: str, experiment_id: str, attempt: int
    ) -> Path:
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        path = self.artifact_root / _identifier(run_id) / _identifier(experiment_id)
        path = path / "attempt_{0}".format(attempt)
        _require_within(path.resolve(strict=False), self.artifact_root)
        _reject_symlink_components(path)
        path.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(path)
        return path.resolve(strict=True)

    def write_text(
        self,
        relative_path: str,
        text: str,
        *,
        kind: str,
        content_type: str = "text/plain; charset=utf-8",
    ) -> Any:
        return self._write_bytes(
            relative_path,
            text.encode("utf-8"),
            kind=kind,
            content_type=content_type,
        )

    def write_json(self, relative_path: str, value: Any, *, kind: str) -> Any:
        encoded = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=lambda item: dict(model_to_mapping(item)),
            ).encode("utf-8")
            + b"\n"
        )
        return self._write_bytes(
            relative_path,
            encoded,
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
        _require_within(resolved, self.artifact_root)
        if not resolved.is_file():
            raise ValueError("artifact reference must name a regular file")
        return self._canonical_write(
            resolved.relative_to(self.repository_root).as_posix(),
            resolved.read_bytes(),
            kind=kind,
            content_type=content_type,
        )

    def verify(self, artifact_ref: Any) -> Path:
        self.store.verify(artifact_ref)
        relative = _normalized_relative(str(getattr(artifact_ref, "path")))
        candidate = self.repository_root.joinpath(*relative.parts)
        _reject_symlink_components(candidate)
        resolved = candidate.resolve(strict=True)
        _require_within(resolved, self.artifact_root)
        if not resolved.is_file():
            raise ValueError("artifact reference does not name a regular file")
        return resolved

    def _write_bytes(
        self,
        relative_path: str,
        content: bytes,
        *,
        kind: str,
        content_type: Optional[str],
    ) -> Any:
        relative = _normalized_relative(relative_path)
        destination = self.artifact_root.joinpath(*relative.parts)
        _require_within(destination.resolve(strict=False), self.artifact_root)
        _reject_symlink_components(destination)
        repository_relative = destination.relative_to(self.repository_root).as_posix()
        return self._canonical_write(
            repository_relative,
            content,
            kind=kind,
            content_type=content_type,
        )

    def _canonical_write(
        self,
        repository_relative: str,
        content: bytes,
        *,
        kind: str,
        content_type: Optional[str],
    ) -> Any:
        from tacorank.schemas import ArtifactKind

        digest = hashlib.sha256(content).hexdigest()
        return self.store.write(
            artifact_id="sha256-{0}".format(digest),
            kind=ArtifactKind(kind),
            relative_path=repository_relative,
            content=content,
            content_type=content_type,
        )


def _normalized_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be normalized and relative")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("artifact path must be normalized and relative")
    return path


def _identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in value
        )
    ):
        raise ValueError("artifact identity is invalid")
    return value


def _require_within(candidate: Path, root: Path) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("artifact path escapes its approved root") from error


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError("artifact path contains a symbolic link")
        if current == current.parent:
            return
        current = current.parent
