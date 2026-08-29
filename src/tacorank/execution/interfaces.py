"""Narrow shared-model and artifact-store ports owned by execution.

Person 2 supplies the canonical artifact store and shared schema classes.  This
module defines only the behavior execution consumes and fails explicitly while
those shared integrations are unavailable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
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
