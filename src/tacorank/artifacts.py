"""Content-addressed artifact validation for the orchestration boundary."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Optional

from .schemas import ArtifactKind, ArtifactRef, normalize_relative_path


class ArtifactError(ValueError):
    """Raised when an artifact does not match its immutable reference."""


class ArtifactStore:
    def __init__(self, repository_root: Path, approved_roots: Iterable[str] = ("artifacts", "runs")):
        self.repository_root = repository_root.resolve()
        self.approved_roots = tuple(normalize_relative_path(root) for root in approved_roots)

    @staticmethod
    def sha256_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _resolve(self, relative_path: str, require_exists: bool = True) -> Path:
        normalized = normalize_relative_path(relative_path)
        if not any(
            normalized == root or normalized.startswith(root + "/")
            for root in self.approved_roots
        ):
            raise ArtifactError("artifact path is outside approved roots: %s" % normalized)

        candidate = self.repository_root / normalized
        cursor = candidate
        while cursor != self.repository_root:
            if cursor.is_symlink():
                raise ArtifactError("artifact paths may not contain symlinks: %s" % normalized)
            cursor = cursor.parent

        if require_exists and (not candidate.exists() or not candidate.is_file()):
            raise ArtifactError("artifact bytes are missing: %s" % normalized)
        resolved_parent = candidate.parent.resolve()
        try:
            resolved_parent.relative_to(self.repository_root)
        except ValueError as exc:
            raise ArtifactError("artifact path escapes the repository") from exc
        return candidate

    def verify(self, ref: ArtifactRef) -> Path:
        path = self._resolve(ref.path)
        data = path.read_bytes()
        if len(data) != ref.size_bytes:
            raise ArtifactError("artifact size mismatch for %s" % ref.artifact_id)
        if self.sha256_bytes(data) != ref.sha256:
            raise ArtifactError("artifact hash mismatch for %s" % ref.artifact_id)
        return path

    def write(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        relative_path: str,
        content: bytes,
        content_type: Optional[str] = None,
    ) -> ArtifactRef:
        path = self._resolve(relative_path, require_exists=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = path.read_bytes()
            if existing != content:
                raise ArtifactError("immutable artifact already exists with different bytes")
        else:
            # Exclusive creation makes accidental overwrites impossible.
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                if path.exists():
                    path.unlink()
                raise

        return ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            path=normalize_relative_path(relative_path),
            sha256=self.sha256_bytes(content),
            size_bytes=len(content),
            content_type=content_type,
        )
