from __future__ import annotations

import pytest

from tacorank.artifacts import ArtifactError, ArtifactStore
from tacorank.schemas import ArtifactKind, ArtifactRef


def test_artifact_hash_and_root_are_revalidated(repository):
    store = ArtifactStore(repository)
    ref = store.write(
        artifact_id="report_1",
        kind=ArtifactKind.REPORT,
        relative_path="artifacts/report.txt",
        content=b"trusted bytes",
    )
    store.verify(ref)
    bad = ref.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ArtifactError, match="hash mismatch"):
        store.verify(bad)
    with pytest.raises(ArtifactError, match="outside approved"):
        store.verify(
            ArtifactRef(
                artifact_id="outside",
                kind=ArtifactKind.OTHER,
                path="solution/file.txt",
                sha256="0" * 64,
                size_bytes=0,
            )
        )
