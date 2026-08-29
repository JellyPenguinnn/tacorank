from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tacorank.artifacts import ArtifactStore
from tacorank.execution import CanonicalArtifactStoreAdapter
from tacorank.execution.interfaces import (
    SharedSchemaUnavailable,
    default_model_factory,
)

from .conftest import stub_model_factory
from .store import ExecutionTestArtifactError, TestArtifactStore


def test_artifact_is_hash_addressed_and_verified(execution_layout: Any) -> None:
    store = execution_layout.store
    reference = store.write_text(
        "run_001/exp_0001/attempt_1/result.txt",
        "deterministic\n",
        kind="other",
    )

    assert reference.artifact_id == "sha256-" + reference.sha256
    assert reference.path.startswith("artifacts/")
    assert store.verify(reference).read_text() == "deterministic\n"

    store.verify(reference).write_text("changed", encoding="utf-8")
    with pytest.raises(ExecutionTestArtifactError, match="size mismatch|sha256 mismatch"):
        store.verify(reference)


def test_artifact_paths_are_normalized_immutable_and_attempts_start_at_one(
    execution_layout: Any,
) -> None:
    store = execution_layout.store
    with pytest.raises(ExecutionTestArtifactError):
        store.attempt_directory("run_001", "exp_0001", 0)
    with pytest.raises(ExecutionTestArtifactError):
        store.write_text("../escape.txt", "x", kind="other")
    with pytest.raises(ExecutionTestArtifactError, match="normalized"):
        store.write_text(
            "run_001//exp_0001/attempt_1/value.txt", "x", kind="other"
        )

    path = "run_001/exp_0001/attempt_1/value.txt"
    store.write_text(path, "first", kind="other")
    with pytest.raises(ExecutionTestArtifactError, match="immutable"):
        store.write_text(path, "second", kind="other")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_artifact_root_rejects_existing_symlink_components(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    real_root = repository / "real-artifacts"
    linked_root = repository / "artifacts"
    real_root.mkdir(parents=True)
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ExecutionTestArtifactError, match="symbolic link"):
        TestArtifactStore(
            repository,
            linked_root,
            model_factory=stub_model_factory,
        )


def test_canonical_person2_artifact_store_adapter(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    adapter = CanonicalArtifactStoreAdapter(ArtifactStore(repository))

    reference = adapter.write_text(
        "run_001/exp_0001/attempt_1/result.txt",
        "canonical\n",
        kind="other",
    )

    assert reference.artifact_id == "sha256-" + reference.sha256
    assert adapter.verify(reference).read_text(encoding="utf-8") == "canonical\n"


def test_default_factory_builds_the_canonical_artifact_schema() -> None:
    reference = default_model_factory(
        "ArtifactRef",
        artifact_id="sha256-" + "0" * 64,
        kind="other",
        path="artifacts/a",
        sha256="0" * 64,
        size_bytes=0,
        content_type=None,
    )

    assert reference.artifact_id == "sha256-" + reference.sha256


def test_missing_canonical_artifact_schema_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tacorank import schemas

    monkeypatch.delattr(schemas, "ArtifactRef")
    with pytest.raises(SharedSchemaUnavailable, match="ArtifactRef"):
        default_model_factory(
            "ArtifactRef",
            artifact_id="sha256-" + "0" * 64,
            kind="other",
            path="artifacts/a",
            sha256="0" * 64,
            size_bytes=0,
            content_type=None,
        )
