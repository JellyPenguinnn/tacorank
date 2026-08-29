from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from tacorank.git import capture_commit_patch
from tacorank.safety import (
    DataAccessPolicy,
    DataViewPolicy,
    SMOKE_ISOLATION_CAPABILITY,
    PatchGate,
    ProtectedManifest,
    ReceiptStore,
    SharedSchemaFactories,
)


class Record:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


def _record_factory(record_type: str):
    def create(**values: Any) -> Record:
        return Record(record_type=record_type, **values)

    return create


FACTORIES = SharedSchemaFactories(
    check_result=_record_factory("CheckResult"),
    violation=_record_factory("Violation"),
    patch_check_result=_record_factory("PatchCheckResult"),
    output_check_result=_record_factory("OutputCheckResult"),
    artifact_ref=_record_factory("ArtifactRef"),
)

DATA_SHA = "d" * 64
COMMIT_SHA = "c" * 40


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def artifact(path: str, content: bytes, kind: str) -> Record:
    return Record(
        artifact_id="sha256-{}".format(hashlib.sha256(content).hexdigest()),
        kind=kind,
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_type="text/csv" if kind == "predictions" else "text/x-diff",
    )


class IsolatedSmokeStub:
    isolation_capability = SMOKE_ISOLATION_CAPABILITY

    def __init__(self, passed: bool = True, summary: str = "tiny sample passed") -> None:
        self.passed = passed
        self.summary = summary

    def run(self, repository_root: Path, candidate: Any) -> tuple[bool, str]:
        del repository_root, candidate
        return self.passed, self.summary


def git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


def init_repository(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init")
    write(root / "contract" / "COMPETITION.md", "sealed contract\n")
    write(root / "solution" / "__init__.py", "")
    git(root, "add", "--all")
    git(
        root,
        "-c",
        "user.name=TacoRank Test",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-m",
        "baseline",
    )
    return git(root, "rev-parse", "HEAD")


def artifact_repository_for(root: Path) -> Path:
    repository = root.parent / "{}-artifact-repository".format(root.name)
    repository.mkdir(parents=True, exist_ok=True)
    return repository


def commit_candidate(
    root: Path,
    changes: Mapping[str, Optional[str]],
    *,
    attempt: int = 1,
    artifact_repository_root: Optional[Path] = None,
    changed_files: Optional[Sequence[str]] = None,
) -> Record:
    base_commit_sha = git(root, "rev-parse", "HEAD")
    for relative_path, content in changes.items():
        target = root.joinpath(*relative_path.split("/"))
        if content is None:
            target.unlink()
        else:
            write(target, content)
    git(root, "add", "--all")
    git(
        root,
        "-c",
        "user.name=TacoRank Test",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-m",
        "candidate attempt {}".format(attempt),
    )
    patch_commit_sha = git(root, "rev-parse", "HEAD")
    sealed = capture_commit_patch(root, base_commit_sha, patch_commit_sha)
    artifact_root = artifact_repository_root or artifact_repository_for(root)
    relative = "artifacts/run_1/exp_1/attempt_{}/patch.diff".format(attempt)
    destination = artifact_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(sealed.diff)
    return Record(
        run_id="run_1",
        experiment_id="exp_1",
        attempt=attempt,
        base_commit_sha=base_commit_sha,
        patch_commit_sha=patch_commit_sha,
        diff_sha256=sealed.diff_sha256,
        changed_files=list(
            sealed.changed_files if changed_files is None else changed_files
        ),
        diff_artifact=artifact(relative, sealed.diff, "diff"),
    )


def diff_for(path: str, source: str) -> bytes:
    additions = "".join("+{}\n".format(line) for line in source.splitlines())
    line_count = max(1, len(source.splitlines()))
    return (
        "diff --git a/{0} b/{0}\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/{0}\n"
        "@@ -0,0 +1,{1} @@\n"
        "{2}".format(path, line_count, additions)
    ).encode("utf-8")


def make_candidate(diff_bytes: bytes, changed_files: Sequence[str]) -> Record:
    return Record(
        run_id="run_1",
        experiment_id="exp_1",
        attempt=1,
        base_commit_sha="b" * 40,
        patch_commit_sha=COMMIT_SHA,
        diff_sha256=hashlib.sha256(diff_bytes).hexdigest(),
        changed_files=list(changed_files),
        diff_artifact=artifact("artifacts/run_1/exp_1/diff.patch", diff_bytes, "diff"),
    )


def make_manifest(root: Path) -> ProtectedManifest:
    write(root / "contract" / "COMPETITION.md", "sealed contract\n")
    return ProtectedManifest.capture(
        root,
        ("contract",),
        contract_paths=("contract",),
        data_manifest_sha256=DATA_SHA,
        require_minimum=False,
    )


def make_patch_gate(
    root: Path,
    manifest: ProtectedManifest,
    *,
    receipt_repository_root: Optional[Path] = None,
    **overrides: Any,
) -> PatchGate:
    artifact_repository_root = overrides.pop(
        "artifact_repository_root", artifact_repository_for(root)
    )
    store = ReceiptStore(
        receipt_repository_root or artifact_repository_root,
        FACTORIES,
        clock=lambda: datetime(2026, 8, 29, tzinfo=timezone.utc),
    )
    data_policy = DataAccessPolicy(
        views=(
            DataViewPolicy(
                "training",
                ("user_id", "item_id", "training_target"),
                ("data/training",),
            ),
        ),
        protected_columns=("protected_target",),
        hidden_path_tokens=("hidden_labels", "test_labels"),
        future_column_patterns=(r"(?:^|_)future(?:_|$)", r"^next_"),
    )
    values = {
        "repository_root": root,
        "editable_roots": ("solution",),
        "protected_manifest": manifest,
        "receipt_store": store,
        "data_access_policy": data_policy,
        "allowed_command_ids": ("candidate_smoke",),
        "factories": FACTORIES,
        "artifact_repository_root": artifact_repository_root,
    }
    values.update(overrides)
    return PatchGate(**values)
