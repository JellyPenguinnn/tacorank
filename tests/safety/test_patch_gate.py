from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from tacorank.schemas import PatchCheckResult

from tacorank.safety import (
    InterfaceRequirement,
    ReceiptIdentity,
    ReceiptStore,
    SharedSchemaFactories,
    SharedSchemaUnavailable,
    ViolationCode,
    parse_git_diff,
)

from .helpers import (
    DATA_SHA,
    FACTORIES,
    IsolatedSmokeStub,
    artifact,
    artifact_repository_for,
    commit_candidate,
    diff_for,
    git,
    init_repository,
    make_manifest,
    make_patch_gate,
    write,
)


def run(coro):
    return asyncio.run(coro)


def layout(tmp_path: Path):
    repository = tmp_path / "candidate"
    root_commit = init_repository(repository)
    manifest = make_manifest(repository)
    artifacts = artifact_repository_for(repository)
    return repository, artifacts, root_commit, manifest


def test_receipt_store_supports_run_scoped_production_layout(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    store = ReceiptStore(
        repository,
        FACTORIES,
        artifact_root="runs/run_1/artifacts",
        include_run_id=False,
        clock=lambda: datetime(2026, 8, 29, tzinfo=timezone.utc),
    )
    identity = ReceiptIdentity(
        run_id="run_1",
        experiment_id="exp_1",
        attempt=2,
        patch_commit_sha="a" * 40,
        diff_sha256="b" * 64,
        contract_sha256="c" * 64,
        protected_manifest_sha256="d" * 64,
        data_manifest_sha256="e" * 64,
    )

    receipt = store.write(
        identity,
        [{"name": "gate_a", "status": "pass", "summary": "accepted"}],
    )

    assert receipt.relative_path.startswith(
        "runs/run_1/artifacts/exp_1/attempt_002/gate_a/"
    )
    assert store.verify(
        receipt.artifact_ref,
        identity,
        receipt_id=receipt.receipt_id,
    )["attempt"] == 2


def test_gate_a_accepts_recomputed_commit_and_writes_verifiable_receipt(
    tmp_path: Path,
) -> None:
    repository, artifacts, root_commit, manifest = layout(tmp_path)
    source = "def train(rows):\n    return list(rows)\n"
    candidate = commit_candidate(
        repository,
        {"solution/model.py": source},
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
        interface_requirements=(
            InterfaceRequirement("solution/model.py", "train", parameters=("rows",)),
        ),
        smoke_check=IsolatedSmokeStub(),
    )

    result = run(
        gate.check(
            candidate,
            contract_sha256=manifest.contract_sha256,
            protected_manifest_sha256=manifest.manifest_sha256,
            data_manifest_sha256=DATA_SHA,
        )
    )

    assert result.accepted
    assert result.receipt_id
    assert result.receipt_artifact.sha256 == result.receipt_id
    assert all(check.status == "pass" for check in result.checks)
    identity = ReceiptIdentity(
        "run_1",
        "exp_1",
        1,
        candidate.patch_commit_sha,
        candidate.diff_sha256,
        manifest.contract_sha256,
        manifest.manifest_sha256,
        DATA_SHA,
        root_commit,
        candidate.diff_sha256,
    )
    payload = gate.receipt_store.verify(
        result.receipt_artifact,
        identity,
        receipt_id=result.receipt_id,
    )
    assert payload["patch_commit_sha"] == candidate.patch_commit_sha
    assert payload["experiment_root_commit_sha"] == root_commit
    assert payload["cumulative_diff_sha256"] == candidate.diff_sha256


def test_gate_a_emits_person2_canonical_models(tmp_path: Path) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "def train(rows):\n    return list(rows)\n"},
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
        smoke_check=IsolatedSmokeStub(),
    )
    gate.factories = None
    gate.receipt_store.factories = None

    result = run(gate.check(candidate))

    assert isinstance(result, PatchCheckResult)
    assert result.accepted
    assert result.receipt_artifact.artifact_id == "sha256-" + result.receipt_id


def test_gate_a_separates_candidate_worktree_from_artifact_repository(
    tmp_path: Path,
) -> None:
    source_repository = tmp_path / "source"
    root_commit = init_repository(source_repository)
    manifest = make_manifest(source_repository)
    candidate_worktree = tmp_path / "candidate-worktree"
    git(
        source_repository,
        "worktree",
        "add",
        "-b",
        "experiment/test",
        str(candidate_worktree),
        root_commit,
    )
    candidate = commit_candidate(
        candidate_worktree,
        {"solution/model.py": "def train(rows):\n    return rows\n"},
        artifact_repository_root=source_repository,
    )
    gate = make_patch_gate(
        candidate_worktree,
        manifest,
        receipt_repository_root=source_repository,
        artifact_repository_root=source_repository,
    )

    accepted = run(gate.check(candidate))
    assert accepted.accepted
    assert (source_repository / accepted.receipt_artifact.path).is_file()
    assert not (candidate_worktree / candidate.diff_artifact.path).exists()

    write(
        candidate_worktree / "contract" / "COMPETITION.md",
        "candidate tampered with protected contract\n",
    )
    rejected = run(gate.check(candidate))
    assert not rejected.accepted
    codes = {violation.code for violation in rejected.violations}
    assert ViolationCode.DIFF_MISMATCH.value in codes
    assert ViolationCode.PROTECTED_PATH_MODIFIED.value in codes


def test_receipt_verification_rejects_cross_attempt_path_substitution(
    tmp_path: Path,
) -> None:
    repository, artifacts, root_commit, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "safe = True\n"},
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )
    result = run(gate.check(candidate))
    assert result.accepted
    identity = ReceiptIdentity(
        "run_1",
        "exp_1",
        1,
        candidate.patch_commit_sha,
        candidate.diff_sha256,
        manifest.contract_sha256,
        manifest.manifest_sha256,
        DATA_SHA,
        root_commit,
        candidate.diff_sha256,
    )
    original = artifacts / result.receipt_artifact.path
    substituted_relative = result.receipt_artifact.path.replace(
        "/attempt_1/",
        "/attempt_2/",
    )
    substituted = artifacts / substituted_relative
    substituted.parent.mkdir(parents=True, exist_ok=True)
    substituted.write_bytes(original.read_bytes())
    result.receipt_artifact.path = substituted_relative

    with pytest.raises(ValueError, match="exact identity"):
        gate.receipt_store.verify(
            result.receipt_artifact,
            identity,
            receipt_id=result.receipt_id,
        )


def test_gate_a_rejects_artifact_patch_substitution(tmp_path: Path) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "safe = True\n"},
        artifact_repository_root=artifacts,
    )
    substituted = diff_for("solution/model.py", "safe = False\n")
    (artifacts / candidate.diff_artifact.path).write_bytes(substituted)
    candidate.diff_artifact = artifact(
        candidate.diff_artifact.path,
        substituted,
        "diff",
    )
    candidate.diff_sha256 = hashlib.sha256(substituted).hexdigest()
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )

    result = run(gate.check(candidate))

    assert not result.accepted
    assert result.receipt_id is None
    assert ViolationCode.DIFF_MISMATCH.value in {
        violation.code for violation in result.violations
    }


def test_gate_a_rejects_omitted_actual_changed_path(tmp_path: Path) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "safe = True\n"},
        artifact_repository_root=artifacts,
    )
    candidate.changed_files = []
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )

    result = run(gate.check(candidate))

    assert not result.accepted
    assert ViolationCode.DIFF_MISMATCH.value in {
        violation.code for violation in result.violations
    }


def test_gate_a_rejects_cross_attempt_and_oversized_diff_artifacts(
    tmp_path: Path,
) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "safe = True\n"},
        artifact_repository_root=artifacts,
    )
    original = artifacts / candidate.diff_artifact.path
    cross_attempt_path = "artifacts/run_1/exp_1/attempt_2/patch.diff"
    cross_attempt = artifacts / cross_attempt_path
    cross_attempt.parent.mkdir(parents=True, exist_ok=True)
    cross_attempt.write_bytes(original.read_bytes())
    candidate.diff_artifact.path = cross_attempt_path
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )
    substituted = run(gate.check(candidate))
    assert not substituted.accepted
    assert ViolationCode.DIFF_MISMATCH.value in {
        violation.code for violation in substituted.violations
    }

    candidate.diff_artifact.path = original.relative_to(artifacts).as_posix()
    bounded_gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
        max_diff_bytes=8,
    )
    oversized = run(bounded_gate.check(candidate))
    assert not oversized.accepted
    assert ViolationCode.DIFF_MISMATCH.value in {
        violation.code for violation in oversized.violations
    }


@pytest.mark.parametrize("state", ("dirty", "head_mismatch"))
def test_gate_a_rejects_nonexact_candidate_git_state(
    tmp_path: Path,
    state: str,
) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "safe = True\n"},
        artifact_repository_root=artifacts,
    )
    if state == "dirty":
        write(repository / "solution" / "uncommitted.py", "unsafe = True\n")
    else:
        git(
            repository,
            "-c",
            "user.name=TacoRank Test",
            "-c",
            "user.email=tacorank@invalid",
            "commit",
            "--allow-empty",
            "-m",
            "unexpected head",
        )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )

    result = run(gate.check(candidate))

    assert not result.accepted
    assert result.receipt_id is None
    assert ViolationCode.DIFF_MISMATCH.value in {
        violation.code for violation in result.violations
    }


@pytest.mark.parametrize(
    ("source", "expected_code"),
    (
        ("def broken(:\n    pass\n", ViolationCode.SYNTAX_IMPORT_FAILURE),
        ("import requests\n", ViolationCode.UNAPPROVED_NETWORK),
        ('protected_target = "protected_target"\n', ViolationCode.HIDDEN_LABEL_ACCESS),
        ("future_watch_time = 1\n", ViolationCode.FUTURE_INFORMATION_LEAKAGE),
        (
            'password = "abcdefghijklmnopqrstuvwxyz123456"\n',
            ViolationCode.SECRET_DETECTED,
        ),
    ),
)
def test_gate_a_rejects_static_policy_violations(
    tmp_path: Path,
    source: str,
    expected_code: ViolationCode,
) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": source},
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )

    result = run(gate.check(candidate))

    assert not result.accepted
    assert expected_code.value in {violation.code for violation in result.violations}


def test_gate_a_rejects_wrong_interface_and_unisolated_smoke(tmp_path: Path) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "def other(rows):\n    return rows\n"},
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
        interface_requirements=(
            InterfaceRequirement("solution/model.py", "train", parameters=("rows",)),
        ),
    )
    result = run(gate.check(candidate))
    assert ViolationCode.INTERFACE_MISMATCH.value in {
        violation.code for violation in result.violations
    }

    with pytest.raises(ValueError, match="isolated-execution"):
        make_patch_gate(
            repository,
            manifest,
            artifact_repository_root=artifacts,
            smoke_check=lambda root, item: (True, "unsafe in-process callback"),
        )


def test_gate_a_reports_isolated_smoke_failure(tmp_path: Path) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "def train(rows):\n    return rows\n"},
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
        smoke_check=IsolatedSmokeStub(False, "tiny sample failed"),
    )

    result = run(gate.check(candidate))

    assert ViolationCode.SMOKE_FAILURE.value in {
        violation.code for violation in result.violations
    }


def test_gate_a_rechecks_inherited_rejected_code_cumulatively(tmp_path: Path) -> None:
    repository, artifacts, root_commit, manifest = layout(tmp_path)
    unsafe = commit_candidate(
        repository,
        {"solution/unsafe.py": "import requests\n"},
        attempt=1,
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )
    assert not run(gate.check(unsafe)).accepted

    inherited = commit_candidate(
        repository,
        {"solution/other.py": "safe = True\n"},
        attempt=2,
        artifact_repository_root=artifacts,
    )
    missing_root = run(gate.check(inherited))
    assert not missing_root.accepted
    inherited_result = run(
        gate.check(inherited, experiment_root_commit_sha=root_commit)
    )
    assert not inherited_result.accepted
    assert ViolationCode.UNAPPROVED_NETWORK.value in {
        violation.code for violation in inherited_result.violations
    }

    repaired = commit_candidate(
        repository,
        {"solution/unsafe.py": "safe = True\n"},
        attempt=3,
        artifact_repository_root=artifacts,
    )
    repaired_result = run(
        gate.check(repaired, experiment_root_commit_sha=root_commit)
    )
    assert repaired_result.accepted


def test_gate_a_applies_cumulative_path_and_dependency_boundaries(
    tmp_path: Path,
) -> None:
    repository, artifacts, root_commit, manifest = layout(tmp_path)
    commit_candidate(
        repository,
        {
            "README.md": "candidate changed an out-of-scope file\n",
            "solution/requirements.txt": "unreviewed-package==1.0\n",
        },
        attempt=1,
        artifact_repository_root=artifacts,
    )
    inherited = commit_candidate(
        repository,
        {"solution/model.py": "safe = True\n"},
        attempt=2,
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )

    result = run(gate.check(inherited, experiment_root_commit_sha=root_commit))

    assert not result.accepted
    codes = {violation.code for violation in result.violations}
    assert ViolationCode.PATH_TRAVERSAL.value in codes
    assert ViolationCode.DEPENDENCY_CHANGE.value in codes


def test_gate_a_bounds_cumulative_repair_diff(tmp_path: Path) -> None:
    repository, artifacts, root_commit, manifest = layout(tmp_path)
    first = commit_candidate(
        repository,
        {"solution/first.py": "FIRST = {!r}\n".format("a" * 512)},
        attempt=1,
        artifact_repository_root=artifacts,
    )
    second = commit_candidate(
        repository,
        {"solution/second.py": "SECOND = {!r}\n".format("b" * 512)},
        attempt=2,
        artifact_repository_root=artifacts,
    )
    immediate_limit = max(
        first.diff_artifact.size_bytes,
        second.diff_artifact.size_bytes,
    ) + 32
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
        max_diff_bytes=immediate_limit,
    )

    result = run(gate.check(second, experiment_root_commit_sha=root_commit))

    assert not result.accepted
    assert ViolationCode.DIFF_MISMATCH.value in {
        violation.code for violation in result.violations
    }


def test_gate_a_attempt_one_rejects_different_experiment_root(tmp_path: Path) -> None:
    repository, artifacts, _, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {"solution/model.py": "safe = True\n"},
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )

    result = run(
        gate.check(
            candidate,
            experiment_root_commit_sha=candidate.patch_commit_sha,
        )
    )

    assert not result.accepted
    assert ViolationCode.DIFF_MISMATCH.value in {
        violation.code for violation in result.violations
    }


def test_gate_a_rejects_files_outside_experiment_spec_targets(tmp_path: Path) -> None:
    repository, artifacts, root_commit, manifest = layout(tmp_path)
    candidate = commit_candidate(
        repository,
        {
            "solution/candidate.py": "def run(invocation):\n    return None\n",
            "solution/smoke_check.py": "UNPLANNED = True\n",
        },
        artifact_repository_root=artifacts,
    )
    gate = make_patch_gate(
        repository,
        manifest,
        artifact_repository_root=artifacts,
    )

    result = run(
        gate.check(
            candidate,
            experiment_root_commit_sha=root_commit,
            authorized_changed_files=("solution/candidate.py",),
        )
    )

    assert not result.accepted
    assert ViolationCode.UNAPPROVED_TARGET_FILE.value in {
        violation.code for violation in result.violations
    }
    assert any(
        violation.path == "solution/smoke_check.py"
        for violation in result.violations
    )


def test_diff_parser_marks_traversal_without_resolving_it() -> None:
    diff_bytes = diff_for("../contract/rules.md", "changed = True\n")
    parsed = parse_git_diff(diff_bytes)
    assert parsed.changes[0].new_path == "../contract/rules.md"


def test_diff_parser_rejects_substituted_file_markers() -> None:
    diff_bytes = diff_for("solution/model.py", "safe = True\n").replace(
        b"+++ b/solution/model.py",
        b"+++ b/contract/COMPETITION.md",
    )
    with pytest.raises(ValueError):
        parse_git_diff(diff_bytes)


def test_missing_shared_schema_fails_explicitly() -> None:
    with pytest.raises(SharedSchemaUnavailable):
        SharedSchemaFactories.from_shared_module("does_not_exist.tacorank_schemas")
