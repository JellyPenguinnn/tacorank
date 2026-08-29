from __future__ import annotations

from pathlib import Path

import pytest

from tacorank.safety import (
    ChangedPath,
    PathPolicy,
    ProtectedManifest,
    ProtectedManifestError,
    ViolationCode,
    normalize_policy_path,
    parse_protected_paths_markdown,
)

from .helpers import DATA_SHA, git, init_repository, make_manifest, write


@pytest.mark.parametrize(
    "path",
    ("../contract/rules.md", "/tmp/file", "solution//model.py", "C:\\file"),
)
def test_noncanonical_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        normalize_policy_path(path)


def test_path_policy_rejects_protected_outside_symlink_and_submodule(
    tmp_path: Path,
) -> None:
    write(tmp_path / "contract" / "rules.md", "sealed")
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "escape").symlink_to(tmp_path / "contract")
    policy = PathPolicy(tmp_path, ("solution",), ("contract",))

    findings = policy.inspect(
        (
            ChangedPath("contract/rules.md", "contract/rules.md"),
            ChangedPath("README.md", "README.md"),
            ChangedPath("solution/escape/x.py", "solution/escape/x.py"),
            ChangedPath("solution/vendor", "solution/vendor", new_mode="160000"),
        )
    )

    codes = {finding.code for finding in findings}
    assert ViolationCode.PROTECTED_PATH_MODIFIED in codes
    assert ViolationCode.PATH_TRAVERSAL in codes
    assert ViolationCode.SYMLINK_ESCAPE in codes
    assert ViolationCode.SUBMODULE_ESCAPE in codes


def test_manifest_detects_protected_content_change(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    assert manifest.verify().valid

    write(tmp_path / "contract" / "COMPETITION.md", "tampered\n")

    verification = manifest.verify()
    assert not verification.valid
    assert verification.changed_paths == ("contract",)
    assert verification.current_contract_sha256 != manifest.contract_sha256


def test_git_manifest_ignores_controller_runtime_files_but_binds_tracked_tree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    init_repository(repository)
    write(repository / ".gitignore", "contract/runtime/\n")
    git(repository, "add", ".gitignore")
    git(
        repository,
        "-c",
        "user.name=TacoRank Test",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-m",
        "ignore controller runtime",
    )
    write(repository / "contract/runtime/ledger.jsonl", "controller-only\n")
    manifest = ProtectedManifest.capture(
        repository,
        ("contract",),
        data_manifest_sha256=DATA_SHA,
        require_minimum=False,
    )

    write(repository / "contract/runtime/ledger.jsonl", "appended-controller-only\n")
    assert manifest.verify().valid
    write(repository / "contract/COMPETITION.md", "tampered\n")
    assert not manifest.verify().valid


def test_markdown_manifest_is_hash_bound_and_requires_paths(tmp_path: Path) -> None:
    write(tmp_path / "contract" / "COMPETITION.md", "sealed\n")
    manifest_path = tmp_path / "PROTECTED_PATHS.md"
    write(manifest_path, "# Protected\n\n- `contract/`\n")
    entries = parse_protected_paths_markdown(manifest_path.read_text(encoding="utf-8"))
    assert entries == ("contract",)

    manifest = ProtectedManifest.from_markdown(
        manifest_path,
        tmp_path,
        data_manifest_sha256=DATA_SHA,
        require_minimum=False,
    )
    assert manifest.verify().valid

    with pytest.raises(ProtectedManifestError):
        parse_protected_paths_markdown("# No entries\n")


def test_manifest_binds_initialized_submodule_and_rejects_uninitialized_capture(
    tmp_path: Path,
) -> None:
    submodule = tmp_path / "official-source"
    init_repository(submodule)
    write(submodule / "evaluate.py", "OFFICIAL = True\n")
    git(submodule, "add", "--all")
    git(
        submodule,
        "-c",
        "user.name=TacoRank Test",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-m",
        "official evaluator",
    )

    repository = tmp_path / "superproject"
    init_repository(repository)
    git(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule),
        "official",
    )
    git(
        repository,
        "-c",
        "user.name=TacoRank Test",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-m",
        "add official starter kit",
    )
    manifest = ProtectedManifest.capture(
        repository,
        ("official/evaluate.py",),
        data_manifest_sha256=DATA_SHA,
        require_minimum=False,
    )
    assert manifest.verify(repository).valid

    candidate = tmp_path / "candidate-worktree"
    git(repository, "worktree", "add", "-b", "candidate", str(candidate), "HEAD")
    verification = manifest.verify(candidate)
    assert not verification.valid
    assert verification.changed_paths == ("official/evaluate.py",)
    with pytest.raises(ProtectedManifestError, match="uninitialized"):
        ProtectedManifest.capture(
            candidate,
            ("official/evaluate.py",),
            data_manifest_sha256=DATA_SHA,
            require_minimum=False,
        )
