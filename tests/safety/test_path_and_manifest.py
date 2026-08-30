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


def test_git_manifest_canonicalizes_cross_platform_line_endings(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    init_repository(repository)
    write(repository / ".gitattributes", "*.md text eol=lf\n")
    git(repository, "add", ".gitattributes")
    git(
        repository,
        "-c",
        "user.name=TacoRank Test",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-m",
        "normalize protected text",
    )
    protected = repository / "contract" / "COMPETITION.md"
    protected.write_bytes(b"sealed contract\r\n")
    assert git(repository, "status", "--porcelain").strip() == ""
    manifest = ProtectedManifest.capture(
        repository,
        ("contract",),
        data_manifest_sha256=DATA_SHA,
        require_minimum=False,
    )

    candidate = tmp_path / "candidate-worktree"
    git(repository, "worktree", "add", "-b", "candidate-eol", str(candidate), "HEAD")
    candidate_protected = candidate / "contract" / "COMPETITION.md"
    candidate_protected.unlink()
    git(
        candidate,
        "-c",
        "core.autocrlf=false",
        "checkout",
        "HEAD",
        "--",
        "contract/COMPETITION.md",
    )
    assert candidate_protected.read_bytes() == b"sealed contract\n"
    assert git(candidate, "status", "--porcelain").strip() == ""

    assert manifest.verify(candidate).valid

    candidate_protected.write_bytes(b"tampered contract\n")
    verification = manifest.verify(candidate)
    assert not verification.valid
    assert verification.changed_paths == ("contract",)


def test_git_manifest_does_not_normalize_binary_line_endings(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    init_repository(repository)
    binary = repository / "contract" / "sealed.bin"
    binary.write_bytes(b"\x00sealed\r\n")
    git(repository, "add", "contract/sealed.bin")
    git(
        repository,
        "-c",
        "user.name=TacoRank Test",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-m",
        "seal binary contract",
    )
    manifest = ProtectedManifest.capture(
        repository,
        ("contract",),
        data_manifest_sha256=DATA_SHA,
        require_minimum=False,
    )

    binary.write_bytes(b"\x00sealed\n")

    verification = manifest.verify(repository)
    assert not verification.valid
    assert verification.changed_paths == ("contract",)


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

    official_evaluator = repository / "official" / "evaluate.py"
    indexed_text = b"OFFICIAL = True\n"
    official_evaluator.write_bytes(indexed_text.replace(b"\n", b"\r\n"))
    assert manifest.verify(repository).valid
    official_evaluator.write_bytes(b"OFFICIAL = False\n")
    tampered = manifest.verify(repository)
    assert not tampered.valid
    assert tampered.changed_paths == ("official/evaluate.py",)
    official_evaluator.write_bytes(indexed_text)

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
