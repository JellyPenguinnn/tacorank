"""Gate A: deterministic patch, boundary, and safety verification."""

from __future__ import annotations

import asyncio
import ast
import hashlib
import importlib.util
import os
import re
import shlex
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from tacorank.git import (
    GitOperationError,
    require_ancestor,
    resolve_commit,
    validated_repository,
)

from .command_policy import (
    inspect_dependency_changes,
    inspect_secrets,
    inspect_source_capabilities,
)
from .data_access_policy import DataAccessPolicy
from .path_policy import (
    ChangedPath,
    PathPolicy,
    PolicyViolation,
    ViolationCode,
    normalize_policy_path,
)
from .protected_manifest import ProtectedManifest, SHA256_RE
from .receipts import (
    ReceiptIdentity,
    ReceiptStore,
    SharedSchemaFactories,
)


CHECK_ORDER = (
    "diff_parse",
    "changed_file_match",
    "editable_path",
    "protected_path",
    "path_escape",
    "contract_hash",
    "syntax_import",
    "interface_contract",
    "command_policy",
    "data_boundary",
    "network_policy",
    "secret_scan",
    "dependency_policy",
    "smoke_test",
)


class PatchCandidateLike(Protocol):
    run_id: str
    experiment_id: str
    attempt: int
    base_commit_sha: str
    patch_commit_sha: str
    diff_sha256: str
    changed_files: Sequence[str]
    diff_artifact: Any


SMOKE_ISOLATION_CAPABILITY = "tacorank.hardened-sandbox-smoke.v1"


@runtime_checkable
class IsolatedSmokeCheck(Protocol):
    """Capability-bearing adapter for smoke checks outside the controller process.

    Implementations are supplied by the hardened execution backend.  A plain
    callable is intentionally insufficient because executing unaccepted
    candidate code in the controller process would cross the isolation boundary.
    """

    isolation_capability: str

    def run(
        self, repository_root: Path, candidate: PatchCandidateLike
    ) -> Tuple[bool, str]: ...


@dataclass(frozen=True)
class InterfaceRequirement:
    path: str
    symbol: str
    kind: str = "function"
    parameters: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        normalize_policy_path(self.path)
        if self.kind not in ("function", "class"):
            raise ValueError("interface requirement kind must be function or class")
        if not self.symbol.isidentifier():
            raise ValueError("interface symbol must be a Python identifier")


@dataclass(frozen=True)
class CheckRecord:
    name: str
    status: str
    summary: str

    def as_payload(self) -> dict:
        return {"name": self.name, "status": self.status, "summary": self.summary}


@dataclass(frozen=True)
class ParsedDiff:
    changes: Tuple[ChangedPath, ...]

    @property
    def changed_files(self) -> Tuple[str, ...]:
        return tuple(sorted({change.reported_path for change in self.changes}))


@dataclass(frozen=True)
class _VerifiedGitState:
    base_commit_sha: str
    patch_commit_sha: str
    experiment_root_commit_sha: str
    immediate_diff: bytes
    immediate_changed_files: Tuple[str, ...]
    cumulative_diff: bytes
    cumulative_changes: Tuple[ChangedPath, ...]

    @property
    def cumulative_changed_files(self) -> Tuple[str, ...]:
        return tuple(sorted({change.reported_path for change in self.cumulative_changes}))


class DiffParseError(ValueError):
    pass


class PatchGate:
    """Run the fourteen required Gate A checks and issue an exact receipt."""

    def __init__(
        self,
        *,
        repository_root: Path,
        editable_roots: Sequence[str],
        protected_manifest: ProtectedManifest,
        receipt_store: ReceiptStore,
        data_access_policy: DataAccessPolicy,
        allowed_command_ids: Sequence[str],
        factories: Optional[SharedSchemaFactories] = None,
        interface_requirements: Sequence[InterfaceRequirement] = (),
        allowed_import_roots: Optional[Sequence[str]] = None,
        allowed_capability_imports: Sequence[str] = (),
        allowed_dependency_changes: Sequence[str] = (),
        artifact_roots: Sequence[str] = ("artifacts",),
        artifact_repository_root: Optional[Path] = None,
        max_diff_bytes: int = 16 * 1024 * 1024,
        smoke_check: Optional[IsolatedSmokeCheck] = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.artifact_repository_root = Path(
            artifact_repository_root or self.repository_root
        ).resolve(strict=True)
        self.manifest = protected_manifest
        self.receipt_store = receipt_store
        self.factories = factories
        self.path_policy = PathPolicy(
            self.repository_root, editable_roots, protected_manifest.protected_paths
        )
        self.data_access_policy = data_access_policy
        self.allowed_command_ids = tuple(sorted(set(allowed_command_ids)))
        if not self.allowed_command_ids:
            raise ValueError("Gate A requires at least one allowed command id")
        self.interface_requirements = tuple(interface_requirements)
        self.allowed_import_roots = (
            None
            if allowed_import_roots is None
            else frozenset(allowed_import_roots)
        )
        self.allowed_capability_imports = tuple(allowed_capability_imports)
        self.allowed_dependency_changes = tuple(allowed_dependency_changes)
        self.artifact_roots = tuple(
            normalize_policy_path(root.rstrip("/")) for root in artifact_roots
        )
        if not self.artifact_roots:
            raise ValueError("at least one diff artifact root is required")
        if (
            isinstance(max_diff_bytes, bool)
            or not isinstance(max_diff_bytes, int)
            or max_diff_bytes < 1
        ):
            raise ValueError("max_diff_bytes must be a positive integer")
        self.max_diff_bytes = max_diff_bytes
        if smoke_check is not None and (
            not isinstance(smoke_check, IsolatedSmokeCheck)
            or smoke_check.isolation_capability != SMOKE_ISOLATION_CAPABILITY
        ):
            raise ValueError(
                "smoke_check must be a hardened isolated-execution adapter"
            )
        self.smoke_check = smoke_check

    async def check(
        self,
        candidate: PatchCandidateLike,
        *,
        contract_sha256: Optional[str] = None,
        protected_manifest_sha256: Optional[str] = None,
        data_manifest_sha256: Optional[str] = None,
        experiment_root_commit_sha: Optional[str] = None,
    ) -> Any:
        """Return Person 2's ``PatchCheckResult``; never consult an LLM."""

        return await asyncio.to_thread(
            self._check_sync,
            candidate,
            contract_sha256=contract_sha256,
            protected_manifest_sha256=protected_manifest_sha256,
            data_manifest_sha256=data_manifest_sha256,
            experiment_root_commit_sha=experiment_root_commit_sha,
        )

    def _check_sync(
        self,
        candidate: PatchCandidateLike,
        *,
        contract_sha256: Optional[str] = None,
        protected_manifest_sha256: Optional[str] = None,
        data_manifest_sha256: Optional[str] = None,
        experiment_root_commit_sha: Optional[str] = None,
    ) -> Any:

        factories = self.factories or SharedSchemaFactories.from_shared_module()
        if contract_sha256 is None:
            contract_sha256 = self.manifest.contract_sha256
        if protected_manifest_sha256 is None:
            protected_manifest_sha256 = self.manifest.manifest_sha256
        if data_manifest_sha256 is None:
            data_manifest_sha256 = self.manifest.data_manifest_sha256
        findings = []
        check_overrides: Dict[str, str] = {}

        git_state: Optional[_VerifiedGitState]
        try:
            git_state = self._verify_git_state(
                candidate,
                experiment_root_commit_sha=experiment_root_commit_sha,
            )
        except (GitOperationError, OSError, subprocess.SubprocessError, ValueError) as exc:
            git_state = None
            findings.append(
                PolicyViolation(
                    ViolationCode.DIFF_MISMATCH,
                    "changed_file_match",
                    "candidate Git state is not the exact clean declared commit: {}".format(
                        getattr(exc, "code", type(exc).__name__)
                    ),
                )
            )

        artifact_diff: Optional[bytes]
        try:
            artifact_diff = self._load_diff_artifact(candidate)
        except (OSError, TypeError, ValueError) as exc:
            artifact_diff = None
            findings.append(
                PolicyViolation(
                    ViolationCode.DIFF_MISMATCH,
                    "changed_file_match",
                    "diff artifact is missing, unsafe, or does not match its reference: {}".format(
                        type(exc).__name__
                    ),
                )
            )

        diff_bytes = artifact_diff or b""

        parsed: Optional[ParsedDiff]
        try:
            parsed = parse_git_diff(diff_bytes)
            check_overrides["diff_parse"] = "pass"
        except (DiffParseError, UnicodeDecodeError) as exc:
            parsed = None
            findings.append(
                PolicyViolation(
                    ViolationCode.DIFF_PARSE_FAILURE,
                    "diff_parse",
                    "patch is not a well-formed UTF-8 Git diff: {}".format(exc),
                )
            )

        actual_diff_sha = (
            hashlib.sha256(git_state.immediate_diff).hexdigest()
            if git_state is not None
            else hashlib.sha256(diff_bytes).hexdigest()
        )
        candidate_diff_sha = _field(candidate, "diff_sha256")
        reported_paths = _field(candidate, "changed_files")
        if isinstance(reported_paths, (str, bytes)) or not isinstance(reported_paths, Sequence):
            reported_paths = ()
        parsed_paths = parsed.changed_files if parsed is not None else ()
        normalized_reported, reported_paths_valid = _normalize_reported_paths(reported_paths)
        git_paths = git_state.immediate_changed_files if git_state is not None else ()
        mismatch = (
            git_state is None
            or artifact_diff is None
            or artifact_diff != git_state.immediate_diff
            or candidate_diff_sha != actual_diff_sha
            or not reported_paths_valid
            or normalized_reported != git_paths
            or (parsed is not None and parsed_paths != git_paths)
        )
        if mismatch:
            findings.append(
                PolicyViolation(
                    ViolationCode.DIFF_MISMATCH,
                    "changed_file_match",
                    "reported diff identity or changed-file list does not match exact diff bytes",
                )
            )

        cumulative_changes = git_state.cumulative_changes if git_state is not None else ()
        if git_state is not None:
            findings.extend(self.path_policy.inspect(cumulative_changes))

        identity_hashes_valid = all(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
            for value in (
                contract_sha256,
                protected_manifest_sha256,
                data_manifest_sha256,
            )
        )
        try:
            verification = self.manifest.verify(self.repository_root)
        except (OSError, ValueError) as exc:
            verification = None
            findings.append(
                PolicyViolation(
                    ViolationCode.CONTRACT_HASH_MISMATCH,
                    "contract_hash",
                    "protected state could not be verified: {}".format(type(exc).__name__),
                )
            )
        if (
            not identity_hashes_valid
            or contract_sha256 != self.manifest.contract_sha256
            or protected_manifest_sha256 != self.manifest.manifest_sha256
            or data_manifest_sha256 != self.manifest.data_manifest_sha256
            or verification is None
            or verification.current_contract_sha256 != self.manifest.contract_sha256
        ):
            findings.append(
                PolicyViolation(
                    ViolationCode.CONTRACT_HASH_MISMATCH,
                    "contract_hash",
                    "contract, protected-manifest, or data-manifest identity is not sealed",
                )
            )
        if verification is not None:
            for changed_path in verification.changed_paths:
                findings.append(
                    PolicyViolation(
                        ViolationCode.PROTECTED_PATH_MODIFIED,
                        "contract_hash",
                        "protected content differs from the frozen snapshot",
                        changed_path,
                    )
                )

        source_by_path, source_findings = self._load_changed_source(
            cumulative_changes,
            git_state.patch_commit_sha if git_state is not None else None,
        )
        findings.extend(source_findings)
        findings.extend(self._inspect_syntax_and_imports(source_by_path))
        if not self.interface_requirements:
            check_overrides["interface_contract"] = "not_applicable"
        findings.extend(
            self._inspect_interfaces(
                source_by_path,
                git_state.patch_commit_sha if git_state is not None else None,
            )
        )
        for path, source in source_by_path.items():
            capability_findings = inspect_source_capabilities(
                source,
                path,
                allowed_command_ids=self.allowed_command_ids,
                allowed_imports=self.allowed_capability_imports,
            )
            findings.extend(capability_findings)
            findings.extend(self.data_access_policy.inspect_source(source, path))
            findings.extend(inspect_secrets(source, path))
        try:
            diff_text = diff_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            diff_text = ""
        if diff_text:
            findings.extend(inspect_secrets(diff_text, "<exact-diff>"))
        findings.extend(
            inspect_dependency_changes(
                (
                    git_state.cumulative_changed_files
                    if git_state is not None
                    else ()
                ),
                allowed_paths=self.allowed_dependency_changes,
            )
        )

        if self.smoke_check is None:
            check_overrides["smoke_test"] = "not_applicable"
        elif not findings:
            try:
                passed, summary = self.smoke_check.run(self.repository_root, candidate)
            except Exception as exc:  # reviewed callback failure is a failed smoke check
                passed, summary = False, "reviewed smoke callback raised {}".format(
                    type(exc).__name__
                )
            if not passed:
                findings.append(
                    PolicyViolation(
                        ViolationCode.SMOKE_FAILURE,
                        "smoke_test",
                        summary or "tiny approved smoke test failed",
                    )
                )
        else:
            check_overrides["smoke_test"] = "not_applicable"

        if not findings and git_state is not None:
            try:
                final_git_state = self._verify_git_state(
                    candidate,
                    experiment_root_commit_sha=git_state.experiment_root_commit_sha,
                )
                if final_git_state != git_state:
                    raise ValueError("candidate Git identity changed during Gate A")
                final_manifest = self.manifest.verify(self.repository_root)
                if not final_manifest.valid:
                    raise ValueError("protected state changed during Gate A")
            except (GitOperationError, OSError, subprocess.SubprocessError, ValueError) as exc:
                findings.append(
                    PolicyViolation(
                        ViolationCode.DIFF_MISMATCH,
                        "changed_file_match",
                        "candidate or protected state changed during Gate A: {}".format(
                            getattr(exc, "code", type(exc).__name__)
                        ),
                    )
                )

        checks = _build_check_records(findings, check_overrides)
        accepted = not findings and all(check.status != "fail" for check in checks)
        receipt_id = None
        receipt_artifact = None
        if accepted:
            receipt = self.receipt_store.write(
                ReceiptIdentity(
                    run_id=_field(candidate, "run_id"),
                    experiment_id=_field(candidate, "experiment_id"),
                    attempt=_field(candidate, "attempt"),
                    patch_commit_sha=_field(candidate, "patch_commit_sha"),
                    diff_sha256=actual_diff_sha,
                    contract_sha256=contract_sha256,
                    protected_manifest_sha256=protected_manifest_sha256,
                    data_manifest_sha256=data_manifest_sha256,
                    experiment_root_commit_sha=(
                        git_state.experiment_root_commit_sha
                        if git_state is not None
                        else None
                    ),
                    cumulative_diff_sha256=(
                        hashlib.sha256(git_state.cumulative_diff).hexdigest()
                        if git_state is not None
                        else None
                    ),
                ),
                [check.as_payload() for check in checks],
            )
            receipt_id = receipt.receipt_id
            receipt_artifact = receipt.artifact_ref

        shared_checks = [
            SharedSchemaFactories.build(factories.check_result, check.as_payload())
            for check in checks
        ]
        shared_violations = [
            SharedSchemaFactories.build(factories.violation, finding.as_payload())
            for finding in _deduplicate_findings(findings)
        ]
        result_payload = {
            "run_id": _field(candidate, "run_id"),
            "experiment_id": _field(candidate, "experiment_id"),
            "attempt": _field(candidate, "attempt"),
            "patch_commit_sha": _field(candidate, "patch_commit_sha"),
            "diff_sha256": actual_diff_sha,
            "accepted": accepted,
            "receipt_id": receipt_id,
            "receipt_artifact": receipt_artifact,
            "checks": shared_checks,
            "violations": shared_violations,
        }
        return SharedSchemaFactories.build(factories.patch_check_result, result_payload)

    def _verify_git_state(
        self,
        candidate: PatchCandidateLike,
        *,
        experiment_root_commit_sha: Optional[str],
    ) -> _VerifiedGitState:
        root = validated_repository(self.repository_root)
        status = _git_output(
            root,
            ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        )
        if status:
            raise GitOperationError(
                "WORKTREE_DIRTY",
                "candidate worktree has tracked or untracked modifications",
            )

        base = resolve_commit(root, _required_field(candidate, "base_commit_sha"))
        patch = resolve_commit(root, _required_field(candidate, "patch_commit_sha"))
        head = _git_line(
            root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            "candidate HEAD",
        )
        if head != patch:
            raise GitOperationError(
                "WORKTREE_COMMIT_MISMATCH",
                "candidate worktree HEAD differs from patch_commit_sha",
            )

        parent_line = _git_line(
            root,
            ("rev-list", "--parents", "-n", "1", patch),
            "candidate commit parents",
        )
        if parent_line.split(" ") != [patch, base]:
            raise GitOperationError(
                "PATCH_PARENT_MISMATCH",
                "patch commit must have exactly base_commit_sha as its parent",
            )
        immediate_diff = _git_output(
            root,
            _committed_diff_arguments(base, patch),
            max_bytes=self.max_diff_bytes,
        )
        immediate_changed_files = _git_changed_files(
            root,
            base,
            patch,
            max_bytes=self.max_diff_bytes,
        )
        attempt = _field(candidate, "attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("candidate attempt must be a positive integer")

        declared_root = experiment_root_commit_sha
        if declared_root is None:
            candidate_root = _field(candidate, "experiment_root_commit_sha")
            if isinstance(candidate_root, str) and candidate_root:
                declared_root = candidate_root
        if attempt == 1:
            experiment_root = (
                base if declared_root is None else resolve_commit(root, declared_root)
            )
            if experiment_root != base:
                raise GitOperationError(
                    "EXPERIMENT_ROOT_MISMATCH",
                    "attempt 1 experiment root must equal base_commit_sha",
                )
        else:
            if declared_root is None:
                raise GitOperationError(
                    "EXPERIMENT_ROOT_REQUIRED",
                    "repair attempts require experiment_root_commit_sha",
                )
            experiment_root = resolve_commit(root, declared_root)
            require_ancestor(root, experiment_root, base)
        require_ancestor(root, experiment_root, patch)

        cumulative_diff = _git_output(
            root,
            _committed_diff_arguments(experiment_root, patch),
            max_bytes=self.max_diff_bytes,
        )
        if cumulative_diff:
            try:
                cumulative_changes = parse_git_diff(cumulative_diff).changes
            except (DiffParseError, UnicodeDecodeError) as exc:
                raise GitOperationError(
                    "CUMULATIVE_DIFF_PARSE_FAILURE",
                    "Git-produced cumulative diff could not be parsed",
                ) from exc
        else:
            cumulative_changes = ()
        return _VerifiedGitState(
            base_commit_sha=base,
            patch_commit_sha=patch,
            experiment_root_commit_sha=experiment_root,
            immediate_diff=immediate_diff,
            immediate_changed_files=immediate_changed_files,
            cumulative_diff=cumulative_diff,
            cumulative_changes=cumulative_changes,
        )


    def _load_diff_artifact(self, candidate: PatchCandidateLike) -> bytes:
        artifact_ref = _field(candidate, "diff_artifact")
        relative = _field(artifact_ref, "path")
        if not isinstance(relative, str):
            raise ValueError("diff artifact reference has no relative path")
        normalized = normalize_policy_path(relative)
        run_id = _safe_identity_component(_field(candidate, "run_id"), "run_id")
        experiment_id = _safe_identity_component(
            _field(candidate, "experiment_id"), "experiment_id"
        )
        attempt = _field(candidate, "attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("candidate attempt must be a positive integer")
        expected_paths = {
            "{}/{}/{}/attempt_{}/patch.diff".format(
                root,
                run_id,
                experiment_id,
                attempt,
            )
            for root in self.artifact_roots
        }
        if normalized not in expected_paths:
            raise ValueError("diff artifact path does not match candidate identity")
        if _field(artifact_ref, "kind") != "diff":
            raise ValueError("diff artifact kind is invalid")
        target = self.artifact_repository_root.joinpath(*normalized.split("/"))
        current = self.artifact_repository_root
        for component in normalized.split("/"):
            current = current / component
            if current.is_symlink():
                raise ValueError("diff artifact path traverses a symbolic link")
            if not current.exists():
                break
        if not target.is_file() or target.is_symlink():
            raise ValueError("diff artifact is missing or is a symbolic link")
        if os.path.commonpath(
            (
                str(self.artifact_repository_root),
                str(target.resolve(strict=True)),
            )
        ) != str(self.artifact_repository_root):
            raise ValueError("diff artifact escapes artifact repository root")
        expected_size = _field(artifact_ref, "size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 1
            or expected_size > self.max_diff_bytes
        ):
            raise ValueError("diff artifact size exceeds the configured bound")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(target), flags)
        try:
            file_status = os.fstat(descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise ValueError("diff artifact is not a regular file")
            if file_status.st_size != expected_size:
                raise ValueError("diff artifact stat size differs from its reference")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                encoded = handle.read(self.max_diff_bytes + 1)
        finally:
            os.close(descriptor)
        if len(encoded) > self.max_diff_bytes:
            raise ValueError("diff artifact exceeds the configured bound")
        if (
            hashlib.sha256(encoded).hexdigest() != _field(artifact_ref, "sha256")
            or len(encoded) != expected_size
        ):
            raise ValueError("diff artifact bytes differ from its reference")
        return encoded

    def _load_changed_source(
        self,
        changes: Sequence[ChangedPath],
        patch_commit_sha: Optional[str],
    ) -> Tuple[Mapping[str, str], Tuple[PolicyViolation, ...]]:
        sources = {}
        findings = []
        for change in changes:
            path = change.new_path
            if path is None:
                continue
            try:
                normalized = normalize_policy_path(path)
            except ValueError:
                continue
            if patch_commit_sha is None:
                continue
            try:
                encoded = _git_output(
                    self.repository_root,
                    ("cat-file", "blob", "{}:{}".format(patch_commit_sha, normalized)),
                    max_bytes=self.max_diff_bytes,
                )
                sources[normalized] = encoded.decode("utf-8", errors="strict")
            except GitOperationError:
                findings.append(
                    PolicyViolation(
                        ViolationCode.DIFF_MISMATCH,
                        "changed_file_match",
                        "changed file is not readable from the declared candidate commit",
                        normalized,
                    )
                )
            except UnicodeDecodeError:
                findings.append(
                    PolicyViolation(
                        ViolationCode.SECRET_DETECTED,
                        "secret_scan",
                        "candidate file is not UTF-8 and cannot be secret-scanned",
                        normalized,
                    )
                )
                if normalized.endswith(".py"):
                    findings.append(
                        PolicyViolation(
                            ViolationCode.SYNTAX_IMPORT_FAILURE,
                            "syntax_import",
                            "Python source is not valid UTF-8",
                            normalized,
                        )
                    )
        return sources, _deduplicate_findings(findings)

    def _inspect_syntax_and_imports(
        self, source_by_path: Mapping[str, str]
    ) -> Tuple[PolicyViolation, ...]:
        findings = []
        for path, source in source_by_path.items():
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError as exc:
                findings.append(
                    PolicyViolation(
                        ViolationCode.SYNTAX_IMPORT_FAILURE,
                        "syntax_import",
                        "Python syntax error at line {}".format(exc.lineno or "?"),
                        path,
                    )
                )
                continue
            for root in _import_roots(tree):
                if self.allowed_import_roots is not None:
                    available = root in self.allowed_import_roots
                else:
                    available = _module_is_available(self.repository_root, root)
                if not available:
                    findings.append(
                        PolicyViolation(
                            ViolationCode.SYNTAX_IMPORT_FAILURE,
                            "syntax_import",
                            "import root {!r} is unavailable or unapproved".format(root),
                            path,
                        )
                    )
        return _deduplicate_findings(findings)

    def _inspect_interfaces(
        self,
        source_by_path: Mapping[str, str],
        patch_commit_sha: Optional[str],
    ) -> Tuple[PolicyViolation, ...]:
        findings = []
        for requirement in self.interface_requirements:
            source = source_by_path.get(requirement.path)
            if source is None and patch_commit_sha is not None:
                try:
                    source = _git_output(
                        self.repository_root,
                        (
                            "cat-file",
                            "blob",
                            "{}:{}".format(patch_commit_sha, requirement.path),
                        ),
                        max_bytes=self.max_diff_bytes,
                    ).decode("utf-8", errors="strict")
                except (GitOperationError, UnicodeDecodeError):
                    source = None
            if source is None:
                findings.append(
                    PolicyViolation(
                        ViolationCode.INTERFACE_MISMATCH,
                        "interface_contract",
                        "required interface file is missing or non-text",
                        requirement.path,
                    )
                )
                continue
            try:
                tree = ast.parse(source, filename=requirement.path)
            except SyntaxError:
                continue
            matching: Sequence[ast.AST]
            if requirement.kind == "class":
                matching = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef)
                    and node.name == requirement.symbol
                ]
            else:
                matching = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == requirement.symbol
                ]
            if not matching:
                findings.append(
                    PolicyViolation(
                        ViolationCode.INTERFACE_MISMATCH,
                        "interface_contract",
                        "required {} {!r} is missing".format(
                            requirement.kind, requirement.symbol
                        ),
                        requirement.path,
                    )
                )
                continue
            if requirement.parameters is not None and requirement.kind == "function":
                actual = _parameter_names(matching[0])
                if actual != requirement.parameters:
                    findings.append(
                        PolicyViolation(
                            ViolationCode.INTERFACE_MISMATCH,
                            "interface_contract",
                            "required function signature does not match the frozen interface",
                            requirement.path,
                        )
                    )
        return _deduplicate_findings(findings)


class FakePatchGate:
    """Return a caller-supplied shared result without defining another schema."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []

    async def check(self, candidate: Any, **kwargs: Any) -> Any:
        self.calls.append(candidate)
        del kwargs
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def parse_git_diff(diff_bytes: bytes) -> ParsedDiff:
    """Parse exact ``git diff --binary --no-ext-diff`` bytes conservatively."""

    text = diff_bytes.decode("utf-8", errors="strict")
    if not text.strip():
        raise DiffParseError("diff is empty")
    changes = []
    current: Optional[Dict[str, Any]] = None
    has_change_marker = False
    in_hunk = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                if not has_change_marker:
                    raise DiffParseError("diff block contains no change marker")
                changes.append(_changed_path_from_block(current))
            try:
                tokens = shlex.split(line, posix=True)
            except ValueError as exc:
                raise DiffParseError("malformed diff header") from exc
            if len(tokens) != 4 or tokens[:2] != ["diff", "--git"]:
                raise DiffParseError("malformed diff header")
            current = {
                "old_path": _diff_header_path(tokens[2], "a/"),
                "new_path": _diff_header_path(tokens[3], "b/"),
                "status": "modified",
                "old_mode": None,
                "new_mode": None,
                "old_marker_seen": False,
                "new_marker_seen": False,
                "old_marker": None,
                "new_marker": None,
            }
            has_change_marker = False
            in_hunk = False
            continue
        if current is None:
            if line.strip():
                raise DiffParseError("content occurs before first diff header")
            continue
        if not in_hunk and line.startswith("new file mode "):
            current["status"] = "added"
            current["new_mode"] = line.rsplit(" ", 1)[-1]
            has_change_marker = True
        elif not in_hunk and line.startswith("deleted file mode "):
            current["status"] = "deleted"
            current["old_mode"] = line.rsplit(" ", 1)[-1]
            current["new_path"] = None
            has_change_marker = True
        elif not in_hunk and line.startswith("old mode "):
            current["old_mode"] = line.rsplit(" ", 1)[-1]
            has_change_marker = True
        elif not in_hunk and line.startswith("new mode "):
            current["new_mode"] = line.rsplit(" ", 1)[-1]
            has_change_marker = True
        elif not in_hunk and line.startswith("rename from "):
            current["old_path"] = _metadata_path(line[len("rename from ") :])
            current["status"] = "renamed"
            has_change_marker = True
        elif not in_hunk and line.startswith("rename to "):
            current["new_path"] = _metadata_path(line[len("rename to ") :])
            current["status"] = "renamed"
            has_change_marker = True
        elif not in_hunk and line.startswith("--- "):
            current["old_marker_seen"] = True
            current["old_marker"] = _file_marker_path(line[4:], "a/")
        elif not in_hunk and line.startswith("+++ "):
            current["new_marker_seen"] = True
            current["new_marker"] = _file_marker_path(line[4:], "b/")
        elif line.startswith("@@ ") or line.startswith("@@@ "):
            has_change_marker = True
            in_hunk = True
        elif line.startswith("GIT binary patch") or line.startswith("Binary files "):
            has_change_marker = True
        elif line.startswith("index "):
            parts = line.split()
            if len(parts) >= 3 and parts[-1] == "160000":
                current["old_mode"] = "160000"
                current["new_mode"] = "160000"
            # Index metadata alone does not prove changed content.
    if current is None:
        raise DiffParseError("no Git diff blocks found")
    if not has_change_marker:
        raise DiffParseError("diff block contains no change marker")
    changes.append(_changed_path_from_block(current))
    if not changes:
        raise DiffParseError("no changed paths found")
    reported = [change.reported_path for change in changes]
    if len(set(reported)) != len(reported):
        raise DiffParseError("diff contains duplicate logical path blocks")
    return ParsedDiff(tuple(changes))


def _changed_path_from_block(block: Mapping[str, Any]) -> ChangedPath:
    old_path = block.get("old_path")
    new_path = block.get("new_path")
    old_seen = bool(block.get("old_marker_seen"))
    new_seen = bool(block.get("new_marker_seen"))
    if old_seen != new_seen:
        raise DiffParseError("diff file markers are incomplete")
    if old_seen:
        expected_old = None if block.get("status") == "added" else old_path
        expected_new = None if block.get("status") == "deleted" else new_path
        if block.get("old_marker") != expected_old or block.get("new_marker") != expected_new:
            raise DiffParseError("diff file markers do not match diff header")
    for mode in (block.get("old_mode"), block.get("new_mode")):
        if mode is not None and re.fullmatch(r"[0-7]{6}", mode) is None:
            raise DiffParseError("diff contains an invalid file mode")
    return ChangedPath(
        old_path=old_path,
        new_path=new_path,
        status=block.get("status") or "modified",
        old_mode=block.get("old_mode"),
        new_mode=block.get("new_mode"),
    )


def _diff_header_path(value: str, prefix: str) -> str:
    if not value.startswith(prefix) or len(value) == len(prefix):
        raise DiffParseError("diff header path lacks {} prefix".format(prefix))
    return value[len(prefix) :]


def _metadata_path(value: str) -> str:
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError as exc:
        raise DiffParseError("malformed rename path") from exc
    if len(tokens) != 1:
        raise DiffParseError("malformed rename path")
    return tokens[0]


def _file_marker_path(value: str, prefix: str) -> Optional[str]:
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError as exc:
        raise DiffParseError("malformed diff file marker") from exc
    if len(tokens) != 1:
        raise DiffParseError("malformed diff file marker")
    if tokens[0] == "/dev/null":
        return None
    return _diff_header_path(tokens[0], prefix)


def _build_check_records(
    findings: Sequence[PolicyViolation], overrides: Mapping[str, str]
) -> Tuple[CheckRecord, ...]:
    by_check: Dict[str, list] = {name: [] for name in CHECK_ORDER}
    for finding in _deduplicate_findings(findings):
        by_check.setdefault(finding.check, []).append(finding)
    records = []
    for name in CHECK_ORDER:
        failures = by_check[name]
        status = "fail" if failures else overrides.get(name, "pass")
        if status == "fail":
            summary = "{} violation(s)".format(len(failures))
        elif status == "not_applicable":
            summary = "check was not applicable to this patch"
        else:
            summary = "check passed"
        records.append(CheckRecord(name, status, summary))
    return tuple(records)


def _normalize_reported_paths(paths: Sequence[str]) -> Tuple[Tuple[str, ...], bool]:
    normalized = []
    try:
        for path in paths:
            normalized.append(normalize_policy_path(path))
    except (TypeError, ValueError):
        return (), False
    return tuple(sorted(set(normalized))), len(set(normalized)) == len(paths)


def _committed_diff_arguments(base: str, patch: str) -> Tuple[str, ...]:
    return (
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-relative",
        "--no-indent-heuristic",
        "--diff-algorithm=myers",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--no-renames",
        base,
        patch,
        "--",
    )


def _git_changed_files(
    repository: Path,
    base: str,
    patch: str,
    *,
    max_bytes: int,
) -> Tuple[str, ...]:
    encoded = _git_output(
        repository,
        ("diff", "--name-only", "-z", "--no-renames", base, patch, "--"),
        max_bytes=max_bytes,
    )
    records = encoded.split(b"\x00")
    if records and records[-1] == b"":
        records.pop()
    paths = []
    for record in records:
        try:
            paths.append(normalize_policy_path(record.decode("utf-8", errors="strict")))
        except (UnicodeDecodeError, ValueError) as exc:
            raise GitOperationError(
                "MALFORMED_DIFF_PATHS",
                "candidate commit contains a noncanonical or non-UTF-8 path",
            ) from exc
    if len(paths) != len(set(paths)):
        raise GitOperationError(
            "MALFORMED_DIFF_PATHS",
            "candidate commit contains duplicate changed paths",
        )
    return tuple(sorted(paths))


def _git_output(
    repository: Path,
    arguments: Sequence[str],
    *,
    max_bytes: Optional[int] = None,
) -> bytes:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "protocol.file.allow=never",
        "-C",
        str(repository),
        *arguments,
    ]
    if max_bytes is not None:
        if max_bytes < 1:
            raise ValueError("Git output bound must be positive")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            shell=False,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            raise GitOperationError(
                "GIT_STATE_CHECK_FAILED", "bounded Git output pipe is unavailable"
            )
        encoded = process.stdout.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            process.kill()
            process.wait()
            raise GitOperationError(
                "GIT_OUTPUT_LIMIT",
                "candidate Git content exceeds the configured safety bound",
            )
        return_code = process.wait()
        if return_code != 0:
            raise GitOperationError(
                "GIT_STATE_CHECK_FAILED",
                "bounded Git state verification failed",
            )
        return encoded
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        shell=False,
    )
    if completed.returncode != 0:
        raise GitOperationError(
            "GIT_STATE_CHECK_FAILED",
            "bounded Git state verification failed",
        )
    return completed.stdout


def _git_line(repository: Path, arguments: Sequence[str], label: str) -> str:
    encoded = _git_output(repository, arguments)
    try:
        value = encoded.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise GitOperationError(
            "GIT_STATE_CHECK_FAILED", "{} is not ASCII".format(label)
        ) from exc
    if not value or "\n" in value or "\r" in value:
        raise GitOperationError(
            "GIT_STATE_CHECK_FAILED", "{} is malformed".format(label)
        )
    return value


def _required_field(value: Any, name: str) -> str:
    field = _field(value, name)
    if not isinstance(field, str) or not field:
        raise ValueError("candidate {} is required".format(name))
    return field


def _safe_identity_component(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None
        or value in {".", ".."}
        or ".." in value
        or value.endswith(".lock")
    ):
        raise ValueError("{} is not a safe artifact component".format(field_name))
    return value


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _import_roots(tree: ast.AST) -> Tuple[str, ...]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return tuple(sorted(roots))


def _module_is_available(repository_root: Path, root: str) -> bool:
    if (repository_root / "src" / root).exists() or (repository_root / root).exists():
        return True
    try:
        return importlib.util.find_spec(root) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _parameter_names(node: ast.AST) -> Tuple[str, ...]:
    arguments = node.args  # type: ignore[attr-defined]
    names = [argument.arg for argument in arguments.posonlyargs]
    names.extend(argument.arg for argument in arguments.args)
    if arguments.vararg is not None:
        names.append("*" + arguments.vararg.arg)
    names.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg is not None:
        names.append("**" + arguments.kwarg.arg)
    return tuple(names)


def _deduplicate_findings(
    findings: Iterable[PolicyViolation],
) -> Tuple[PolicyViolation, ...]:
    unique = {}
    for finding in findings:
        key = (finding.code.value, finding.check, finding.path, finding.message)
        unique[key] = finding
    order = sorted(unique, key=lambda item: tuple(part or "" for part in item))
    return tuple(unique[key] for key in order)
