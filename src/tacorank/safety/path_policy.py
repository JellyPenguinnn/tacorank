"""Repository path policy used by the deterministic patch gate.

The policy deliberately operates on repository-relative POSIX paths.  It never
normalizes an unsafe path into a safe-looking one: non-canonical paths are
rejected before any filesystem access occurs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Optional, Sequence, Tuple


class ViolationCode(str, Enum):
    """Stable machine-readable safety violation codes."""

    PROTECTED_PATH_MODIFIED = "PROTECTED_PATH_MODIFIED"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    SYMLINK_ESCAPE = "SYMLINK_ESCAPE"
    SUBMODULE_ESCAPE = "SUBMODULE_ESCAPE"
    DIFF_PARSE_FAILURE = "DIFF_PARSE_FAILURE"
    DIFF_MISMATCH = "DIFF_MISMATCH"
    CONTRACT_HASH_MISMATCH = "CONTRACT_HASH_MISMATCH"
    INTERFACE_MISMATCH = "INTERFACE_MISMATCH"
    HIDDEN_LABEL_ACCESS = "HIDDEN_LABEL_ACCESS"
    FUTURE_INFORMATION_LEAKAGE = "FUTURE_INFORMATION_LEAKAGE"
    UNAPPROVED_COMMAND = "UNAPPROVED_COMMAND"
    UNAPPROVED_NETWORK = "UNAPPROVED_NETWORK"
    SECRET_DETECTED = "SECRET_DETECTED"
    DEPENDENCY_CHANGE = "DEPENDENCY_CHANGE"
    SYNTAX_IMPORT_FAILURE = "SYNTAX_IMPORT_FAILURE"
    SMOKE_FAILURE = "SMOKE_FAILURE"
    OUTPUT_HEADER_MISMATCH = "OUTPUT_HEADER_MISMATCH"
    OUTPUT_TYPE_MISMATCH = "OUTPUT_TYPE_MISMATCH"
    OUTPUT_ROW_COUNT_MISMATCH = "OUTPUT_ROW_COUNT_MISMATCH"
    OUTPUT_ROW_ID_MISMATCH = "OUTPUT_ROW_ID_MISMATCH"
    OUTPUT_IDENTITY_MISMATCH = "OUTPUT_IDENTITY_MISMATCH"
    OUTPUT_NONFINITE_SCORE = "OUTPUT_NONFINITE_SCORE"
    OUTPUT_DEGENERATE_SCORES = "OUTPUT_DEGENERATE_SCORES"
    OUTPUT_ARTIFACT_MISMATCH = "OUTPUT_ARTIFACT_MISMATCH"
    OUTPUT_PRODUCER_MISMATCH = "OUTPUT_PRODUCER_MISMATCH"
    OUTPUT_PROTECTED_DATA = "OUTPUT_PROTECTED_DATA"


@dataclass(frozen=True)
class PolicyViolation:
    """Internal finding converted to Person 2's shared ``Violation`` model."""

    code: ViolationCode
    check: str
    message: str
    path: Optional[str] = None

    def as_payload(self) -> dict:
        payload = {
            "code": self.code.value,
            "check": self.check,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class ChangedPath:
    """One logical path transition parsed from a Git diff."""

    old_path: Optional[str]
    new_path: Optional[str]
    status: str = "modified"
    old_mode: Optional[str] = None
    new_mode: Optional[str] = None

    @property
    def reported_path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def paths(self) -> Tuple[str, ...]:
        values = []
        for value in (self.old_path, self.new_path):
            if value is not None and value not in values:
                values.append(value)
        return tuple(values)

    @property
    def is_submodule(self) -> bool:
        return self.old_mode == "160000" or self.new_mode == "160000"

    @property
    def is_symlink(self) -> bool:
        return self.old_mode == "120000" or self.new_mode == "120000"


_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def normalize_policy_path(value: str) -> str:
    """Return a canonical repository-relative path or raise ``ValueError``.

    Backslashes and Windows drive paths are rejected even on POSIX so that a
    receipt means the same thing on every executor.
    """

    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("path must be a non-empty string without NUL bytes")
    if "\\" in value or _DRIVE_PREFIX.match(value):
        raise ValueError("path must use repository-relative POSIX syntax")
    pure = PurePosixPath(value)
    if pure.is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError("absolute paths are forbidden")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("path must be canonical and cannot contain '.' or '..'")
    normalized = pure.as_posix()
    if normalized != value or normalized == ".":
        raise ValueError("path is not canonical")
    return normalized.rstrip("/")


def path_is_within(path: str, root: str) -> bool:
    """Return whether *path* equals or descends from normalized *root*."""

    normalized_path = normalize_policy_path(path)
    normalized_root = normalize_policy_path(root.rstrip("/"))
    return normalized_path == normalized_root or normalized_path.startswith(
        normalized_root + "/"
    )


class PathPolicy:
    """Validate changed paths against editable and protected boundaries."""

    def __init__(
        self,
        repository_root: Path,
        editable_roots: Sequence[str],
        protected_paths: Sequence[str],
    ) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.editable_roots = self._normalize_roots(editable_roots, "editable")
        self.protected_paths = self._normalize_roots(protected_paths, "protected")
        if not self.editable_roots:
            raise ValueError("at least one editable root is required")

    @staticmethod
    def _normalize_roots(values: Sequence[str], label: str) -> Tuple[str, ...]:
        roots = []
        for value in values:
            try:
                root = normalize_policy_path(value.rstrip("/"))
            except ValueError as exc:
                raise ValueError("invalid {} root {!r}: {}".format(label, value, exc))
            if root not in roots:
                roots.append(root)
        return tuple(sorted(roots))

    def inspect(self, changes: Iterable[ChangedPath]) -> Tuple[PolicyViolation, ...]:
        findings = []
        for change in changes:
            if change.is_submodule:
                findings.append(
                    PolicyViolation(
                        ViolationCode.SUBMODULE_ESCAPE,
                        "path_escape",
                        "submodule pointers cannot be changed by candidate patches",
                        change.reported_path or None,
                    )
                )
            if change.is_symlink:
                findings.append(
                    PolicyViolation(
                        ViolationCode.SYMLINK_ESCAPE,
                        "path_escape",
                        "candidate patches cannot add or modify symbolic links",
                        change.reported_path or None,
                    )
                )
            for raw_path in change.paths:
                findings.extend(self._inspect_one(raw_path))
        return _deduplicate_findings(findings)

    def _inspect_one(self, raw_path: str) -> Tuple[PolicyViolation, ...]:
        try:
            path = normalize_policy_path(raw_path)
        except ValueError:
            return (
                PolicyViolation(
                    ViolationCode.PATH_TRAVERSAL,
                    "path_escape",
                    "changed path is absolute, non-canonical, or contains traversal",
                    raw_path,
                ),
            )

        findings = []
        if any(path_is_within(path, protected) for protected in self.protected_paths):
            findings.append(
                PolicyViolation(
                    ViolationCode.PROTECTED_PATH_MODIFIED,
                    "protected_path",
                    "candidate patch touches a protected path",
                    path,
                )
            )
        if not any(path_is_within(path, editable) for editable in self.editable_roots):
            findings.append(
                PolicyViolation(
                    ViolationCode.PATH_TRAVERSAL,
                    "editable_path",
                    "candidate patch is outside every editable root",
                    path,
                )
            )
        if self._crosses_symlink(path):
            findings.append(
                PolicyViolation(
                    ViolationCode.SYMLINK_ESCAPE,
                    "path_escape",
                    "changed path traverses a symbolic link",
                    path,
                )
            )
        return tuple(findings)

    def _crosses_symlink(self, relative_path: str) -> bool:
        current = self.repository_root
        for component in PurePosixPath(relative_path).parts:
            current = current / component
            if current.is_symlink():
                return True
            if not current.exists():
                break

        resolved = current.resolve(strict=False)
        try:
            return os.path.commonpath((str(self.repository_root), str(resolved))) != str(
                self.repository_root
            )
        except ValueError:
            return True


def _deduplicate_findings(
    findings: Iterable[PolicyViolation],
) -> Tuple[PolicyViolation, ...]:
    unique = {}
    for finding in findings:
        key = (finding.code.value, finding.check, finding.path, finding.message)
        unique[key] = finding
    order = sorted(unique, key=lambda item: tuple(part or "" for part in item))
    return tuple(unique[key] for key in order)
