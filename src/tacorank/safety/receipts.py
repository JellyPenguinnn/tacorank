"""Hash-bound Gate A receipts and lazy shared-schema construction."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .path_policy import normalize_policy_path, path_is_within
from .protected_manifest import SHA256_RE


class SharedSchemaUnavailable(RuntimeError):
    """Raised when Person 2's shared model factory is not available."""


SchemaFactory = Callable[..., Any]


@dataclass(frozen=True)
class SharedSchemaFactories:
    """Injected constructors for Person 2-owned shared models.

    Factories are called with keyword payloads.  This class contains no fallback
    result records: missing shared models fail explicitly rather than allowing a
    parallel schema to drift.
    """

    check_result: SchemaFactory
    violation: SchemaFactory
    patch_check_result: SchemaFactory
    output_check_result: SchemaFactory
    artifact_ref: SchemaFactory

    @classmethod
    def from_shared_module(
        cls, module_name: str = "tacorank.schemas"
    ) -> "SharedSchemaFactories":
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            raise SharedSchemaUnavailable(
                "cannot import Person 2 shared schemas from {}".format(module_name)
            ) from exc
        names = {
            "check_result": "CheckResult",
            "violation": "Violation",
            "patch_check_result": "PatchCheckResult",
            "output_check_result": "OutputCheckResult",
            "artifact_ref": "ArtifactRef",
        }
        missing = [class_name for class_name in names.values() if not hasattr(module, class_name)]
        if missing:
            raise SharedSchemaUnavailable(
                "Person 2 shared schema module is missing: {}".format(", ".join(missing))
            )
        return cls(**{field: getattr(module, name) for field, name in names.items()})

    @staticmethod
    def build(factory: SchemaFactory, payload: Mapping[str, Any]) -> Any:
        try:
            return factory(**dict(payload))
        except TypeError as exc:
            raise SharedSchemaUnavailable(
                "shared schema factory rejected the agreed keyword payload: {}".format(exc)
            ) from exc


@dataclass(frozen=True)
class ReceiptIdentity:
    run_id: str
    experiment_id: str
    attempt: int
    patch_commit_sha: str
    diff_sha256: str
    contract_sha256: str
    protected_manifest_sha256: str
    data_manifest_sha256: str
    experiment_root_commit_sha: Optional[str] = None
    cumulative_diff_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_component(self.run_id, "run_id")
        _validate_component(self.experiment_id, "experiment_id")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        _validate_git_sha(self.patch_commit_sha)
        for field_name in (
            "diff_sha256",
            "contract_sha256",
            "protected_manifest_sha256",
            "data_manifest_sha256",
        ):
            value = getattr(self, field_name)
            if SHA256_RE.fullmatch(value) is None:
                raise ValueError("{} must be a lowercase SHA-256".format(field_name))
        if (self.experiment_root_commit_sha is None) != (
            self.cumulative_diff_sha256 is None
        ):
            raise ValueError(
                "experiment root and cumulative diff identities must be supplied together"
            )
        if self.experiment_root_commit_sha is not None:
            _validate_git_sha(self.experiment_root_commit_sha)
            if SHA256_RE.fullmatch(self.cumulative_diff_sha256 or "") is None:
                raise ValueError("cumulative_diff_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class WrittenReceipt:
    receipt_id: str
    artifact_ref: Any
    relative_path: str


class ReceiptStore:
    """Write immutable canonical receipts below the approved artifact root."""

    def __init__(
        self,
        repository_root: Path,
        factories: Optional[SharedSchemaFactories] = None,
        *,
        artifact_root: str = "artifacts",
        max_receipt_bytes: int = 1024 * 1024,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.artifact_root = normalize_policy_path(artifact_root.rstrip("/"))
        self.factories = factories
        if (
            isinstance(max_receipt_bytes, bool)
            or not isinstance(max_receipt_bytes, int)
            or max_receipt_bytes < 1
        ):
            raise ValueError("max_receipt_bytes must be a positive integer")
        self.max_receipt_bytes = max_receipt_bytes
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def write(
        self,
        identity: ReceiptIdentity,
        checks: Sequence[Mapping[str, Any]],
    ) -> WrittenReceipt:
        factories = self.factories or SharedSchemaFactories.from_shared_module()
        timestamp = self.clock()
        if timestamp.tzinfo is None:
            raise ValueError("receipt clock must return a timezone-aware datetime")
        normalized_checks = [_json_safe_mapping(check) for check in checks]
        accepted_statuses = {"pass", "not_applicable"}
        if not normalized_checks or any(
            check.get("status") not in accepted_statuses for check in normalized_checks
        ):
            raise ValueError("an acceptance receipt cannot contain failed checks")
        payload = {
            "schema_version": "1.0",
            "run_id": identity.run_id,
            "experiment_id": identity.experiment_id,
            "attempt": identity.attempt,
            "patch_commit_sha": identity.patch_commit_sha,
            "diff_sha256": identity.diff_sha256,
            "contract_sha256": identity.contract_sha256,
            "protected_manifest_sha256": identity.protected_manifest_sha256,
            "data_manifest_sha256": identity.data_manifest_sha256,
            "checks": normalized_checks,
            "timestamp": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if identity.experiment_root_commit_sha is not None:
            payload["experiment_root_commit_sha"] = identity.experiment_root_commit_sha
            payload["cumulative_diff_sha256"] = identity.cumulative_diff_sha256
        encoded = _canonical_json(payload)
        if len(encoded) > self.max_receipt_bytes:
            raise ValueError("receipt exceeds the configured size bound")
        receipt_id = hashlib.sha256(encoded).hexdigest()
        relative_path = self._receipt_relative_path(identity, receipt_id)
        target = self._safe_artifact_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            if target.read_bytes() != encoded:
                raise RuntimeError("receipt path exists with different bytes")
        artifact_payload = {
            "artifact_id": "sha256-{}".format(receipt_id),
            "kind": "verification_receipt",
            "path": relative_path,
            "sha256": receipt_id,
            "size_bytes": len(encoded),
            "content_type": "application/json",
        }
        artifact = SharedSchemaFactories.build(factories.artifact_ref, artifact_payload)
        return WrittenReceipt(receipt_id, artifact, relative_path)

    def verify(
        self,
        artifact_ref: Any,
        identity: ReceiptIdentity,
        *,
        receipt_id: str,
    ) -> Mapping[str, Any]:
        if SHA256_RE.fullmatch(receipt_id) is None:
            raise ValueError("receipt_id must be a lowercase SHA-256")
        relative_path = _read_field(artifact_ref, "path")
        if not isinstance(relative_path, str):
            raise ValueError("receipt artifact path is missing")
        expected_path = self._receipt_relative_path(identity, receipt_id)
        if relative_path != expected_path:
            raise ValueError("receipt artifact path does not match its exact identity")
        if _read_field(artifact_ref, "kind") != "verification_receipt":
            raise ValueError("receipt artifact kind is invalid")
        if _read_field(artifact_ref, "artifact_id") != "sha256-{}".format(receipt_id):
            raise ValueError("receipt artifact id does not match receipt_id")
        target = self._safe_artifact_path(relative_path)
        if target.is_symlink() or not target.is_file():
            raise ValueError("receipt artifact is missing or is a symbolic link")
        expected_size = _read_field(artifact_ref, "size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 1
            or expected_size > self.max_receipt_bytes
        ):
            raise ValueError("receipt artifact size exceeds the configured bound")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(target), flags)
        try:
            file_status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(file_status.st_mode)
                or file_status.st_size != expected_size
                or file_status.st_size > self.max_receipt_bytes
            ):
                raise ValueError("receipt artifact stat identity differs")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                encoded = handle.read(self.max_receipt_bytes + 1)
        finally:
            os.close(descriptor)
        if len(encoded) > self.max_receipt_bytes:
            raise ValueError("receipt artifact exceeds the configured size bound")
        actual_sha = hashlib.sha256(encoded).hexdigest()
        if actual_sha != receipt_id:
            raise ValueError("receipt bytes do not match receipt_id")
        if _read_field(artifact_ref, "sha256") != actual_sha:
            raise ValueError("receipt artifact hash does not match")
        if expected_size != len(encoded):
            raise ValueError("receipt artifact size does not match")
        payload = json.loads(encoded.decode("utf-8"))
        expected = {
            "run_id": identity.run_id,
            "experiment_id": identity.experiment_id,
            "attempt": identity.attempt,
            "patch_commit_sha": identity.patch_commit_sha,
            "diff_sha256": identity.diff_sha256,
            "contract_sha256": identity.contract_sha256,
            "protected_manifest_sha256": identity.protected_manifest_sha256,
            "data_manifest_sha256": identity.data_manifest_sha256,
        }
        if identity.experiment_root_commit_sha is not None:
            expected["experiment_root_commit_sha"] = identity.experiment_root_commit_sha
            expected["cumulative_diff_sha256"] = identity.cumulative_diff_sha256
        elif "experiment_root_commit_sha" in payload or "cumulative_diff_sha256" in payload:
            root_commit = payload.get("experiment_root_commit_sha")
            cumulative_sha = payload.get("cumulative_diff_sha256")
            _validate_git_sha(root_commit)
            if not isinstance(cumulative_sha, str) or SHA256_RE.fullmatch(cumulative_sha) is None:
                raise ValueError("receipt cumulative diff identity is malformed")
        mismatched = [name for name, value in expected.items() if payload.get(name) != value]
        if mismatched:
            raise ValueError(
                "receipt identity mismatch: {}".format(", ".join(sorted(mismatched)))
            )
        if not payload.get("checks") or any(
            check.get("status") not in {"pass", "not_applicable"}
            for check in payload["checks"]
        ):
            raise ValueError("receipt does not prove Gate A accepted every check")
        return payload

    def _receipt_relative_path(
        self,
        identity: ReceiptIdentity,
        receipt_id: str,
    ) -> str:
        return "{}/{}/{}/attempt_{}/gate_a/{}.json".format(
            self.artifact_root,
            identity.run_id,
            identity.experiment_id,
            identity.attempt,
            receipt_id,
        )

    def _safe_artifact_path(self, relative_path: str) -> Path:
        normalized = normalize_policy_path(relative_path)
        if not path_is_within(normalized, self.artifact_root):
            raise ValueError("receipt path is outside the approved artifact root")
        target = self.repository_root.joinpath(*normalized.split("/"))
        current = self.repository_root
        for component in normalized.split("/"):
            current = current / component
            if current.is_symlink():
                raise ValueError("receipt path traverses a symbolic link")
            if not current.exists():
                break
        resolved = target.resolve(strict=False)
        if os.path.commonpath((str(self.repository_root), str(resolved))) != str(
            self.repository_root
        ):
            raise ValueError("receipt path escapes repository root")
        return target


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _json_safe_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("check payload must be a mapping")
    return decoded


def _read_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _validate_component(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
        or value in (".", "..")
    ):
        raise ValueError("{} is not a safe path component".format(field_name))


def _validate_git_sha(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40,64}", value) is None:
        raise ValueError("patch_commit_sha must be a lowercase Git object id")
