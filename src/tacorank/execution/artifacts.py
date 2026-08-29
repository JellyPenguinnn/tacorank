"""Execution-specific artifact layout and capture helpers."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Tuple

from tacorank.execution.commands import ExpectedArtifact, ResolvedCommand, executable_identity
from tacorank.execution.interfaces import ExecutionArtifactStore, sha256_file


@dataclass(frozen=True)
class CapturedOutputs:
    prediction_artifact: Optional[Any]
    checkpoint_artifact: Optional[Any]
    missing_required_roles: Tuple[str, ...]


class ExecutionSealVerificationError(RuntimeError):
    """Raised when a trusted execution-seal sidecar does not bind its output."""


class ExecutionArtifactPathError(RuntimeError):
    """An execution-owned artifact path violated the injected store contract."""


EXECUTION_SEAL_FILENAME = "execution-seal.json"
EXECUTION_SEAL_SCHEMA = "tacorank.execution-seal.v1"


class RunArtifactManager:
    """Own the immutable ``artifacts/<run>/<experiment>/attempt_n`` layout."""

    def __init__(
        self,
        store: ExecutionArtifactStore,
        run_id: str,
        experiment_id: str,
        attempt: int,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.attempt = attempt
        self.directory = store.attempt_directory(run_id, experiment_id, attempt)
        self._prefix = self.directory.relative_to(store.artifact_root).as_posix()

    @property
    def log_path(self) -> Path:
        return self.directory / "execution.log"

    @property
    def telemetry_path(self) -> Path:
        return self.directory / "telemetry.jsonl"

    @property
    def execution_seal_path(self) -> Path:
        return self.directory / EXECUTION_SEAL_FILENAME

    @property
    def output_directory(self) -> Path:
        """Only this subtree is writable by the candidate sandbox."""

        path = self.directory / "outputs"
        if path.is_symlink():
            raise ExecutionArtifactPathError("output directory cannot be a symlink")
        path.mkdir(mode=0o700, exist_ok=True)
        return path

    @property
    def temporary_directory(self) -> Path:
        path = self.output_directory / "tmp"
        if path.is_symlink():
            raise ExecutionArtifactPathError("temporary directory cannot be a symlink")
        path.mkdir(mode=0o700, exist_ok=True)
        return path

    def write_resolved_configuration(
        self,
        command: ResolvedCommand,
        *,
        request_summary: Mapping[str, object],
        limits_summary: Mapping[str, object],
    ) -> Any:
        payload = {
            "request": dict(request_summary),
            "command": command.public_configuration(),
            "limits": dict(limits_summary),
        }
        return self.store.write_json(
            self._relative("resolved-command.json"), payload, kind="other"
        )

    def write_environment_identity(self, command: ResolvedCommand) -> Any:
        payload = {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": executable_identity(command.argv[0]),
            "process_environment_keys": sorted(command.environment),
        }
        return self.store.write_json(
            self._relative("environment.json"), payload, kind="other"
        )

    def write_launch_failure(self, summary: str) -> Any:
        if not self.log_path.exists():
            return self.store.write_text(
                self._relative("execution.log"),
                summary.rstrip() + "\n",
                kind="log",
            )
        return self.log_reference()

    def log_reference(self) -> Any:
        return self.store.reference(
            self.log_path,
            kind="log",
            content_type="text/plain; charset=utf-8",
        )

    def telemetry_reference(self) -> Any:
        return self.store.reference(
            self.telemetry_path,
            kind="other",
            content_type="application/x-ndjson",
        )

    def write_execution_seal(
        self,
        *,
        request: Any,
        command: ResolvedCommand,
        prediction_artifact: Any,
        receipt_sha256: str,
    ) -> Any:
        """Write the controller-owned producer binding outside ``outputs/``."""

        destination = self.execution_seal_path
        if destination.exists() or destination.is_symlink():
            raise ExecutionArtifactPathError("execution seal path already exists")
        if re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None:
            raise ExecutionArtifactPathError("verified receipt hash is required")
        payload = {
            "schema": EXECUTION_SEAL_SCHEMA,
            "producer": "tacorank.execution.ExecutionRunner",
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "execution_attempt": self.attempt,
            "producer_commit_sha": str(_field(request, "patch_commit_sha")),
            "command_id": command.command_id,
            "data_manifest_sha256": str(_field(request, "data_manifest_sha256")),
            "patch_receipt_id": str(_field(request, "patch_receipt_id")),
            "patch_receipt_sha256": receipt_sha256,
            "prediction": {
                "path": str(_field(prediction_artifact, "path")),
                "sha256": str(_field(prediction_artifact, "sha256")),
                "size_bytes": int(_field(prediction_artifact, "size_bytes")),
            },
        }
        reference = self.store.write_json(
            self._relative(EXECUTION_SEAL_FILENAME), payload, kind="other"
        )
        _fsync_directory(self.directory)
        return reference

    def capture_outputs(
        self, expected: Tuple[ExpectedArtifact, ...]
    ) -> CapturedOutputs:
        by_role: Dict[str, Any] = {}
        missing = []
        output_root = self.output_directory.resolve(strict=True)
        for item in expected:
            candidate = self.output_directory / item.relative_path
            _reject_symlinks_below(candidate, output_root)
            resolved = candidate.resolve(strict=False)
            try:
                resolved.relative_to(output_root)
            except ValueError as error:
                raise ExecutionArtifactPathError(
                    "expected output escapes attempt root"
                ) from error
            if candidate.is_symlink():
                raise ExecutionArtifactPathError(
                    "expected output cannot be a symlink"
                )
            if not candidate.is_file():
                if item.required:
                    missing.append(item.role)
                continue
            if item.role in by_role:
                raise ExecutionArtifactPathError(
                    "more than one output declared for role {0}".format(item.role)
                )
            by_role[item.role] = self.store.reference(
                candidate,
                kind=item.kind,
                content_type=item.content_type,
            )
        return CapturedOutputs(
            prediction_artifact=by_role.get("prediction"),
            checkpoint_artifact=by_role.get("checkpoint"),
            missing_required_roles=tuple(sorted(missing)),
        )

    def _relative(self, filename: str) -> str:
        return "{0}/{1}".format(self._prefix, filename)


def safe_request_summary(request: Any) -> Mapping[str, object]:
    """Return only non-secret RunRequest fields for resolved-command evidence."""

    allowed = (
        "run_id",
        "experiment_id",
        "attempt",
        "fidelity",
        "command_id",
        "patch_commit_sha",
        "patch_receipt_id",
        "seed",
        "data_manifest_sha256",
        "timeout_seconds",
        "memory_limit_mb",
        "gpu_memory_limit_mb",
        "network_enabled",
    )
    return {name: _field(request, name) for name in allowed}


def verify_execution_seal(
    repository_root: Path,
    prediction_artifact: Any,
    *,
    run_id: str,
    experiment_id: str,
    execution_attempt: int,
    producer_commit_sha: str,
    command_id: str,
    data_manifest_sha256: str,
    patch_receipt_id: str,
    patch_receipt_sha256: str,
) -> Mapping[str, Any]:
    """Verify the deterministic sidecar and the exact prediction bytes.

    This helper intentionally uses plain mappings so Gate B can consume it
    without introducing another shared schema.
    """

    identifier = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
    if not identifier.fullmatch(run_id) or not identifier.fullmatch(experiment_id):
        raise ExecutionSealVerificationError("invalid execution seal identity")
    if (
        not isinstance(execution_attempt, int)
        or isinstance(execution_attempt, bool)
        or execution_attempt < 1
    ):
        raise ExecutionSealVerificationError("execution attempt must be positive")
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", producer_commit_sha) is None:
        raise ExecutionSealVerificationError("producer commit is malformed")
    if identifier.fullmatch(command_id) is None:
        raise ExecutionSealVerificationError("expected command_id is required")
    if re.fullmatch(r"[0-9a-f]{64}", data_manifest_sha256) is None:
        raise ExecutionSealVerificationError("expected data manifest is required")
    if not patch_receipt_id or len(patch_receipt_id) > 256:
        raise ExecutionSealVerificationError("expected patch receipt id is required")
    if re.fullmatch(r"[0-9a-f]{64}", patch_receipt_sha256) is None:
        raise ExecutionSealVerificationError("expected patch receipt hash is required")
    prediction_text = str(_field(prediction_artifact, "path"))
    prediction_relative = PurePosixPath(prediction_text)
    if (
        "\\" in prediction_text
        or prediction_relative.is_absolute()
        or prediction_relative.as_posix() != prediction_text
        or any(part in {"", ".", ".."} for part in prediction_relative.parts)
    ):
        raise ExecutionSealVerificationError("prediction path is not normalized")
    expected_prefix = PurePosixPath(
        "artifacts",
        run_id,
        experiment_id,
        "attempt_{0}".format(execution_attempt),
        "outputs",
    )
    try:
        prediction_relative.relative_to(expected_prefix)
    except ValueError as error:
        raise ExecutionSealVerificationError(
            "prediction is outside the exact execution output directory"
        ) from error
    if prediction_relative == expected_prefix:
        raise ExecutionSealVerificationError("prediction path names no file")

    root = Path(repository_root).resolve(strict=True)
    prediction_path = root.joinpath(*prediction_relative.parts)
    seal_relative = expected_prefix.parent / EXECUTION_SEAL_FILENAME
    seal_path = root.joinpath(*seal_relative.parts)
    _reject_existing_symlink_components(prediction_path)
    _reject_existing_symlink_components(seal_path)
    try:
        prediction_resolved = prediction_path.resolve(strict=True)
        seal_resolved = seal_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ExecutionSealVerificationError("execution seal evidence is missing") from error
    if not prediction_resolved.is_file() or not seal_resolved.is_file():
        raise ExecutionSealVerificationError("execution seal evidence is not a file")

    raw = seal_resolved.read_bytes()
    if len(raw) > 64 * 1024:
        raise ExecutionSealVerificationError("execution seal is unexpectedly large")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutionSealVerificationError("execution seal is malformed") from error
    canonical = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    if raw != canonical or not isinstance(payload, dict):
        raise ExecutionSealVerificationError("execution seal is not canonical JSON")

    required_keys = {
        "schema",
        "producer",
        "run_id",
        "experiment_id",
        "execution_attempt",
        "producer_commit_sha",
        "command_id",
        "data_manifest_sha256",
        "patch_receipt_id",
        "patch_receipt_sha256",
        "prediction",
    }
    if set(payload) != required_keys:
        raise ExecutionSealVerificationError("execution seal has unexpected fields")
    if not identifier.fullmatch(str(payload.get("command_id", ""))):
        raise ExecutionSealVerificationError("execution seal command_id is malformed")
    if re.fullmatch(
        r"[0-9a-f]{64}", str(payload.get("data_manifest_sha256", ""))
    ) is None:
        raise ExecutionSealVerificationError("execution seal data manifest is malformed")
    receipt_id = payload.get("patch_receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id or len(receipt_id) > 256:
        raise ExecutionSealVerificationError("execution seal receipt id is malformed")
    receipt_sha = payload.get("patch_receipt_sha256")
    if receipt_sha is not None and re.fullmatch(r"[0-9a-f]{64}", str(receipt_sha)) is None:
        raise ExecutionSealVerificationError("execution seal receipt hash is malformed")
    expected_values = {
        "schema": EXECUTION_SEAL_SCHEMA,
        "producer": "tacorank.execution.ExecutionRunner",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "execution_attempt": execution_attempt,
        "producer_commit_sha": producer_commit_sha,
    }
    expected_values.update(
        {
            "command_id": command_id,
            "data_manifest_sha256": data_manifest_sha256,
            "patch_receipt_id": patch_receipt_id,
            "patch_receipt_sha256": patch_receipt_sha256,
        }
    )
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise ExecutionSealVerificationError(
                "execution seal {0} mismatch".format(field)
            )

    prediction = payload.get("prediction")
    if not isinstance(prediction, dict) or set(prediction) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ExecutionSealVerificationError("execution seal prediction is malformed")
    actual_sha256 = sha256_file(prediction_resolved)
    actual_size = prediction_resolved.stat().st_size
    artifact_sha256 = str(_field(prediction_artifact, "sha256"))
    artifact_size = int(_field(prediction_artifact, "size_bytes"))
    if not re.fullmatch(r"[0-9a-f]{64}", actual_sha256):
        raise ExecutionSealVerificationError("prediction digest is malformed")
    if prediction != {
        "path": prediction_relative.as_posix(),
        "sha256": actual_sha256,
        "size_bytes": actual_size,
    }:
        raise ExecutionSealVerificationError("execution seal prediction mismatch")
    if artifact_sha256 != actual_sha256 or artifact_size != actual_size:
        raise ExecutionSealVerificationError("prediction ArtifactRef mismatch")
    return payload


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _reject_existing_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ExecutionSealVerificationError(
                "execution seal path contains a symbolic link"
            )
        if current == current.parent:
            return
        current = current.parent


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_symlinks_below(candidate: Path, root: Path) -> None:
    current = candidate
    while True:
        if current.is_symlink():
            raise ExecutionArtifactPathError(
                "expected output path contains a symlink"
            )
        if current == root:
            return
        if current == current.parent:
            raise ExecutionArtifactPathError(
                "expected output path escapes output root"
            )
        current = current.parent
