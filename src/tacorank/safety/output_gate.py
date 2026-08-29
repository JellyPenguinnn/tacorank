"""Gate B: contract-defined prediction structure and alignment checks."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from .path_policy import (
    PolicyViolation,
    ViolationCode,
    normalize_policy_path,
    path_is_within,
)
from .receipts import SharedSchemaFactories


OUTPUT_CHECK_ORDER = (
    "header",
    "column_types",
    "row_count",
    "row_id",
    "identity_alignment",
    "duplicate_preservation",
    "finite_scores",
    "score_diversity",
    "artifact_identity",
    "producer_commit",
    "protected_data",
)


class RunResultLike(Protocol):
    run_id: str
    experiment_id: str
    attempt: int
    patch_commit_sha: str
    prediction_artifact: Any


@dataclass(frozen=True)
class ExecutionSealExpectation:
    """Controller-owned identities that Gate B must bind to the run seal."""

    run_id: str
    experiment_id: str
    execution_attempt: int
    producer_commit_sha: str
    command_id: str
    data_manifest_sha256: str
    patch_receipt_id: str
    patch_receipt_sha256: str

    def __post_init__(self) -> None:
        identity = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
        for name in ("run_id", "experiment_id", "command_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or identity.fullmatch(value) is None
                or value in {".", ".."}
                or ".." in value
                or value.endswith(".lock")
            ):
                raise ValueError("{} is not a valid sealed identity".format(name))
        if (
            isinstance(self.execution_attempt, bool)
            or not isinstance(self.execution_attempt, int)
            or self.execution_attempt < 1
        ):
            raise ValueError("execution_attempt must be a positive integer")
        if re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", self.producer_commit_sha
        ) is None:
            raise ValueError("producer_commit_sha must be a lowercase Git object id")
        if re.fullmatch(r"[0-9a-f]{64}", self.data_manifest_sha256) is None:
            raise ValueError("data_manifest_sha256 must be a lowercase SHA-256")
        if (
            not isinstance(self.patch_receipt_id, str)
            or not self.patch_receipt_id
            or len(self.patch_receipt_id) > 256
            or any(character in self.patch_receipt_id for character in "\r\n\0")
        ):
            raise ValueError("patch_receipt_id is malformed")
        if re.fullmatch(r"[0-9a-f]{64}", self.patch_receipt_sha256) is None:
            raise ValueError("patch_receipt_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class OutputColumn:
    name: str
    kind: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("output column name cannot be empty")
        if self.kind not in ("integer", "number", "string", "boolean"):
            raise ValueError("unsupported output column kind: {}".format(self.kind))


@dataclass(frozen=True)
class OutputContract:
    """Frozen structural contract; it deliberately contains no metric logic."""

    columns: Tuple[OutputColumn, ...]
    score_column: str
    expected_rows: Tuple[Mapping[str, Any], ...]
    identity_columns: Tuple[str, ...]
    row_id_column: Optional[str] = "row_id"
    require_contiguous_row_id: bool = True
    forbidden_columns: Tuple[str, ...] = ()
    minimum_unique_scores: int = 2

    def __post_init__(self) -> None:
        names = tuple(column.name for column in self.columns)
        if not names or len(set(names)) != len(names):
            raise ValueError("output columns must be non-empty and unique")
        if self.score_column not in names:
            raise ValueError("score_column must be present in output columns")
        if any(column not in names for column in self.identity_columns):
            raise ValueError("identity columns must be present in output columns")
        if self.row_id_column is not None and self.row_id_column not in names:
            raise ValueError("row_id_column must be present in output columns")
        if self.minimum_unique_scores < 1:
            raise ValueError("minimum_unique_scores must be positive")
        for index, row in enumerate(self.expected_rows):
            missing = [name for name in self.identity_columns if name not in row]
            if missing:
                raise ValueError(
                    "expected row {} lacks identity columns: {}".format(
                        index, ", ".join(missing)
                    )
                )
            if self.row_id_column is not None and self.row_id_column not in row:
                raise ValueError("expected row {} lacks row id".format(index))

    @property
    def header(self) -> Tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def kinds(self) -> Mapping[str, str]:
        return {column.name: column.kind for column in self.columns}


class OutputGate:
    """Validate prediction bytes before any Person 5 evaluator can see them."""

    def __init__(
        self,
        *,
        repository_root: Path,
        contract: OutputContract,
        factories: Optional[SharedSchemaFactories] = None,
        artifact_roots: Sequence[str] = ("artifacts",),
        max_prediction_bytes: int = 256 * 1024 * 1024,
        execution_seal_verifier: Optional[Callable[..., Mapping[str, Any]]] = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.contract = contract
        self.factories = factories
        self.artifact_roots = tuple(
            normalize_policy_path(root.rstrip("/")) for root in artifact_roots
        )
        if not self.artifact_roots:
            raise ValueError("at least one prediction artifact root is required")
        if (
            isinstance(max_prediction_bytes, bool)
            or not isinstance(max_prediction_bytes, int)
            or max_prediction_bytes < 1
        ):
            raise ValueError("max_prediction_bytes must be a positive integer")
        self.max_prediction_bytes = max_prediction_bytes
        self.execution_seal_verifier = (
            execution_seal_verifier or _verify_execution_seal
        )

    async def check(
        self,
        run_result: RunResultLike,
        *,
        expected_execution: ExecutionSealExpectation,
    ) -> Any:
        return await asyncio.to_thread(
            self._check_sync,
            run_result,
            expected_execution=expected_execution,
        )

    def _check_sync(
        self,
        run_result: RunResultLike,
        *,
        expected_execution: ExecutionSealExpectation,
    ) -> Any:
        factories = self.factories or SharedSchemaFactories.from_shared_module()
        findings: list[PolicyViolation] = []
        statuses = {name: "pass" for name in OUTPUT_CHECK_ORDER}
        artifact_ref = _field(run_result, "prediction_artifact")
        prediction_bytes = self._resolve_artifact(
            artifact_ref,
            run_result,
            findings,
        )

        run_identity = {
            "run_id": _field(run_result, "run_id"),
            "experiment_id": _field(run_result, "experiment_id"),
            "execution_attempt": _field(run_result, "attempt"),
            "producer_commit_sha": _field(run_result, "patch_commit_sha"),
        }
        expected_run_identity = {
            "run_id": expected_execution.run_id,
            "experiment_id": expected_execution.experiment_id,
            "execution_attempt": expected_execution.execution_attempt,
            "producer_commit_sha": expected_execution.producer_commit_sha,
        }
        if run_identity != expected_run_identity:
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_PRODUCER_MISMATCH,
                    "producer_commit",
                    "run result identity differs from the controller-owned execution expectation",
                    _field(artifact_ref, "path"),
                )
            )
        elif prediction_bytes is not None:
            try:
                seal = self.execution_seal_verifier(
                    self.repository_root,
                    artifact_ref,
                    run_id=expected_execution.run_id,
                    experiment_id=expected_execution.experiment_id,
                    execution_attempt=expected_execution.execution_attempt,
                    producer_commit_sha=expected_execution.producer_commit_sha,
                    command_id=expected_execution.command_id,
                    data_manifest_sha256=expected_execution.data_manifest_sha256,
                    patch_receipt_id=expected_execution.patch_receipt_id,
                    patch_receipt_sha256=expected_execution.patch_receipt_sha256,
                )
                if (
                    seal.get("producer_commit_sha")
                    != expected_execution.producer_commit_sha
                ):
                    raise ValueError("execution seal returned a different producer")
            except Exception as exc:
                findings.append(
                    PolicyViolation(
                        ViolationCode.OUTPUT_PRODUCER_MISMATCH,
                        "producer_commit",
                        "trusted execution-seal evidence is missing or invalid: {}".format(
                            type(exc).__name__
                        ),
                        _field(artifact_ref, "path"),
                    )
                )

        header: Tuple[str, ...] = ()
        rows = []
        if prediction_bytes is not None:
            try:
                decoded_prediction = prediction_bytes.decode("utf-8", errors="strict")
                with io.StringIO(decoded_prediction, newline="") as handle:
                    reader = csv.DictReader(handle, strict=True)
                    header = tuple(reader.fieldnames or ())
                    rows = list(reader)
                    if any(None in row for row in rows):
                        findings.append(
                            PolicyViolation(
                                ViolationCode.OUTPUT_HEADER_MISMATCH,
                                "header",
                                "prediction row contains more fields than the exact header",
                                _field(artifact_ref, "path"),
                            )
                        )
            except (UnicodeDecodeError, csv.Error, OSError):
                findings.append(
                    PolicyViolation(
                        ViolationCode.OUTPUT_HEADER_MISMATCH,
                        "header",
                        "prediction artifact is not a readable strict UTF-8 CSV",
                        _field(artifact_ref, "path"),
                    )
                )

        if header != self.contract.header:
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_HEADER_MISMATCH,
                    "header",
                    "prediction header differs from the frozen contract",
                    _field(artifact_ref, "path"),
                )
            )
        forbidden = sorted(set(header).intersection(self.contract.forbidden_columns))
        if forbidden:
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_PROTECTED_DATA,
                    "protected_data",
                    "prediction output includes protected target or hidden-data columns",
                    _field(artifact_ref, "path"),
                )
            )

        if len(rows) != len(self.contract.expected_rows):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ROW_COUNT_MISMATCH,
                    "row_count",
                    "prediction row count differs from the frozen population",
                    _field(artifact_ref, "path"),
                )
            )

        converted_rows, conversion_findings = self._convert_rows(rows, artifact_ref)
        findings.extend(conversion_findings)
        self._check_row_ids(converted_rows, findings, artifact_ref)
        self._check_identities(converted_rows, findings, artifact_ref)

        scores = []
        for row in converted_rows:
            score = row.get(self.contract.score_column)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                continue
            if not math.isfinite(float(score)):
                findings.append(
                    PolicyViolation(
                        ViolationCode.OUTPUT_NONFINITE_SCORE,
                        "finite_scores",
                        "prediction score contains NaN or infinity",
                        _field(artifact_ref, "path"),
                    )
                )
            else:
                scores.append(float(score))
        if len(scores) != len(rows):
            # Non-numeric values are reported by column_types; make the finite
            # score invariant fail independently without duplicating evidence.
            if not any(
                finding.check == "finite_scores" for finding in findings
            ):
                findings.append(
                    PolicyViolation(
                        ViolationCode.OUTPUT_NONFINITE_SCORE,
                        "finite_scores",
                        "every prediction row must contain a finite numeric score",
                        _field(artifact_ref, "path"),
                    )
                )
        required_unique = min(self.contract.minimum_unique_scores, len(rows))
        if scores and len(set(scores)) < required_unique:
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_DEGENERATE_SCORES,
                    "score_diversity",
                    "prediction scores do not provide sufficient ordering diversity",
                    _field(artifact_ref, "path"),
                )
            )

        for finding in _deduplicate_findings(findings):
            statuses[finding.check] = "fail"
        score_stats = _score_stats(scores, len(rows))
        shared_violations = [
            SharedSchemaFactories.build(factories.violation, finding.as_payload())
            for finding in _deduplicate_findings(findings)
        ]
        payload = {
            "run_id": _field(run_result, "run_id"),
            "experiment_id": _field(run_result, "experiment_id"),
            "attempt": _field(run_result, "attempt"),
            "prediction_artifact": artifact_ref,
            "accepted": not findings,
            "checks": statuses,
            "score_stats": score_stats,
            "violations": shared_violations,
        }
        return SharedSchemaFactories.build(factories.output_check_result, payload)


    def _resolve_artifact(
        self,
        artifact_ref: Any,
        run_result: RunResultLike,
        findings: list,
    ) -> Optional[bytes]:
        relative = _field(artifact_ref, "path")
        if not isinstance(relative, str):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "prediction artifact reference has no relative path",
                )
            )
            return None
        try:
            normalized = normalize_policy_path(relative)
        except ValueError:
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "prediction artifact path is absolute, non-canonical, or traversing",
                    relative,
                )
            )
            return None
        try:
            run_id = _safe_identity_component(_field(run_result, "run_id"), "run_id")
            experiment_id = _safe_identity_component(
                _field(run_result, "experiment_id"), "experiment_id"
            )
            attempt = _field(run_result, "attempt")
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
                raise ValueError("attempt must be a positive integer")
            expected_output_roots = tuple(
                "{}/{}/{}/attempt_{}/outputs".format(
                    root,
                    run_id,
                    experiment_id,
                    attempt,
                )
                for root in self.artifact_roots
            )
        except ValueError:
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "run identity cannot resolve an exact output artifact subtree",
                    normalized,
                )
            )
            return None
        if not any(
            normalized != expected_root
            and path_is_within(normalized, expected_root)
            for expected_root in expected_output_roots
        ):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "prediction artifact is outside the exact run/experiment/attempt outputs subtree",
                    normalized,
                )
            )
            return None
        target = self.repository_root.joinpath(*normalized.split("/"))
        if _field(artifact_ref, "kind") != "predictions":
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "artifact kind is not predictions",
                    normalized,
                )
            )
            return None
        if target.is_symlink() or not target.is_file():
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "prediction artifact is missing or is a symbolic link",
                    normalized,
                )
            )
            return None
        current = self.repository_root
        for part in normalized.split("/"):
            current = current / part
            if current.is_symlink():
                findings.append(
                    PolicyViolation(
                        ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                        "artifact_identity",
                        "prediction artifact path traverses a symbolic link",
                        normalized,
                    )
                )
                return None
        resolved = target.resolve(strict=True)
        if os.path.commonpath((str(self.repository_root), str(resolved))) != str(
            self.repository_root
        ):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "prediction artifact escapes repository root",
                    normalized,
                )
            )
            return None
        expected_sha = _field(artifact_ref, "sha256")
        expected_size = _field(artifact_ref, "size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or expected_size > self.max_prediction_bytes
        ):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "prediction artifact size exceeds the configured bound",
                    normalized,
                )
            )
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(str(target), flags)
            try:
                file_status = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(file_status.st_mode)
                    or file_status.st_size != expected_size
                    or file_status.st_size > self.max_prediction_bytes
                ):
                    raise ValueError("prediction artifact stat identity differs")
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    encoded = handle.read(self.max_prediction_bytes + 1)
            finally:
                os.close(descriptor)
        except (OSError, ValueError):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "prediction artifact could not be read through a bounded regular-file handle",
                    normalized,
                )
            )
            return None
        if (
            len(encoded) > self.max_prediction_bytes
            or not isinstance(expected_sha, str)
            or hashlib.sha256(encoded).hexdigest() != expected_sha
            or len(encoded) != expected_size
        ):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ARTIFACT_MISMATCH,
                    "artifact_identity",
                    "prediction artifact bytes differ from the artifact reference",
                    normalized,
                )
            )
            return None
        return encoded

    def _convert_rows(
        self, rows: Sequence[Mapping[str, Optional[str]]], artifact_ref: Any
    ) -> Tuple[Sequence[Mapping[str, Any]], Tuple[PolicyViolation, ...]]:
        converted = []
        findings = []
        kinds = self.contract.kinds
        for row_index, row in enumerate(rows):
            output: Dict[str, Any] = {}
            for name in self.contract.header:
                raw_value = row.get(name)
                try:
                    output[name] = _convert_value(raw_value, kinds[name])
                except ValueError:
                    findings.append(
                        PolicyViolation(
                            ViolationCode.OUTPUT_TYPE_MISMATCH,
                            "column_types",
                            "column {!r} has an invalid value at row {}".format(
                                name, row_index
                            ),
                            _field(artifact_ref, "path"),
                        )
                    )
            converted.append(output)
        return converted, _deduplicate_findings(findings)

    def _check_row_ids(
        self,
        rows: Sequence[Mapping[str, Any]],
        findings: list,
        artifact_ref: Any,
    ) -> None:
        name = self.contract.row_id_column
        if name is None or not self.contract.require_contiguous_row_id:
            return
        actual = [row.get(name) for row in rows]
        if actual != list(range(len(rows))):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_ROW_ID_MISMATCH,
                    "row_id",
                    "row ids must be zero-based and contiguous in official order",
                    _field(artifact_ref, "path"),
                )
            )

    def _check_identities(
        self,
        rows: Sequence[Mapping[str, Any]],
        findings: list,
        artifact_ref: Any,
    ) -> None:
        identity_columns = self.contract.identity_columns
        actual = [tuple(row.get(name) for name in identity_columns) for row in rows]
        expected = [
            tuple(row.get(name) for name in identity_columns)
            for row in self.contract.expected_rows
        ]
        if actual != expected:
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_IDENTITY_MISMATCH,
                    "identity_alignment",
                    "user/item identities or official row order do not match",
                    _field(artifact_ref, "path"),
                )
            )
        if Counter(actual) != Counter(expected):
            findings.append(
                PolicyViolation(
                    ViolationCode.OUTPUT_IDENTITY_MISMATCH,
                    "duplicate_preservation",
                    "repeated identity rows were collapsed, added, or removed",
                    _field(artifact_ref, "path"),
                )
            )


class FakeOutputGate:
    """Return a caller-supplied shared result for orchestration tests."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []

    async def check(self, run_result: Any, **kwargs: Any) -> Any:
        self.calls.append(run_result)
        del kwargs
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _verify_execution_seal(
    repository_root: Path,
    prediction_artifact: Any,
    **identity: Any,
) -> Mapping[str, Any]:
    # Lazy import avoids a safety <-> execution import cycle while retaining a
    # fail-closed production default.
    from tacorank.execution import verify_execution_seal

    return verify_execution_seal(
        repository_root,
        prediction_artifact,
        **identity,
    )


def _convert_value(value: Optional[str], kind: str) -> Any:
    if value is None:
        raise ValueError("missing value")
    if kind == "string":
        return value
    if kind == "integer":
        if re.fullmatch(r"[+-]?[0-9]+", value) is None:
            raise ValueError("not an integer")
        return int(value)
    if kind == "number":
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("not a number") from exc
    if kind == "boolean":
        folded = value.casefold()
        if folded in ("true", "1"):
            return True
        if folded in ("false", "0"):
            return False
        raise ValueError("not a boolean")
    raise ValueError("unknown type")


def _score_stats(scores: Sequence[float], total_rows: int) -> Mapping[str, Any]:
    if not scores:
        return {
            "count": total_rows,
            "finite_count": 0,
            "unique_count": 0,
        }
    return {
        "count": total_rows,
        "finite_count": len(scores),
        "unique_count": len(set(scores)),
        "minimum": min(scores),
        "maximum": max(scores),
        "mean": sum(scores) / len(scores),
    }


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


def _deduplicate_findings(
    findings: Iterable[PolicyViolation],
) -> Tuple[PolicyViolation, ...]:
    unique = {}
    for finding in findings:
        key = (finding.code.value, finding.check, finding.path, finding.message)
        unique[key] = finding
    order = sorted(unique, key=lambda item: tuple(part or "" for part in item))
    return tuple(unique[key] for key in order)
