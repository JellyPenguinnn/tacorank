"""Reviewed KuaiRand entrypoints for the symbolic execution registry."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, List, Mapping

from tacorank.evaluation.adapter import sha256_file

from .submission_adapter import validate_submission


def run_baseline(invocation: Any) -> None:
    """Copy an operator-frozen baseline prediction into the sealed output path.

    The input view must contain ``baseline_predictions.csv`` and its adjacent
    ``baseline_predictions.sha256`` file.  This entrypoint does not train,
    score, or infer a baseline identity from mutable code.
    """

    if invocation.mode != "baseline" or invocation.fidelity != "full":
        raise ValueError("KuaiRand baseline entrypoint requires baseline/full mode")
    source = _regular_file(invocation.input_root / "baseline_predictions.csv")
    digest_file = _regular_file(
        invocation.input_root / "baseline_predictions.sha256"
    )
    expected = digest_file.read_text(encoding="ascii").strip()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("baseline prediction hash is malformed")
    if sha256_file(source) != expected:
        raise ValueError("baseline prediction bytes do not match the frozen hash")
    _exclusive_copy(source, invocation.output_path)


def check_submission(invocation: Any) -> None:
    """Validate the exact verified prediction against frozen submission rows."""

    rows_path = _regular_file(invocation.contract_root / "submission_rows.csv")
    rows: List[Mapping[str, object]] = []
    with rows_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not {"row_id", "user_id", "video_id"}.issubset(reader.fieldnames or ()):
            raise ValueError("submission_rows.csv has an invalid header")
        for expected, record in enumerate(reader):
            if int(record["row_id"]) != expected:
                raise ValueError("submission rows are not contiguous and ordered")
            rows.append(
                {
                    "row_id": expected,
                    "user_id": record["user_id"],
                    "video_id": record["video_id"],
                }
            )
    validate_submission(invocation.prediction_path, rows)


def _regular_file(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("required KuaiRand pipeline file is missing")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError("KuaiRand pipeline files must use canonical paths")
    return resolved


def _exclusive_copy(source: Path, destination: Path) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
