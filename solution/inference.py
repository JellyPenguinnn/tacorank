"""Candidate-owned parent-score and prediction-output helpers."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from solution.features import ScoringRow


def load_verified_parent(input_root: Path, rows: Sequence[ScoringRow]) -> np.ndarray:
    """Load the authenticated FM parent and verify exact score-row alignment."""

    root = Path(input_root)
    parent_path = _regular_file(root / "fm_baseline_predictions.csv")
    digest_path = _regular_file(root / "fm_baseline_predictions.sha256")
    expected_digest = digest_path.read_text(encoding="ascii").strip()
    if not _is_sha256(expected_digest) or _sha256_file(parent_path) != expected_digest:
        raise ValueError("frozen FM prediction identity is invalid")
    scores: list[float] = []
    with parent_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != ["row_id", "user_id", "video_id", "score"]:
            raise ValueError("frozen FM predictions have an invalid header")
        for expected, (score_row, parent_row) in enumerate(zip(rows, reader)):
            score = float(parent_row["score"])
            if (
                score_row.row_id != expected
                or int(parent_row["row_id"]) != expected
                or parent_row["user_id"] != score_row.user_id
                or parent_row["video_id"] != score_row.video_id
            ):
                raise ValueError("frozen FM predictions do not align with score.csv")
            if not math.isfinite(score):
                raise ValueError("frozen FM predictions must be finite")
            scores.append(score)
        if next(reader, None) is not None or len(scores) != len(rows):
            raise ValueError("frozen FM predictions have the wrong row count")
    return np.asarray(scores, dtype=np.float64)


def add_bounded_residual(
    parent_scores: np.ndarray,
    residual_scores: np.ndarray,
    *,
    maximum_absolute_residual: float,
) -> np.ndarray:
    """Add a bounded residual without transforming the FM parent's scale."""

    if maximum_absolute_residual <= 0.0:
        raise ValueError("maximum_absolute_residual must be positive")
    parent = np.asarray(parent_scores, dtype=np.float64)
    residual = np.asarray(residual_scores, dtype=np.float64)
    if parent.shape != residual.shape or parent.ndim != 1:
        raise ValueError("parent and residual scores must be aligned vectors")
    if not np.isfinite(parent).all() or not np.isfinite(residual).all():
        raise ValueError("parent and residual scores must be finite")
    bounded = maximum_absolute_residual * np.tanh(
        residual / maximum_absolute_residual
    )
    combined = parent + bounded
    if not np.isfinite(combined).all():
        raise ValueError("combined scores must be finite")
    return combined


def write_predictions_exclusive(
    output_path: Path,
    rows: Sequence[ScoringRow],
    scores: np.ndarray,
) -> None:
    """Create exactly one ordered prediction CSV without overwriting evidence."""

    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("prediction scores must be aligned and finite")
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            descriptor = -1
            writer = csv.writer(handle)
            writer.writerow(("row_id", "user_id", "video_id", "score"))
            for row, score in zip(rows, values):
                writer.writerow((row.row_id, row.user_id, row.video_id, repr(float(score))))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _regular_file(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("required candidate input is missing")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError("candidate inputs must use canonical paths")
    return resolved
