"""Executable FM-parity parent for TacoRank research candidates.

The deployment supplies predictions from the frozen official FM for the exact
ordered score population. The baseline implementation validates and copies
those scores. Research patches should keep them as the strong parent and add
one bounded, train-only residual unless an approved experiment explicitly
tests replacing the parent.

Bound that residual relative to the spread of the parent scores, not to an
absolute constant. The parent scores span several units, so a cap chosen far
below one of their standard deviations leaves the ranking indistinguishable
from this baseline: the evaluation gate measures the fraction of within-user
item pairs a candidate reorders, and one that reorders essentially nothing is
recorded as a no-op rather than as a result. Only within-user ordering is
scored, so a term that is constant inside a single user's list cannot move
GAUC or nDCG@5 however large it is.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterator, Tuple


PredictionRow = Tuple[int, str, str, float]


def run(invocation: Any) -> None:
    """Reproduce the frozen official FM on the requested ordered population."""

    score_path = _regular_file(invocation.input_root / "score.csv")
    parent_path = _regular_file(
        invocation.input_root / "fm_baseline_predictions.csv"
    )
    digest_path = _regular_file(
        invocation.input_root / "fm_baseline_predictions.sha256"
    )
    expected_digest = digest_path.read_text(encoding="ascii").strip()
    if not _is_sha256(expected_digest) or _sha256_file(parent_path) != expected_digest:
        raise ValueError("frozen FM prediction identity is invalid")

    with score_path.open(newline="", encoding="utf-8") as score_handle:
        score_rows = csv.DictReader(score_handle, strict=True)
        required = {"row_id", "user_id", "video_id"}
        if not required.issubset(score_rows.fieldnames or ()):
            raise ValueError("score.csv is missing required columns")
        for expected, score_row, parent_row in _aligned_rows(
            score_rows, _prediction_rows(parent_path)
        ):
            row_id, user_id, video_id, _ = parent_row
            if int(score_row["row_id"]) != expected or row_id != expected:
                raise ValueError("candidate rows must be contiguous and ordered")
            if score_row["user_id"] != user_id or score_row["video_id"] != video_id:
                raise ValueError("frozen FM predictions do not align with score.csv")

    _exclusive_copy(parent_path, invocation.output_path)


def _prediction_rows(path: Path) -> Iterator[PredictionRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != ["row_id", "user_id", "video_id", "score"]:
            raise ValueError("frozen FM predictions have an invalid header")
        for row in reader:
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError("frozen FM predictions must be finite")
            yield int(row["row_id"]), row["user_id"], row["video_id"], score


def _aligned_rows(
    score_rows: Iterator[dict[str, str]],
    parent_rows: Iterator[PredictionRow],
) -> Iterator[Tuple[int, dict[str, str], PredictionRow]]:
    sentinel = object()
    expected = 0
    while True:
        score_row = next(score_rows, sentinel)
        parent_row = next(parent_rows, sentinel)
        if score_row is sentinel and parent_row is sentinel:
            if expected == 0:
                raise ValueError("score population is empty")
            return
        if score_row is sentinel or parent_row is sentinel:
            raise ValueError("frozen FM predictions have the wrong row count")
        yield expected, score_row, parent_row
        expected += 1


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


def _exclusive_copy(source: Path, destination: Path) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _regular_file(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("required candidate input is missing")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError("candidate inputs must use canonical paths")
    return resolved
