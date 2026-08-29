"""Hash-checked KuaiRand submission validation for final readiness.

Gate B remains Person 3's responsibility.  This adapter gives Person 5 a
deterministic final pre-flight check that matches the official CSV contract.
"""

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tacorank.evaluation.adapter import EvaluationIntegrityError, sha256_file


HEADER = ("row_id", "user_id", "video_id", "score")


@dataclass(frozen=True)
class SubmissionCheck:
    rows: int
    unique_scores: int
    unique_score_fraction: float
    minimum: float
    maximum: float
    scores: Tuple[float, ...]


class KuaiRandSubmissionAdapter:
    def __init__(self, checker_path: Path, expected_checker_sha256: str) -> None:
        self.checker_path = Path(checker_path)
        self.expected_checker_sha256 = str(expected_checker_sha256)

    def check(
        self,
        path: Path,
        expected_rows: Sequence[Sequence[object]],
        minimum_unique_fraction: float = 0.01,
    ) -> SubmissionCheck:
        if sha256_file(self.checker_path) != self.expected_checker_sha256:
            raise EvaluationIntegrityError("protected submission checker hash mismatch")
        return validate_submission(path, expected_rows, minimum_unique_fraction)


def validate_submission(
    path: Path,
    expected_rows: Sequence[Sequence[object]],
    minimum_unique_fraction: float = 0.01,
) -> SubmissionCheck:
    """Validate exact row order, duplicate preservation, and finite scores."""
    if minimum_unique_fraction < 0 or minimum_unique_fraction > 1:
        raise ValueError("minimum_unique_fraction must be in [0, 1]")
    scores: List[float] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if tuple(header or ()) != HEADER:
            raise ValueError("header must be %s" % ",".join(HEADER))
        for expected_index, record in enumerate(reader):
            line_number = expected_index + 2
            if len(record) != 4:
                raise ValueError("line %d must contain four fields" % line_number)
            row_id, user_id, video_id, raw_score = record
            try:
                parsed_row_id = int(row_id)
            except ValueError:
                raise ValueError("line %d row_id is not an integer" % line_number)
            if parsed_row_id != expected_index:
                raise ValueError("line %d row_id must be %d" % (line_number, expected_index))
            if expected_index >= len(expected_rows):
                raise ValueError("submission has more rows than the evaluation population")
            expected = expected_rows[expected_index]
            if len(expected) < 3:
                raise ValueError("expected rows must expose user_id and video_id at indexes 1 and 2")
            if user_id != str(expected[1]) or video_id != str(expected[2]):
                raise ValueError("line %d does not align with the evaluation row" % line_number)
            try:
                score = float(raw_score)
            except ValueError:
                raise ValueError("line %d score is not numeric" % line_number)
            if not math.isfinite(score):
                raise ValueError("line %d score must be finite" % line_number)
            scores.append(score)
    if len(scores) != len(expected_rows):
        raise ValueError(
            "submission row count %d does not match expected %d"
            % (len(scores), len(expected_rows))
        )
    if not scores:
        raise ValueError("submission must not be empty")
    unique = len(set(scores))
    unique_fraction = unique / len(scores)
    if unique_fraction < minimum_unique_fraction:
        raise ValueError(
            "score diversity %.6f is below required %.6f"
            % (unique_fraction, minimum_unique_fraction)
        )
    return SubmissionCheck(
        rows=len(scores),
        unique_scores=unique,
        unique_score_fraction=unique_fraction,
        minimum=min(scores),
        maximum=max(scores),
        scores=tuple(scores),
    )
