"""Hash-checked KuaiRand submission validation for final readiness.

Gate B remains Person 3's responsibility.  This adapter gives Person 5 a
deterministic final pre-flight check that matches the official CSV contract.
"""

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

from tacorank.evaluation.adapter import (
    EvaluationIntegrityError,
    OutputGateEvidence,
    PredictionBatch,
    ordered_prediction_sha256,
    ordered_row_identity_sha256,
    sha256_file,
)
from tacorank.evaluation.types import Population


HEADER = ("row_id", "user_id", "video_id", "score")


@dataclass(frozen=True)
class SubmissionCheck:
    rows: int
    unique_scores: int
    unique_score_fraction: float
    minimum: float
    maximum: float
    scores: Tuple[float, ...]
    artifact_sha256: str
    row_ids: Tuple[int, ...]
    user_ids: Tuple[str, ...]
    item_ids: Tuple[str, ...]
    ordered_row_identity_sha256: str
    ordered_prediction_sha256: str

    def prediction_batch(self, artifact_id: str) -> PredictionBatch:
        return PredictionBatch(
            artifact_id=artifact_id,
            artifact_sha256=self.artifact_sha256,
            row_ids=self.row_ids,
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            scores=self.scores,
        )

    def gate_evidence(
        self,
        event_id: str,
        artifact_id: str,
        population: Population,
    ) -> OutputGateEvidence:
        return OutputGateEvidence(
            event_id=event_id,
            accepted=True,
            prediction_artifact_id=artifact_id,
            prediction_artifact_sha256=self.artifact_sha256,
            population=population,
            ordered_row_identity_sha256=self.ordered_row_identity_sha256,
            ordered_prediction_sha256=self.ordered_prediction_sha256,
        )


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
    row_ids: List[int] = []
    user_ids: List[str] = []
    item_ids: List[str] = []
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
                raise ValueError(
                    "line %d row_id must be %d" % (line_number, expected_index)
                )
            if expected_index >= len(expected_rows):
                raise ValueError(
                    "submission has more rows than the evaluation population"
                )
            expected = expected_rows[expected_index]
            if isinstance(expected, Mapping):
                if "user_id" not in expected or "video_id" not in expected:
                    raise ValueError(
                        "expected row mappings must expose user_id and video_id"
                    )
                expected_user = expected["user_id"]
                expected_video = expected["video_id"]
            else:
                if len(expected) < 3:
                    raise ValueError(
                        "expected rows must expose user_id and video_id at indexes 1 and 2"
                    )
                expected_user = expected[1]
                expected_video = expected[2]
            if user_id != str(expected_user) or video_id != str(expected_video):
                raise ValueError(
                    "line %d does not align with the evaluation row" % line_number
                )
            try:
                score = float(raw_score)
            except ValueError:
                raise ValueError("line %d score is not numeric" % line_number)
            if not math.isfinite(score):
                raise ValueError("line %d score must be finite" % line_number)
            row_ids.append(parsed_row_id)
            user_ids.append(user_id)
            item_ids.append(video_id)
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
    row_identity = ordered_row_identity_sha256(row_ids, user_ids, item_ids)
    prediction_identity = ordered_prediction_sha256(
        row_ids, user_ids, item_ids, scores
    )
    return SubmissionCheck(
        rows=len(scores),
        unique_scores=unique,
        unique_score_fraction=unique_fraction,
        minimum=min(scores),
        maximum=max(scores),
        scores=tuple(scores),
        artifact_sha256=sha256_file(path),
        row_ids=tuple(row_ids),
        user_ids=tuple(user_ids),
        item_ids=tuple(item_ids),
        ordered_row_identity_sha256=row_identity,
        ordered_prediction_sha256=prediction_identity,
    )
