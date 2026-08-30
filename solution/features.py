"""Candidate-owned KuaiRand feature parsing and encoding helpers.

This module deliberately knows only the label-bearing training view and the
unlabelled scoring view supplied by the controller. Official split selection
and evaluation remain outside the editable ``solution`` package.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np


TRAIN_COLUMNS = (
    "date",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
    "long_view",
)
SCORE_COLUMNS = (
    "row_id",
    "date",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
)
FIELD_NAMES = ("user_id", "video_id", "author_id", "tab", "duration_bucket")


@dataclass(frozen=True)
class TrainingRow:
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float
    long_view: int


@dataclass(frozen=True)
class ScoringRow:
    row_id: int
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float


FeatureRow = TrainingRow | ScoringRow


def read_training_rows(path: Path) -> list[TrainingRow]:
    """Read the controller-created training view with strict schema checks."""

    rows: list[TrainingRow] = []
    for raw in _read_dict_rows(path, TRAIN_COLUMNS):
        label = raw["long_view"]
        if label not in {"0", "1"}:
            raise ValueError("long_view must be encoded as 0 or 1")
        rows.append(
            TrainingRow(
                date=_date(raw["date"]),
                user_id=_category(raw["user_id"], "user_id"),
                video_id=_category(raw["video_id"], "video_id"),
                author_id=_category(raw["author_id"], "author_id"),
                tab=_category(raw["tab"], "tab"),
                duration_ms=_duration(raw["duration_ms"]),
                long_view=int(label),
            )
        )
    if not rows:
        raise ValueError("training population is empty")
    return rows


def read_scoring_rows(path: Path) -> list[ScoringRow]:
    """Read the unlabelled scoring view and enforce ordered contiguous rows."""

    rows: list[ScoringRow] = []
    for expected, raw in enumerate(_read_dict_rows(path, SCORE_COLUMNS)):
        row_id = int(raw["row_id"])
        if row_id != expected:
            raise ValueError("score rows must be contiguous and ordered")
        rows.append(
            ScoringRow(
                row_id=row_id,
                date=_date(raw["date"]),
                user_id=_category(raw["user_id"], "user_id"),
                video_id=_category(raw["video_id"], "video_id"),
                author_id=_category(raw["author_id"], "author_id"),
                tab=_category(raw["tab"], "tab"),
                duration_ms=_duration(raw["duration_ms"]),
            )
        )
    if not rows:
        raise ValueError("scoring population is empty")
    return rows


@dataclass(frozen=True)
class FeatureEncoder:
    """Five-field FM encoder fitted exclusively on training rows."""

    duration_edges: np.ndarray
    vocabularies: Tuple[dict[str, int], ...]
    unknown_ids: Tuple[int, ...]
    offsets: np.ndarray
    dimension: int

    @classmethod
    def fit(
        cls,
        rows: Sequence[TrainingRow],
        *,
        duration_buckets: int = 10,
    ) -> "FeatureEncoder":
        if not rows:
            raise ValueError("cannot fit an encoder without training rows")
        if duration_buckets < 2:
            raise ValueError("duration_buckets must be at least 2")
        edges = np.quantile(
            np.asarray([row.duration_ms for row in rows], dtype=np.float64),
            np.linspace(0.0, 1.0, duration_buckets + 1)[1:-1],
        )
        vocabularies = tuple({} for _ in FIELD_NAMES)
        for row in rows:
            for field, value in enumerate(_raw_fields(row, edges)):
                vocabulary = vocabularies[field]
                if value not in vocabulary:
                    vocabulary[value] = len(vocabulary)
        unknown_ids = tuple(len(vocabulary) for vocabulary in vocabularies)
        field_dimensions = [value + 1 for value in unknown_ids]
        offsets = np.cumsum([0] + field_dimensions[:-1]).astype(np.int32)
        return cls(
            duration_edges=edges,
            vocabularies=vocabularies,
            unknown_ids=unknown_ids,
            offsets=offsets,
            dimension=int(sum(field_dimensions)),
        )

    def transform(self, rows: Sequence[FeatureRow]) -> np.ndarray:
        """Encode rows, mapping scoring-only categories to per-field UNK IDs."""

        encoded = np.empty((len(rows), len(FIELD_NAMES)), dtype=np.int32)
        for row_index, row in enumerate(rows):
            for field, value in enumerate(_raw_fields(row, self.duration_edges)):
                encoded[row_index, field] = (
                    self.vocabularies[field].get(value, self.unknown_ids[field])
                    + self.offsets[field]
                )
        return encoded


def _read_dict_rows(
    path: Path, expected_columns: Sequence[str]
) -> Iterable[dict[str, str]]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("candidate input must be a regular file")
    with candidate.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError("candidate input has an unexpected schema")
        yield from reader


def _raw_fields(row: FeatureRow, duration_edges: np.ndarray) -> Tuple[str, ...]:
    bucket = int(np.searchsorted(duration_edges, row.duration_ms))
    return row.user_id, row.video_id, row.author_id, row.tab, str(bucket)


def _date(value: str) -> int:
    parsed = int(value)
    if parsed < 20_000_000 or parsed > 99_999_999:
        raise ValueError("date must be an integer YYYYMMDD value")
    return parsed


def _duration(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError("duration_ms must be finite and non-negative")
    return parsed


def _category(value: str, name: str) -> str:
    if not value:
        raise ValueError("%s must not be empty" % name)
    return value
