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
from typing import Iterable, Optional, Sequence, Tuple, Union

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
TRAIN_AUXILIARY_COLUMNS = (
    "time_ms",
    "hourmin",
    "is_click",
    "play_time_ms",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
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
    time_ms: Optional[int] = None
    hourmin: Optional[int] = None
    is_click: Optional[int] = None
    play_time_ms: Optional[float] = None
    is_like: Optional[int] = None
    is_follow: Optional[int] = None
    is_comment: Optional[int] = None
    is_forward: Optional[int] = None
    is_hate: Optional[int] = None
    is_profile_enter: Optional[int] = None


@dataclass(frozen=True)
class ScoringRow:
    row_id: int
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float


FeatureRow = Union[TrainingRow, ScoringRow]


def read_training_rows(path: Path) -> list[TrainingRow]:
    """Read the controller-created training view with strict schema checks."""

    rows: list[TrainingRow] = []
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("candidate input must be a regular file")
    with candidate.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = tuple(reader.fieldnames or ())
        allowed = set(TRAIN_COLUMNS).union(TRAIN_AUXILIARY_COLUMNS)
        if not set(TRAIN_COLUMNS).issubset(fields) or set(fields) - allowed:
            raise ValueError("candidate training input has an unexpected schema")
        for raw in reader:
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
                    time_ms=_optional_int(raw.get("time_ms")),
                    hourmin=_optional_int(raw.get("hourmin")),
                    is_click=_optional_binary(raw.get("is_click"), "is_click"),
                    play_time_ms=_optional_nonnegative(
                        raw.get("play_time_ms"), "play_time_ms"
                    ),
                    is_like=_optional_binary(raw.get("is_like"), "is_like"),
                    is_follow=_optional_binary(raw.get("is_follow"), "is_follow"),
                    is_comment=_optional_binary(raw.get("is_comment"), "is_comment"),
                    is_forward=_optional_binary(raw.get("is_forward"), "is_forward"),
                    is_hate=_optional_binary(raw.get("is_hate"), "is_hate"),
                    is_profile_enter=_optional_binary(
                        raw.get("is_profile_enter"), "is_profile_enter"
                    ),
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


def _optional_int(value: Optional[str]) -> Optional[int]:
    return None if value in (None, "") else int(value)


def _optional_binary(value: Optional[str], name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    parsed = int(float(value))
    if parsed not in (0, 1):
        raise ValueError("%s must be binary" % name)
    return parsed


def _optional_nonnegative(value: Optional[str], name: str) -> Optional[float]:
    if value in (None, ""):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError("%s must be finite and non-negative" % name)
    return parsed
