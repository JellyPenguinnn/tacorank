"""Read-only aggregate EDA tools for the research planner.

The toolbox deliberately understands only the candidate-visible full input view.
It never accepts an arbitrary path, column, expression, or query and never emits
raw rows or entity identifiers.  Training labels may be summarized; score labels
cannot be read because the accepted score schema does not contain one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..schemas import (
    PlannerDataProfile,
    PlannerEdaNumericSummary,
    PlannerEdaOverlapSummary,
    PlannerEdaRateSlice,
)


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
ENTITY_COLUMNS = ("user_id", "video_id", "author_id")
CARDINALITY_COLUMNS = ENTITY_COLUMNS + ("tab",)
EDA_TOOL_IDS = (
    "dataset_overview",
    "target_distribution",
    "temporal_distribution",
    "duration_distribution",
    "entity_sparsity",
    "score_train_overlap",
)
_MAX_RATE_SLICES = 32


class PlannerEdaError(RuntimeError):
    """The approved planner data view could not be safely inspected."""


class _ViewAccumulator:
    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self.rows = 0
        self.missing = {column: 0 for column in columns}
        self.dates: List[int] = []
        self.durations: List[float] = []
        self.entities: Dict[str, Set[str]] = {
            column: set() for column in CARDINALITY_COLUMNS
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rounded(value: float) -> float:
    return round(float(value), 8)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _numeric_summary(values: Iterable[float]) -> PlannerEdaNumericSummary:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise PlannerEdaError("planner EDA numeric input is empty")
    if any(not math.isfinite(value) for value in ordered):
        raise PlannerEdaError("planner EDA numeric input is non-finite")
    return PlannerEdaNumericSummary(
        count=len(ordered),
        minimum=_rounded(ordered[0]),
        p25=_rounded(_percentile(ordered, 0.25)),
        p50=_rounded(_percentile(ordered, 0.50)),
        p75=_rounded(_percentile(ordered, 0.75)),
        p90=_rounded(_percentile(ordered, 0.90)),
        p95=_rounded(_percentile(ordered, 0.95)),
        p99=_rounded(_percentile(ordered, 0.99)),
        maximum=_rounded(ordered[-1]),
        mean=_rounded(math.fsum(ordered) / len(ordered)),
    )


def _parse_date(value: str, *, view: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise PlannerEdaError(
            "%s date is invalid at data row %d" % (view, row_number)
        ) from exc
    if parsed < 10_000_000 or parsed > 99_999_999:
        raise PlannerEdaError(
            "%s date is outside YYYYMMDD form at data row %d" % (view, row_number)
        )
    return parsed


def _parse_duration(value: str, *, view: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise PlannerEdaError(
            "%s duration_ms is invalid at data row %d" % (view, row_number)
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise PlannerEdaError(
            "%s duration_ms is non-finite or negative at data row %d"
            % (view, row_number)
        )
    return parsed


def _prepare_row(
    raw: Mapping[str, str],
    *,
    expected_columns: Sequence[str],
    view: str,
    row_number: int,
) -> Dict[str, str]:
    if None in raw:
        raise PlannerEdaError(
            "%s contains an over-wide row at data row %d" % (view, row_number)
        )
    row = {}
    for column in expected_columns:
        value = raw.get(column)
        if value is None:
            raise PlannerEdaError(
                "%s contains a short row at data row %d" % (view, row_number)
            )
        row[column] = value.strip()
    return row


def _update_common(
    accumulator: _ViewAccumulator,
    row: Mapping[str, str],
    *,
    view: str,
    row_number: int,
) -> Tuple[int, float]:
    accumulator.rows += 1
    for column, value in row.items():
        if not value:
            accumulator.missing[column] += 1
    if not row["date"]:
        raise PlannerEdaError("%s date is missing at data row %d" % (view, row_number))
    if not row["duration_ms"]:
        raise PlannerEdaError(
            "%s duration_ms is missing at data row %d" % (view, row_number)
        )
    date = _parse_date(row["date"], view=view, row_number=row_number)
    duration = _parse_duration(
        row["duration_ms"], view=view, row_number=row_number
    )
    accumulator.dates.append(date)
    accumulator.durations.append(duration)
    for column in CARDINALITY_COLUMNS:
        if row[column]:
            accumulator.entities[column].add(row[column])
    return date, duration


def _read_csv(path: Path, expected_columns: Sequence[str]):
    handle = path.open("r", newline="", encoding="utf-8")
    reader = csv.DictReader(handle)
    if tuple(reader.fieldnames or ()) != tuple(expected_columns):
        handle.close()
        raise PlannerEdaError(
            "%s must have exact columns %s"
            % (path.name, ",".join(expected_columns))
        )
    return handle, reader


def _rate_slice(value: str, counts: Tuple[int, int]) -> PlannerEdaRateSlice:
    rows, positives = counts
    return PlannerEdaRateSlice(
        value=value,
        row_count=rows,
        positive_count=positives,
        positive_rate=_rounded(positives / rows),
    )


def _bounded_rate_slices(
    counts: Mapping[str, Tuple[int, int]], *, chronological: bool
) -> List[PlannerEdaRateSlice]:
    if chronological:
        ordered = sorted(counts.items(), key=lambda item: item[0])
    else:
        ordered = sorted(counts.items(), key=lambda item: (-item[1][0], item[0]))
    if len(ordered) <= _MAX_RATE_SLICES:
        return [_rate_slice(value, totals) for value, totals in ordered]
    retained = ordered[: _MAX_RATE_SLICES - 1]
    remainder = ordered[_MAX_RATE_SLICES - 1 :]
    other_rows = sum(totals[0] for _, totals in remainder)
    other_positives = sum(totals[1] for _, totals in remainder)
    return [
        *[_rate_slice(value, totals) for value, totals in retained],
        _rate_slice("__other__", (other_rows, other_positives)),
    ]


def _overlap_summary(
    score_values: Set[str], train_values: Set[str]
) -> PlannerEdaOverlapSummary:
    seen = len(score_values.intersection(train_values))
    total = len(score_values)
    return PlannerEdaOverlapSummary(
        score_distinct_count=total,
        seen_in_train_count=seen,
        unseen_in_train_count=total - seen,
        seen_in_train_rate=_rounded(seen / total) if total else 0.0,
    )


class PlannerEdaToolbox:
    """Run the fixed aggregate EDA tool suite against one approved input root."""

    tool_ids = EDA_TOOL_IDS

    def __init__(self, candidate_full_input_root: Path) -> None:
        self.input_root = Path(candidate_full_input_root)
        self._cached_profile: Optional[PlannerDataProfile] = None

    def _approved_file(self, name: str) -> Path:
        if self.input_root.is_symlink():
            raise PlannerEdaError("planner EDA input root must not be a symlink")
        try:
            root = self.input_root.resolve(strict=True)
        except OSError as exc:
            raise PlannerEdaError("planner EDA input root is unavailable") from exc
        if not root.is_dir():
            raise PlannerEdaError("planner EDA input root must be a directory")
        candidate = root / name
        if candidate.is_symlink() or not candidate.is_file():
            raise PlannerEdaError("planner EDA %s must be a regular file" % name)
        resolved = candidate.resolve(strict=True)
        if resolved.parent != root:
            raise PlannerEdaError("planner EDA file escapes its approved input root")
        return resolved

    def inspect(self) -> PlannerDataProfile:
        """Execute every fixed tool once and return a cached typed profile."""

        if self._cached_profile is not None:
            return self._cached_profile

        train_path = self._approved_file("train.csv")
        score_path = self._approved_file("score.csv")
        train = _ViewAccumulator(TRAIN_COLUMNS)
        score = _ViewAccumulator(SCORE_COLUMNS)
        interactions = {column: Counter() for column in ENTITY_COLUMNS}
        tab_rates = defaultdict(lambda: [0, 0])
        date_rates = defaultdict(lambda: [0, 0])
        positive_count = 0

        train_handle, train_reader = _read_csv(train_path, TRAIN_COLUMNS)
        try:
            for row_number, raw in enumerate(train_reader, start=1):
                row = _prepare_row(
                    raw,
                    expected_columns=TRAIN_COLUMNS,
                    view="train",
                    row_number=row_number,
                )
                date, _ = _update_common(
                    train, row, view="train", row_number=row_number
                )
                try:
                    label = int(row["long_view"])
                except ValueError as exc:
                    raise PlannerEdaError(
                        "train long_view is invalid at data row %d" % row_number
                    ) from exc
                if label not in (0, 1):
                    raise PlannerEdaError(
                        "train long_view must be binary at data row %d" % row_number
                    )
                positive_count += label
                if row["tab"]:
                    tab_rates[row["tab"]][0] += 1
                    tab_rates[row["tab"]][1] += label
                date_key = str(date)
                date_rates[date_key][0] += 1
                date_rates[date_key][1] += label
                for column in ENTITY_COLUMNS:
                    if row[column]:
                        interactions[column][row[column]] += 1
        finally:
            train_handle.close()

        score_handle, score_reader = _read_csv(score_path, SCORE_COLUMNS)
        try:
            for row_number, raw in enumerate(score_reader, start=1):
                row = _prepare_row(
                    raw,
                    expected_columns=SCORE_COLUMNS,
                    view="score",
                    row_number=row_number,
                )
                try:
                    row_id = int(row["row_id"])
                except ValueError as exc:
                    raise PlannerEdaError(
                        "score row_id is invalid at data row %d" % row_number
                    ) from exc
                if row_id != row_number - 1:
                    raise PlannerEdaError(
                        "score row_id must be contiguous at data row %d" % row_number
                    )
                _update_common(score, row, view="score", row_number=row_number)
        finally:
            score_handle.close()

        if train.rows == 0 or score.rows == 0:
            raise PlannerEdaError("planner EDA views must not be empty")
        train_rates = {
            key: (int(value[0]), int(value[1])) for key, value in tab_rates.items()
        }
        temporal_rates = {
            key: (int(value[0]), int(value[1])) for key, value in date_rates.items()
        }
        payload = {
            "schema_version": "1.0",
            "source_view": "candidate_full",
            "tool_ids": list(EDA_TOOL_IDS),
            "train_file_sha256": _sha256_file(train_path),
            "score_file_sha256": _sha256_file(score_path),
            "train_columns": list(TRAIN_COLUMNS),
            "score_columns": list(SCORE_COLUMNS),
            "train_rows": train.rows,
            "score_rows": score.rows,
            "train_date_min": min(train.dates),
            "train_date_max": max(train.dates),
            "score_date_min": min(score.dates),
            "score_date_max": max(score.dates),
            "train_positive_count": positive_count,
            "train_positive_rate": _rounded(positive_count / train.rows),
            "train_cardinalities": {
                column: len(train.entities[column]) for column in CARDINALITY_COLUMNS
            },
            "score_cardinalities": {
                column: len(score.entities[column]) for column in CARDINALITY_COLUMNS
            },
            "train_missing_counts": train.missing,
            "score_missing_counts": score.missing,
            "train_duration_ms": _numeric_summary(train.durations),
            "score_duration_ms": _numeric_summary(score.durations),
            "train_interactions_per_entity": {
                column: _numeric_summary(counter.values())
                for column, counter in interactions.items()
            },
            "score_entity_overlap": {
                column: _overlap_summary(
                    score.entities[column], train.entities[column]
                )
                for column in ENTITY_COLUMNS
            },
            "train_long_view_by_tab": _bounded_rate_slices(
                train_rates, chronological=False
            ),
            "train_long_view_by_date": _bounded_rate_slices(
                temporal_rates, chronological=True
            ),
        }
        canonical = json.dumps(
            payload,
            default=lambda value: value.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._cached_profile = PlannerDataProfile(
            profile_sha256=hashlib.sha256(canonical).hexdigest(),
            **payload,
        )
        return self._cached_profile
