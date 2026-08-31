"""Leakage-safe point-in-time features for the KuaiRand ranking benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


FEATURE_SCHEMA_VERSION = "1.0"
HISTORY_FEATURE_COLUMNS = (
    "history_exposure",
    "global_positive_rate",
    "tag_exposure",
    "tag_positive",
    "tag_coverage",
    "tag_positive_age_days",
    "author_exposure",
    "author_positive",
    "author_positive_age_days",
    "duration_exposure",
    "duration_positive",
    "tab_exposure",
    "tab_positive",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "log_duration_scaled",
    "item_age_scaled",
    "duration_bucket",
)
TRAIN_IDENTITY_COLUMNS = ("row_id", "date", "time_ms", "user_id", "video_id")
SCORE_IDENTITY_COLUMNS = TRAIN_IDENTITY_COLUMNS
TRAIN_FEATURE_COLUMNS = TRAIN_IDENTITY_COLUMNS + HISTORY_FEATURE_COLUMNS + (
    "long_view",
)
SCORE_FEATURE_COLUMNS = SCORE_IDENTITY_COLUMNS + HISTORY_FEATURE_COLUMNS
SECONDS_PER_DAY_MS = 86_400_000.0


class FeatureMaterializationError(RuntimeError):
    """The reviewed raw data cannot produce a trustworthy feature view."""


@dataclass
class _HistoryCount:
    exposures: int = 0
    positives: int = 0
    last_positive_time_ms: Optional[int] = None


@dataclass
class _UserHistory:
    exposures: int = 0
    tags: Dict[str, _HistoryCount] = field(default_factory=dict)
    authors: Dict[str, _HistoryCount] = field(default_factory=dict)
    durations: Dict[str, _HistoryCount] = field(default_factory=dict)
    tabs: Dict[str, _HistoryCount] = field(default_factory=dict)


@dataclass(frozen=True)
class _VideoMetadata:
    author_id: str
    tags: Tuple[str, ...]
    upload_date: Optional[datetime]


@dataclass(frozen=True)
class _Event:
    source_index: int
    date: int
    time_ms: int
    hourmin: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float
    label: int


def materialize_history_features(
    *,
    data_directory: Path,
    official_train: Sequence[Sequence[Any]],
    official_valid: Sequence[Sequence[Any]],
    official_test: Sequence[Sequence[Any]],
    output_directory: Path,
) -> Mapping[str, Any]:
    """Create immutable train/valid/test views from strictly earlier outcomes."""

    data = Path(data_directory).resolve(strict=True)
    output = Path(output_directory)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    metadata = _load_video_metadata(data / "video_features_basic_pure.csv")
    train_path = output / "history-train.csv"
    valid_path = output / "history-valid.csv"
    test_path = output / "history-test.csv"
    histories: Dict[str, _UserHistory] = {}
    global_exposures = 0
    global_positives = 0
    cutoff_time_ms = 0
    cutoff_date = 0

    with tempfile.TemporaryDirectory(prefix="tacorank-feature-sort-") as temporary:
        database = Path(temporary) / "events.sqlite3"
        with sqlite3.connect(str(database)) as connection:
            _configure_database(connection)
            _create_event_table(connection)
            _insert_train_events(
                connection,
                data / "log_standard_4_08_to_4_21_pure.csv",
                metadata,
                official_train,
            )
            with train_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(TRAIN_FEATURE_COLUMNS)
                query = connection.execute(
                    "SELECT source_index,date,time_ms,hourmin,user_id,video_id,"
                    "author_id,tab,duration_ms,label FROM events "
                    "ORDER BY time_ms,source_index"
                )
                for row_id, values in enumerate(query):
                    event = _Event(*values)
                    feature_values = _feature_values(
                        event,
                        metadata[event.video_id],
                        histories.get(event.user_id),
                        global_exposures,
                        global_positives,
                    )
                    writer.writerow(
                        (
                            row_id,
                            event.date,
                            event.time_ms,
                            event.user_id,
                            event.video_id,
                            *feature_values,
                            event.label,
                        )
                    )
                    _update_history(
                        histories,
                        event,
                        metadata[event.video_id],
                    )
                    global_exposures += 1
                    global_positives += event.label
                    cutoff_time_ms = max(cutoff_time_ms, event.time_ms)
                    cutoff_date = max(cutoff_date, event.date)

    if global_exposures != len(official_train):
        raise FeatureMaterializationError("training feature row count is inconsistent")
    _write_score_features(
        source=data / "log_standard_4_22_to_5_08_pure.csv",
        metadata=metadata,
        histories=histories,
        global_exposures=global_exposures,
        global_positives=global_positives,
        cutoff_date=cutoff_date,
        official_valid=official_valid,
        official_test=official_test,
        valid_path=valid_path,
        test_path=test_path,
    )
    files = {"train": train_path, "valid": valid_path, "test": test_path}
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "files": files,
        "sha256": {name: _sha256_file(path) for name, path in files.items()},
        "row_counts": {
            "train": len(official_train),
            "valid": len(official_valid),
            "test": len(official_test),
        },
        "train_columns": list(TRAIN_FEATURE_COLUMNS),
        "score_columns": list(SCORE_FEATURE_COLUMNS),
        "feature_columns": list(HISTORY_FEATURE_COLUMNS),
        "cutoff_time_ms": cutoff_time_ms,
        "cutoff_date": cutoff_date,
        "history_update_policy": "emit_before_update_train_frozen_for_score",
        "static_feature_policy": "candidate_conditioned_context_only",
    }


def _configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")


def _create_event_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE events ("
        "source_index INTEGER NOT NULL,date INTEGER NOT NULL,time_ms INTEGER NOT NULL,"
        "hourmin INTEGER NOT NULL,user_id TEXT NOT NULL,video_id TEXT NOT NULL,"
        "author_id TEXT NOT NULL,tab TEXT NOT NULL,duration_ms REAL NOT NULL,"
        "label INTEGER NOT NULL)"
    )


def _insert_train_events(
    connection: sqlite3.Connection,
    path: Path,
    metadata: Mapping[str, _VideoMetadata],
    official_rows: Sequence[Sequence[Any]],
) -> None:
    batch = []
    official_index = 0
    for source_index, raw in enumerate(_read_csv(path)):
        date = _integer(raw, "date")
        if not 20220408 <= date <= 20220421:
            continue
        event = _parse_event(source_index, raw, metadata)
        if official_index >= len(official_rows):
            raise FeatureMaterializationError("raw training log has extra split rows")
        _validate_official_row(event, official_rows[official_index])
        official_index += 1
        batch.append(
            (
                event.source_index,
                event.date,
                event.time_ms,
                event.hourmin,
                event.user_id,
                event.video_id,
                event.author_id,
                event.tab,
                event.duration_ms,
                event.label,
            )
        )
        if len(batch) >= 10_000:
            connection.executemany(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", batch
            )
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
    if official_index != len(official_rows):
        raise FeatureMaterializationError("raw training log is missing split rows")
    connection.commit()


def _write_score_features(
    *,
    source: Path,
    metadata: Mapping[str, _VideoMetadata],
    histories: Mapping[str, _UserHistory],
    global_exposures: int,
    global_positives: int,
    cutoff_date: int,
    official_valid: Sequence[Sequence[Any]],
    official_test: Sequence[Sequence[Any]],
    valid_path: Path,
    test_path: Path,
) -> None:
    valid_index = 0
    test_index = 0
    with valid_path.open("x", newline="", encoding="utf-8") as valid_handle, test_path.open(
        "x", newline="", encoding="utf-8"
    ) as test_handle:
        writers = {
            "valid": csv.writer(valid_handle, lineterminator="\n"),
            "test": csv.writer(test_handle, lineterminator="\n"),
        }
        writers["valid"].writerow(SCORE_FEATURE_COLUMNS)
        writers["test"].writerow(SCORE_FEATURE_COLUMNS)
        for source_index, raw in enumerate(_read_csv(source)):
            date = _integer(raw, "date")
            if 20220422 <= date <= 20220428:
                split = "valid"
                row_id = valid_index
                expected = official_valid
                valid_index += 1
            elif 20220429 <= date <= 20220508:
                split = "test"
                row_id = test_index
                expected = official_test
                test_index += 1
            else:
                continue
            event = _parse_event(source_index, raw, metadata)
            if event.date <= cutoff_date:
                raise FeatureMaterializationError(
                    "score interaction does not follow the training cutoff"
                )
            if row_id >= len(expected):
                raise FeatureMaterializationError("raw score log has extra split rows")
            _validate_official_row(event, expected[row_id])
            values = _feature_values(
                event,
                metadata[event.video_id],
                histories.get(event.user_id),
                global_exposures,
                global_positives,
            )
            writers[split].writerow(
                (
                    row_id,
                    event.date,
                    event.time_ms,
                    event.user_id,
                    event.video_id,
                    *values,
                )
            )
    if valid_index != len(official_valid) or test_index != len(official_test):
        raise FeatureMaterializationError("raw score log is missing split rows")


def _feature_values(
    event: _Event,
    metadata: _VideoMetadata,
    history: Optional[_UserHistory],
    global_exposures: int,
    global_positives: int,
) -> Tuple[Any, ...]:
    global_rate = global_positives / global_exposures if global_exposures else 0.5
    tags = _summarize_keys(
        history.tags if history is not None else {},
        metadata.tags,
        event.time_ms,
    )
    author = _summarize_keys(
        history.authors if history is not None else {},
        (event.author_id,),
        event.time_ms,
    )
    duration_bucket = _duration_bucket(event.duration_ms)
    duration = _summarize_keys(
        history.durations if history is not None else {},
        (duration_bucket,),
        event.time_ms,
    )
    tab = _summarize_keys(
        history.tabs if history is not None else {},
        (event.tab,),
        event.time_ms,
    )
    hour = max(0.0, min(23.999, event.hourmin // 100 + event.hourmin % 100 / 60.0))
    hour_angle = 2.0 * math.pi * hour / 24.0
    event_date = datetime.strptime(str(event.date), "%Y%m%d")
    weekday_angle = 2.0 * math.pi * event_date.weekday() / 7.0
    duration_scale = min(1.0, math.log1p(event.duration_ms) / math.log1p(600_000.0))
    item_age = 0.0
    if metadata.upload_date is not None:
        item_age = max(0.0, (event_date - metadata.upload_date).days)
    item_age_scale = min(1.0, item_age / 3650.0)
    return (
        history.exposures if history is not None else 0,
        _number(global_rate),
        _number(tags[0]),
        _number(tags[1]),
        _number(tags[2]),
        _number(tags[3]),
        _number(author[0]),
        _number(author[1]),
        _number(author[3]),
        _number(duration[0]),
        _number(duration[1]),
        _number(tab[0]),
        _number(tab[1]),
        _number(math.sin(hour_angle)),
        _number(math.cos(hour_angle)),
        _number(math.sin(weekday_angle)),
        _number(math.cos(weekday_angle)),
        _number(duration_scale),
        _number(item_age_scale),
        duration_bucket,
    )


def _summarize_keys(
    values: Mapping[str, _HistoryCount],
    keys: Iterable[str],
    time_ms: int,
) -> Tuple[float, float, float, float]:
    distinct = tuple(dict.fromkeys(key for key in keys if key))
    if not distinct:
        return 0.0, 0.0, 0.0, -1.0
    matched = [values[key] for key in distinct if key in values]
    exposure = sum(value.exposures for value in matched) / len(distinct)
    positive = sum(value.positives for value in matched) / len(distinct)
    coverage = len(matched) / len(distinct)
    positive_times = [
        value.last_positive_time_ms
        for value in matched
        if value.last_positive_time_ms is not None
    ]
    age = (
        max(0.0, (time_ms - max(positive_times)) / SECONDS_PER_DAY_MS)
        if positive_times
        else -1.0
    )
    return exposure, positive, coverage, age


def _update_history(
    histories: Dict[str, _UserHistory],
    event: _Event,
    metadata: _VideoMetadata,
) -> None:
    history = histories.setdefault(event.user_id, _UserHistory())
    history.exposures += 1
    groups = (
        (history.tags, metadata.tags),
        (history.authors, (event.author_id,)),
        (history.durations, (_duration_bucket(event.duration_ms),)),
        (history.tabs, (event.tab,)),
    )
    for values, keys in groups:
        for key in dict.fromkeys(key for key in keys if key):
            count = values.setdefault(key, _HistoryCount())
            count.exposures += 1
            count.positives += event.label
            if event.label:
                count.last_positive_time_ms = event.time_ms


def _load_video_metadata(path: Path) -> Mapping[str, _VideoMetadata]:
    metadata: Dict[str, _VideoMetadata] = {}
    for row in _read_csv(path):
        video_id = _required(row, "video_id")
        if video_id in metadata:
            raise FeatureMaterializationError("video metadata contains duplicate ids")
        tags = tuple(
            dict.fromkeys(
                value.strip()
                for value in row.get("tag", "").split(",")
                if value.strip() and value.strip().lower() != "nan"
            )
        )
        upload_date = None
        raw_upload = row.get("upload_dt", "").strip()
        if raw_upload and raw_upload.lower() != "nan":
            try:
                upload_date = datetime.strptime(raw_upload, "%Y-%m-%d")
            except ValueError as error:
                raise FeatureMaterializationError("invalid video upload date") from error
        metadata[video_id] = _VideoMetadata(
            author_id=_required(row, "author_id"),
            tags=tags,
            upload_date=upload_date,
        )
    if not metadata:
        raise FeatureMaterializationError("video metadata is empty")
    return metadata


def _parse_event(
    source_index: int,
    row: Mapping[str, str],
    metadata: Mapping[str, _VideoMetadata],
) -> _Event:
    video_id = _required(row, "video_id")
    if video_id not in metadata:
        raise FeatureMaterializationError("interaction references unknown video metadata")
    duration = _floating(row, "duration_ms")
    if duration < 0:
        raise FeatureMaterializationError("duration_ms must be non-negative")
    label = _integer(row, "long_view")
    if label not in (0, 1):
        raise FeatureMaterializationError("long_view must be binary")
    return _Event(
        source_index=source_index,
        date=_integer(row, "date"),
        time_ms=_integer(row, "time_ms"),
        hourmin=_integer(row, "hourmin"),
        user_id=_required(row, "user_id"),
        video_id=video_id,
        author_id=metadata[video_id].author_id,
        tab=_required(row, "tab"),
        duration_ms=duration,
        label=label,
    )


def _validate_official_row(event: _Event, row: Sequence[Any]) -> None:
    expected = (
        int(row[0]),
        str(row[1]),
        str(row[2]),
        str(row[3]),
        str(row[4]),
        float(row[5]),
        int(row[6]),
    )
    actual = (
        event.date,
        event.user_id,
        event.video_id,
        event.author_id,
        event.tab,
        event.duration_ms,
        event.label,
    )
    if actual != expected:
        raise FeatureMaterializationError("raw feature row does not align with official split")


def _duration_bucket(duration_ms: float) -> str:
    if duration_ms <= 7_000:
        return "le_7s"
    if duration_ms <= 18_000:
        return "le_18s"
    if duration_ms <= 60_000:
        return "le_60s"
    return "gt_60s"


def _read_csv(path: Path) -> Iterable[Mapping[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise FeatureMaterializationError("required raw feature file is missing")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not reader.fieldnames:
            raise FeatureMaterializationError("raw feature file has no header")
        for row in reader:
            yield row


def _required(row: Mapping[str, str], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise FeatureMaterializationError("required raw feature value is empty: %s" % key)
    return value


def _integer(row: Mapping[str, str], key: str) -> int:
    try:
        value = float(_required(row, key))
    except ValueError as error:
        raise FeatureMaterializationError("raw feature value is not numeric: %s" % key) from error
    if not math.isfinite(value) or not value.is_integer():
        raise FeatureMaterializationError("raw feature value is not an integer: %s" % key)
    return int(value)


def _floating(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(_required(row, key))
    except ValueError as error:
        raise FeatureMaterializationError("raw feature value is not numeric: %s" % key) from error
    if not math.isfinite(value):
        raise FeatureMaterializationError("raw feature value is not finite: %s" % key)
    return value


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise FeatureMaterializationError("materialized feature is not finite")
    return "%.12g" % value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
