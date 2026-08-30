from __future__ import annotations

import csv
from pathlib import Path

from tacorank.feature_materialization import materialize_history_features


LOG_HEADER = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "long_view",
    "duration_ms",
    "tab",
)


def _official(date, user, video, author, tab, duration, label):
    return (date, user, video, author, tab, float(duration), label)


def _write_data(root: Path, validation_label: int) -> tuple[list, list, list]:
    root.mkdir()
    (root / "video_features_basic_pure.csv").write_text(
        "video_id,author_id,upload_dt,tag\n"
        "v1,a1,2022-01-01,7\n"
        "v2,a2,2022-01-02,7\n"
        "v3,a3,2022-01-03,8\n",
        encoding="utf-8",
    )
    train = [
        _official(20220408, "u1", "v2", "a2", "1", 20000, 1),
        _official(20220408, "u1", "v1", "a1", "1", 10000, 0),
        _official(20220408, "u2", "v3", "a3", "0", 70000, 1),
    ]
    valid = [
        _official(20220422, "u1", "v1", "a1", "1", 10000, validation_label)
    ]
    test = [_official(20220429, "u1", "v2", "a2", "1", 20000, 0)]
    with (root / "log_standard_4_08_to_4_21_pure.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_HEADER)
        # Source order is intentionally not chronological.
        writer.writerow(("u1", "v2", 20220408, 1300, 2000, 1, 20000, "1"))
        writer.writerow(("u1", "v1", 20220408, 1200, 1000, 0, 10000, "1"))
        writer.writerow(("u2", "v3", 20220408, 1400, 3000, 1, 70000, "0"))
    with (root / "log_standard_4_22_to_5_08_pure.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_HEADER)
        writer.writerow(
            ("u1", "v1", 20220422, 1200, 4000, validation_label, 10000, "1")
        )
        writer.writerow(("u1", "v2", 20220429, 1200, 5000, 0, 20000, "1"))
    return train, valid, test


def _materialize(tmp_path: Path, name: str, validation_label: int):
    data = tmp_path / (name + "-data")
    train, valid, test = _write_data(data, validation_label)
    return materialize_history_features(
        data_directory=data,
        official_train=train,
        official_valid=valid,
        official_test=test,
        output_directory=tmp_path / (name + "-features"),
    )


def test_history_features_are_chronological_and_emit_before_update(tmp_path: Path) -> None:
    result = _materialize(tmp_path, "first", 0)

    with result["files"]["train"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, strict=True))

    assert [row["video_id"] for row in rows] == ["v1", "v2", "v3"]
    assert rows[0]["history_exposure"] == "0"
    assert rows[0]["tag_exposure"] == "0"
    assert rows[1]["history_exposure"] == "1"
    assert rows[1]["tag_exposure"] == "1"
    assert rows[1]["tag_positive"] == "0"
    assert result["history_update_policy"] == (
        "emit_before_update_train_frozen_for_score"
    )


def test_score_features_ignore_validation_outcomes_and_are_deterministic(
    tmp_path: Path,
) -> None:
    first = _materialize(tmp_path, "first", 0)
    second = _materialize(tmp_path, "second", 1)
    third = _materialize(tmp_path, "third", 0)

    assert first["files"]["valid"].read_bytes() == second["files"]["valid"].read_bytes()
    assert first["files"]["test"].read_bytes() == second["files"]["test"].read_bytes()
    assert first["files"]["train"].read_bytes() == third["files"]["train"].read_bytes()
    assert first["files"]["valid"].read_bytes() == third["files"]["valid"].read_bytes()
