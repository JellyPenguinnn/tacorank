from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from solution.features import FeatureEncoder, read_scoring_rows, read_training_rows
from solution.inference import (
    add_bounded_residual,
    load_verified_parent,
    write_predictions_exclusive,
)
from solution.model import FactorizationMachine


def _write_views(root: Path) -> tuple[Path, Path]:
    train = root / "train.csv"
    score = root / "score.csv"
    train.write_text(
        "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
        "20220408,u1,v1,a1,t1,1000,1\n"
        "20220409,u1,v2,a2,t1,2000,0\n"
        "20220410,u2,v1,a1,t2,3000,1\n",
        encoding="utf-8",
    )
    score.write_text(
        "row_id,date,user_id,video_id,author_id,tab,duration_ms\n"
        "0,20220422,u1,v1,a1,t1,1000\n"
        "1,20220422,new,v3,a3,t2,4000\n",
        encoding="utf-8",
    )
    return train, score


def test_candidate_scaffold_is_deterministic_and_keeps_unknown_slots(
    tmp_path: Path,
) -> None:
    train_path, score_path = _write_views(tmp_path)
    training_rows = read_training_rows(train_path)
    scoring_rows = read_scoring_rows(score_path)
    encoder = FeatureEncoder.fit(training_rows)
    train_features = encoder.transform(training_rows)
    score_features = encoder.transform(scoring_rows)
    labels = np.asarray([row.long_view for row in training_rows], dtype=np.float32)

    first = FactorizationMachine(encoder.dimension, seed=17)
    second = FactorizationMachine(encoder.dimension, seed=17)
    first.pointwise_step(train_features, labels)
    second.pointwise_step(train_features, labels)

    assert np.array_equal(first.predict(score_features), second.predict(score_features))
    assert score_features[1, 0] == encoder.unknown_ids[0] + encoder.offsets[0]


def test_inference_helpers_authenticate_parent_and_bound_only_residual(
    tmp_path: Path,
) -> None:
    _, score_path = _write_views(tmp_path)
    rows = read_scoring_rows(score_path)
    parent_document = (
        "row_id,user_id,video_id,score\n"
        "0,u1,v1,-2.5\n"
        "1,new,v3,3.5\n"
    )
    parent_path = tmp_path / "fm_baseline_predictions.csv"
    parent_path.write_text(parent_document, encoding="utf-8")
    (tmp_path / "fm_baseline_predictions.sha256").write_text(
        hashlib.sha256(parent_document.encode("utf-8")).hexdigest(),
        encoding="ascii",
    )

    parent = load_verified_parent(tmp_path, rows)
    combined = add_bounded_residual(
        parent,
        np.asarray([100.0, -100.0]),
        maximum_absolute_residual=0.2,
    )

    assert np.all(np.abs(combined - parent) <= 0.2 + 1e-12)
    assert combined[0] < 0.0
    assert combined[1] > 1.0
    output = tmp_path / "prediction.csv"
    write_predictions_exclusive(output, rows, combined)
    with output.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle, strict=True))
    assert [row["row_id"] for row in written] == ["0", "1"]
