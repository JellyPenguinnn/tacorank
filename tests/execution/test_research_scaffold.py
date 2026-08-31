from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from solution.experiment_config import CONFIG
from solution.research_scaffold import (
    HISTORY_RAW_COLUMNS,
    _listwise_rows,
    _pairwise_rows,
    run_experiment,
    self_test,
    validate_config,
)


def _input_root(tmp_path: Path) -> Path:
    root = tmp_path / "input"
    root.mkdir()
    (root / "train.csv").write_text(
        "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
        "20220401,1,10,100,0,1000.0,1\n"
        "20220402,1,11,101,0,2000.0,0\n"
        "20220401,2,12,102,1,3000.0,1\n"
        "20220402,2,13,103,1,4000.0,0\n",
        encoding="utf-8",
    )
    (root / "score.csv").write_text(
        "row_id,date,user_id,video_id,author_id,tab,duration_ms\n"
        "0,20220403,1,10,100,0,1000.0\n"
        "1,20220403,1,11,101,0,2000.0\n"
        "2,20220403,2,12,102,1,3000.0\n"
        "3,20220403,2,13,103,1,4000.0\n",
        encoding="utf-8",
    )
    parent = root / "fm_baseline_predictions.csv"
    parent.write_text(
        "row_id,user_id,video_id,score\n"
        "0,1,10,0.1\n1,1,11,-0.1\n2,2,12,0.2\n3,2,13,-0.2\n",
        encoding="utf-8",
    )
    (root / "fm_baseline_predictions.sha256").write_text(
        hashlib.sha256(parent.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    return root.resolve()


def _run(tmp_path: Path, root: Path, formulation: str, suffix: str) -> Path:
    output_root = tmp_path / suffix
    output_root.mkdir()
    output = output_root / "predictions.csv"
    config = {**CONFIG, "formulation": formulation, "epochs": 1, "embedding_dim": 2}
    run_experiment(
        SimpleNamespace(
            input_root=root,
            output_path=output,
            fidelity="smoke",
            seed=17,
        ),
        config,
    )
    return output


def _write_history_inputs(root: Path) -> None:
    rows = []
    for index, (user, video, label) in enumerate(
        (("1", "10", 1), ("1", "11", 0), ("2", "12", 1), ("2", "13", 0))
    ):
        rows.append(
            {
                "row_id": str(index),
                "date": "20220401",
                "time_ms": str(1000 + index),
                "user_id": user,
                "video_id": video,
                "history_exposure": str(index),
                "global_positive_rate": "0.5",
                "tag_exposure": "4",
                "tag_positive": "3" if label else "1",
                "tag_coverage": "1",
                "tag_positive_age_days": "1",
                "author_exposure": "2",
                "author_positive": str(label),
                "author_positive_age_days": "2" if label else "-1",
                "duration_exposure": "4",
                "duration_positive": "2",
                "tab_exposure": "4",
                "tab_positive": "2",
                "hour_sin": "0",
                "hour_cos": "1",
                "weekday_sin": "0.5",
                "weekday_cos": "-0.5",
                "log_duration_scaled": "0.6",
                "item_age_scaled": "0.1",
                "duration_bucket": "le_18s",
                "long_view": str(label),
            }
        )
    train = root / "history_train.csv"
    with train.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*HISTORY_RAW_COLUMNS, "long_view"]
        )
        writer.writeheader()
        writer.writerows(rows)
    score = root / "history_score.csv"
    with score.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HISTORY_RAW_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in HISTORY_RAW_COLUMNS})
    manifest = {
        "schema_version": "1.0",
        "train_file": "history_train.csv",
        "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "score_file": "history_score.csv",
        "score_sha256": hashlib.sha256(score.read_bytes()).hexdigest(),
        "train_columns": [*HISTORY_RAW_COLUMNS, "long_view"],
        "score_columns": list(HISTORY_RAW_COLUMNS),
        "feature_columns": [],
        "history_update_policy": "emit_before_update_train_frozen_for_score",
        "static_feature_policy": "candidate_conditioned_context_only",
    }
    (root / "history-feature-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_passthrough_reproduces_frozen_parent_bytes(tmp_path: Path) -> None:
    root = _input_root(tmp_path)
    output = _run(tmp_path, root, "passthrough", "first")

    assert output.read_bytes() == (root / "fm_baseline_predictions.csv").read_bytes()
    diagnostics = json.loads(
        output.with_name("training-diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["formulation"] == "passthrough"


@pytest.mark.parametrize("formulation", ["pointwise", "bpr", "listwise", "temporal_history"])
def test_trainable_formulations_are_finite_and_deterministic(
    tmp_path: Path, formulation: str
) -> None:
    root = _input_root(tmp_path)
    first = _run(tmp_path, root, formulation, "first")
    second = _run(tmp_path, root, formulation, "second")

    assert first.read_bytes() == second.read_bytes()
    rows = list(csv.DictReader(first.open(newline="", encoding="utf-8")))
    assert [int(row["row_id"]) for row in rows] == [0, 1, 2, 3]
    assert all(float(row["score"]) == float(row["score"]) for row in rows)
    diagnostics = json.loads(
        first.with_name("training-diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["train_rows"] == 4
    assert diagnostics["gradient_norm"] >= 0


def test_history_affinity_is_hash_bound_regularized_and_deterministic(
    tmp_path: Path,
) -> None:
    root = _input_root(tmp_path)
    _write_history_inputs(root)

    first = _run(tmp_path, root, "history_affinity", "first")
    second = _run(tmp_path, root, "history_affinity", "second")

    assert first.read_bytes() == second.read_bytes()
    diagnostics = json.loads(
        first.with_name("training-diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["implementation_id"] == "features_history_affinity_v1"
    assert diagnostics["training_semantics"]["point_in_time_features"] is True
    assert diagnostics["training_semantics"]["raw_id_features"] is False
    assert diagnostics["training_semantics"]["feature_count"] == 18
    assert diagnostics["interaction_coverage"] == 0.75


def test_history_affinity_rejects_tampered_feature_file(tmp_path: Path) -> None:
    root = _input_root(tmp_path)
    _write_history_inputs(root)
    with (root / "history_score.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="history feature identity is invalid"):
        _run(tmp_path, root, "history_affinity", "first")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"history_shrinkage": 0.0}, "shrinkage"),
        ({"l2": 0.0}, "regularization"),
        ({"epochs": 6}, "5 epochs"),
        ({"residual_scale": 0.25}, "residual_scale"),
    ],
)
def test_history_affinity_enforces_overfit_bounds(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_config({**CONFIG, "formulation": "history_affinity", **override})


def test_scaffold_gradient_and_negative_sampling_checks() -> None:
    self_test()


def test_bpr_negative_count_changes_the_executed_pair_count() -> None:
    rows = [
        (("user_id=1",), "1", 1, 1.0),
        (("user_id=1",), "1", 2, 1.0),
        (("user_id=1",), "1", 3, 0.0),
        (("user_id=1",), "1", 4, 0.0),
    ]

    assert len(_pairwise_rows(rows, random.Random(7), 1)) == 2
    assert len(_pairwise_rows(rows, random.Random(7), 4)) == 8


def test_listwise_uses_one_complete_normalized_list_per_user() -> None:
    rows = [
        (("user_id=1",), "1", 1, 1.0),
        (("user_id=1",), "1", 2, 1.0),
        (("user_id=1",), "1", 3, 0.0),
        (("user_id=1",), "1", 4, 0.0),
        (("user_id=1",), "1", 5, 0.0),
    ]

    groups = _listwise_rows(rows)

    assert groups == [((0, 1, 2, 3, 4), (0.5, 0.5, 0.0, 0.0, 0.0))]
