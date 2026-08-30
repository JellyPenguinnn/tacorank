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
    _listwise_rows,
    _pairwise_rows,
    run_experiment,
    self_test,
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
