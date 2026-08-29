from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from solution.candidate import run as run_candidate
from tacorank import deployment as deployment_module
from tacorank.config import ContractError
from tacorank.orchestrator.live import _verify_data_manifest


def _row(index: int, label: int):
    return (
        20220408 + index,
        "user_%d" % (index % 3),
        "video_%d" % index,
        "author_%d" % index,
        "tab",
        float(1000 + index),
        label,
    )


def test_prepare_data_builds_separate_unlabelled_views_and_attested_labels(
    tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "repository").resolve()
    deployment = root / ".tacorank" / "deployment"
    data = root / "KuaiRand-Pure" / "data"
    (root / "kuairand-starter-kit").mkdir(parents=True)
    deployment.mkdir(parents=True)
    data.mkdir(parents=True)
    for name in deployment_module.RAW_REQUIRED:
        (data / name).write_text("header\n", encoding="utf-8")
    train = [_row(index, index % 2) for index in range(6)]
    valid = [_row(index + 10, index % 2) for index in range(4)]
    test = [_row(index + 20, 0) for index in range(3)]
    monkeypatch.setattr(
        deployment_module,
        "_load_official_splits",
        lambda root, data: {"train": train, "valid": valid, "test": test},
    )
    monkeypatch.setattr(
        deployment_module,
        "split_validation_indices",
        lambda users: ([0, 2], [1, 3]),
    )

    def fake_run(args, *, cwd, label, capture_output=False):
        del cwd, label, capture_output
        baseline = Path(args[2])
        with baseline.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("row_id", "user_id", "video_id", "score"))
            for row_id, row in enumerate(valid):
                writer.writerow((row_id, row[1], row[2], row_id / 10))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(deployment_module, "_run", fake_run)

    result = deployment_module._prepare_data(root, deployment, data)

    smoke_header = (result["input_roots"]["candidate_smoke"] / "score.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "long_view" not in smoke_header
    assert "label" not in smoke_header
    population_header = result["population_csvs"]["smoke"].read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert population_header == "row_id,user_id,video_id,label"
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    attested = {record["path"] for record in manifest["files"]}
    assert (
        result["input_roots"]["candidate_full"] / "score.csv"
    ).relative_to(root).as_posix() in attested
    assert result["population_csvs"]["full"].relative_to(root).as_posix() in attested

    config = SimpleNamespace(
        repository_root=root,
        data_manifest_sha256=hashlib.sha256(
            result["manifest_path"].read_bytes()
        ).hexdigest(),
    )
    live = SimpleNamespace(
        data_manifest_path=result["manifest_path"],
        input_roots=result["input_roots"],
        population_csvs=result["population_csvs"],
        baseline_prediction_csvs=result["baseline_prediction_csvs"],
    )
    _verify_data_manifest(config, live)

    with (result["input_roots"]["candidate_full"] / "score.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("tampered\n")
    with pytest.raises(ContractError, match="identity changed"):
        _verify_data_manifest(config, live)


def test_production_candidate_writes_ordered_finite_predictions(tmp_path: Path) -> None:
    input_root = (tmp_path / "inputs").resolve()
    output_root = (tmp_path / "outputs").resolve()
    input_root.mkdir()
    output_root.mkdir()
    (input_root / "train.csv").write_text(
        "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
        "20220408,u1,v1,a1,t,1000,1\n"
        "20220408,u2,v1,a1,t,1000,0\n"
        "20220408,u1,v2,a2,t,1000,0\n",
        encoding="utf-8",
    )
    (input_root / "score.csv").write_text(
        "row_id,date,user_id,video_id,author_id,tab,duration_ms\n"
        "0,20220422,u1,v1,a1,t,1000\n"
        "1,20220422,u1,unknown,a3,t,1000\n",
        encoding="utf-8",
    )
    output = output_root / "predictions.csv"

    run_candidate(SimpleNamespace(input_root=input_root, output_path=output))

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    assert [int(row["row_id"]) for row in rows] == [0, 1]
    assert [(row["user_id"], row["video_id"]) for row in rows] == [
        ("u1", "v1"),
        ("u1", "unknown"),
    ]
    assert all(float(row["score"]) == float(row["score"]) for row in rows)
