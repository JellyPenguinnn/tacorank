from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from benchmarks.kuairand_pure.pipeline import check_submission, run_baseline


PREDICTIONS = (
    "row_id,user_id,video_id,score\n"
    "0,u1,v1,0.1\n"
    "1,u1,v1,0.9\n"
)


def test_frozen_baseline_copy_and_submission_check(tmp_path: Path) -> None:
    input_root = (tmp_path / "input").resolve()
    contract_root = (tmp_path / "contract").resolve()
    output_root = (tmp_path / "output").resolve()
    input_root.mkdir()
    contract_root.mkdir()
    output_root.mkdir()
    source = input_root / "baseline_predictions.csv"
    source.write_text(PREDICTIONS, encoding="utf-8")
    (input_root / "baseline_predictions.sha256").write_text(
        hashlib.sha256(source.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    output = output_root / "predictions.csv"
    run_baseline(
        SimpleNamespace(
            mode="baseline",
            fidelity="full",
            input_root=input_root,
            output_path=output,
        )
    )
    assert output.read_text(encoding="utf-8") == PREDICTIONS

    (contract_root / "submission_rows.csv").write_text(
        "row_id,user_id,video_id\n0,u1,v1\n1,u1,v1\n",
        encoding="utf-8",
    )
    check_submission(
        SimpleNamespace(
            prediction_path=output,
            contract_root=contract_root,
            artifact_root=output_root,
        )
    )
