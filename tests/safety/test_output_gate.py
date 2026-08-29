from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pytest

from tacorank.safety import (
    ExecutionSealExpectation,
    OutputColumn,
    OutputContract,
    OutputGate,
    ViolationCode,
)

from .helpers import COMMIT_SHA, FACTORIES, Record, artifact


VALID_CSV = (
    "row_id,user_id,item_id,score\n"
    "0,u1,i1,0.1\n"
    "1,u1,i1,0.2\n"
    "2,u2,i2,0.3\n"
)
DATA_MANIFEST_SHA256 = "d" * 64
PATCH_RECEIPT_SHA256 = "e" * 64


def seal_expectation(
    *,
    producer_commit_sha: str = COMMIT_SHA,
    execution_attempt: int = 1,
) -> ExecutionSealExpectation:
    return ExecutionSealExpectation(
        run_id="run_1",
        experiment_id="exp_1",
        execution_attempt=execution_attempt,
        producer_commit_sha=producer_commit_sha,
        command_id="candidate_smoke",
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        patch_receipt_id="receipt-1",
        patch_receipt_sha256=PATCH_RECEIPT_SHA256,
    )


def make_contract() -> OutputContract:
    return OutputContract(
        columns=(
            OutputColumn("row_id", "integer"),
            OutputColumn("user_id", "string"),
            OutputColumn("item_id", "string"),
            OutputColumn("score", "number"),
        ),
        score_column="score",
        expected_rows=(
            {"row_id": 0, "user_id": "u1", "item_id": "i1"},
            {"row_id": 1, "user_id": "u1", "item_id": "i1"},
            {"row_id": 2, "user_id": "u2", "item_id": "i2"},
        ),
        identity_columns=("user_id", "item_id"),
        forbidden_columns=("protected_target",),
    )


def check_csv(
    root: Path,
    csv_text: str,
    *,
    producer_commit_sha: str = COMMIT_SHA,
    tamper_after_reference: bool = False,
    artifact_attempt: int = 1,
    max_prediction_bytes: int = 1024 * 1024,
    execution_seal_verifier: Optional[
        Callable[..., Mapping[str, Any]]
    ] = None,
):
    relative = (
        "artifacts/run_1/exp_1/attempt_{}/outputs/predictions.csv".format(
            artifact_attempt
        )
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = csv_text.encode("utf-8")
    target.write_bytes(encoded)
    artifact_ref = artifact(relative, encoded, "predictions")
    if tamper_after_reference:
        target.write_text(csv_text + "3,u3,i3,0.4\n", encoding="utf-8")
    run_result = Record(
        run_id="run_1",
        experiment_id="exp_1",
        attempt=1,
        patch_commit_sha=COMMIT_SHA,
        prediction_artifact=artifact_ref,
    )
    gate = OutputGate(
        repository_root=root,
        contract=make_contract(),
        factories=FACTORIES,
        max_prediction_bytes=max_prediction_bytes,
        execution_seal_verifier=(
            execution_seal_verifier or accepting_execution_seal
        ),
    )
    return asyncio.run(
        gate.check(
            run_result,
            expected_execution=seal_expectation(
                producer_commit_sha=producer_commit_sha
            ),
        )
    )


def accepting_execution_seal(
    repository_root: Path,
    prediction_artifact: Any,
    **identity: Any,
) -> Mapping[str, Any]:
    del repository_root, prediction_artifact
    return {"producer_commit_sha": identity["producer_commit_sha"]}


def test_gate_b_accepts_exact_official_order_and_duplicate_rows(tmp_path: Path) -> None:
    result = check_csv(tmp_path, VALID_CSV)

    assert result.accepted
    assert result.violations == []
    assert set(result.checks.values()) == {"pass"}
    assert result.score_stats == {
        "count": 3,
        "finite_count": 3,
        "unique_count": 3,
        "minimum": 0.1,
        "maximum": 0.3,
        "mean": pytest.approx(0.2),
    }


def test_gate_b_default_verifier_requires_exact_execution_sidecar(
    tmp_path: Path,
) -> None:
    relative = "artifacts/run_1/exp_1/attempt_1/outputs/predictions.csv"
    encoded = VALID_CSV.encode("utf-8")
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)
    artifact_ref = artifact(relative, encoded, "predictions")
    seal = {
        "schema": "tacorank.execution-seal.v1",
        "producer": "tacorank.execution.ExecutionRunner",
        "run_id": "run_1",
        "experiment_id": "exp_1",
        "execution_attempt": 1,
        "producer_commit_sha": COMMIT_SHA,
        "command_id": "candidate_smoke",
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "patch_receipt_id": "receipt-1",
        "patch_receipt_sha256": PATCH_RECEIPT_SHA256,
        "prediction": {
            "path": relative,
            "sha256": artifact_ref.sha256,
            "size_bytes": artifact_ref.size_bytes,
        },
    }
    (target.parent.parent / "execution-seal.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    run_result = Record(
        run_id="run_1",
        experiment_id="exp_1",
        attempt=1,
        patch_commit_sha=COMMIT_SHA,
        prediction_artifact=artifact_ref,
    )
    gate = OutputGate(
        repository_root=tmp_path,
        contract=make_contract(),
        factories=FACTORIES,
    )

    result = asyncio.run(
        gate.check(run_result, expected_execution=seal_expectation())
    )

    assert result.accepted


@pytest.mark.parametrize(
    ("csv_text", "expected_code"),
    (
        (
            "row_id,user_id,item_id,score,protected_target\n"
            "0,u1,i1,0.1,x\n1,u1,i1,0.2,x\n2,u2,i2,0.3,x\n",
            ViolationCode.OUTPUT_PROTECTED_DATA,
        ),
        (
            "row_id,user_id,item_id,score\n0,u1,i1,0.1\n2,u2,i2,0.3\n",
            ViolationCode.OUTPUT_ROW_COUNT_MISMATCH,
        ),
        (
            "row_id,user_id,item_id,score\n0,u1,i1,0.1\n2,u1,i1,0.2\n3,u2,i2,0.3\n",
            ViolationCode.OUTPUT_ROW_ID_MISMATCH,
        ),
        (
            "row_id,user_id,item_id,score\n0,u2,i2,0.1\n1,u1,i1,0.2\n2,u1,i1,0.3\n",
            ViolationCode.OUTPUT_IDENTITY_MISMATCH,
        ),
        (
            "row_id,user_id,item_id,score\n0,u1,i1,nan\n1,u1,i1,0.2\n2,u2,i2,0.3\n",
            ViolationCode.OUTPUT_NONFINITE_SCORE,
        ),
        (
            "row_id,user_id,item_id,score\n0,u1,i1,nope\n1,u1,i1,0.2\n2,u2,i2,0.3\n",
            ViolationCode.OUTPUT_TYPE_MISMATCH,
        ),
        (
            "row_id,user_id,item_id,score\n0,u1,i1,1\n1,u1,i1,1\n2,u2,i2,1\n",
            ViolationCode.OUTPUT_DEGENERATE_SCORES,
        ),
    ),
)
def test_gate_b_rejects_malformed_predictions(
    tmp_path: Path, csv_text: str, expected_code: ViolationCode
) -> None:
    result = check_csv(tmp_path, csv_text)
    assert not result.accepted
    assert expected_code.value in {violation.code for violation in result.violations}


def test_gate_b_binds_artifact_bytes_and_producer_commit(tmp_path: Path) -> None:
    artifact_result = check_csv(tmp_path, VALID_CSV, tamper_after_reference=True)
    assert ViolationCode.OUTPUT_ARTIFACT_MISMATCH.value in {
        violation.code for violation in artifact_result.violations
    }

    producer_result = check_csv(tmp_path, VALID_CSV, producer_commit_sha="b" * 40)
    assert ViolationCode.OUTPUT_PRODUCER_MISMATCH.value in {
        violation.code for violation in producer_result.violations
    }

    with pytest.raises(TypeError, match="expected_execution"):
        asyncio.run(
            OutputGate(
                repository_root=tmp_path,
                contract=make_contract(),
                factories=FACTORIES,
            ).check(Record())
        )


def test_gate_b_rejects_cross_attempt_artifact_and_missing_execution_seal(
    tmp_path: Path,
) -> None:
    substituted = check_csv(tmp_path, VALID_CSV, artifact_attempt=2)
    assert ViolationCode.OUTPUT_ARTIFACT_MISMATCH.value in {
        violation.code for violation in substituted.violations
    }

    def missing_seal(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        del args, kwargs
        raise ValueError("seal missing")

    unsealed = check_csv(
        tmp_path / "unsealed",
        VALID_CSV,
        execution_seal_verifier=missing_seal,
    )
    assert ViolationCode.OUTPUT_PRODUCER_MISMATCH.value in {
        violation.code for violation in unsealed.violations
    }


def test_gate_b_bounds_bytes_and_parses_the_verified_snapshot(tmp_path: Path) -> None:
    oversized = check_csv(
        tmp_path / "oversized",
        VALID_CSV,
        max_prediction_bytes=8,
    )
    assert ViolationCode.OUTPUT_ARTIFACT_MISMATCH.value in {
        violation.code for violation in oversized.violations
    }

    root = tmp_path / "snapshot"

    def mutate_after_verification(
        repository_root: Path,
        prediction_artifact: Any,
        **identity: Any,
    ) -> Mapping[str, Any]:
        (repository_root / prediction_artifact.path).write_text(
            "not,the,verified,csv\n",
            encoding="utf-8",
        )
        return {"producer_commit_sha": identity["producer_commit_sha"]}

    snapshot_result = check_csv(
        root,
        VALID_CSV,
        execution_seal_verifier=mutate_after_verification,
    )
    assert snapshot_result.accepted


def test_gate_b_rejects_rows_wider_than_the_exact_header(tmp_path: Path) -> None:
    malformed = (
        "row_id,user_id,item_id,score\n"
        "0,u1,i1,0.1,hidden\n"
        "1,u1,i1,0.2,hidden\n"
        "2,u2,i2,0.3,hidden\n"
    )
    result = check_csv(tmp_path, malformed)
    assert ViolationCode.OUTPUT_HEADER_MISMATCH.value in {
        violation.code for violation in result.violations
    }


def test_gate_b_detects_collapsed_duplicate_population(tmp_path: Path) -> None:
    collapsed = (
        "row_id,user_id,item_id,score\n"
        "0,u1,i1,0.1\n"
        "1,u2,i2,0.3\n"
    )
    result = check_csv(tmp_path, collapsed)
    failed_checks = {violation.check for violation in result.violations}
    assert "duplicate_preservation" in failed_checks
    assert "row_count" in failed_checks
