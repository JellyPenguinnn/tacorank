from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from tacorank.execution.commands import CommandProfile, CommandRegistry, ExpectedArtifact
from tacorank.execution.artifacts import verify_execution_seal
from tacorank.execution.runner import InvalidRunRequest

from .conftest import (
    ContinuingObserver,
    build_runner,
    command_registry,
    request,
)


PREDICTION = ExpectedArtifact(
    role="prediction",
    relative_path="predictions.csv",
    kind="predictions",
    content_type="text/csv",
)


def test_successful_run_is_sealed_observed_and_hash_addressed(
    execution_layout: SimpleNamespace,
) -> None:
    code = (
        "from pathlib import Path; import sys, time; "
        "print('loss=0.25', flush=True); "
        "time.sleep(0.08); "
        "Path(sys.argv[1]).write_text('row_id,score\\n0,0.5\\n')"
    )
    registry = command_registry(
        code,
        expected=(PREDICTION,),
        extra_arguments=("{prediction_path}",),
    )
    runner, seal = build_runner(execution_layout, registry)
    observer = ContinuingObserver()

    result = runner.run_sync(request(), observer)

    assert result.outcome == "success"
    assert result.exit_code == 0
    assert result.error_class is None
    assert seal.calls == 2
    assert observer.samples
    assert any(sample.loss == 0.25 for sample in observer.samples)
    prediction_path = execution_layout.store.verify(result.prediction_artifact)
    assert prediction_path.read_text().startswith("row_id,score")
    execution_layout.store.verify(result.log_artifact)
    telemetry_path = execution_layout.store.verify(result.telemetry_artifact)
    samples = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert samples
    assert all(sample["run_id"] == "run_001" for sample in samples)
    assert result.resource_delta.wall_time_ms >= 0
    seal_payload = verify_execution_seal(
        execution_layout.repository,
        result.prediction_artifact,
        run_id="run_001",
        experiment_id="exp_0001",
        execution_attempt=1,
        producer_commit_sha="a" * 40,
        command_id="candidate_smoke",
        data_manifest_sha256="b" * 64,
        patch_receipt_id="receipt_001",
        patch_receipt_sha256="c" * 64,
    )
    assert seal_payload["prediction"]["sha256"] == result.prediction_artifact.sha256
    assert seal_payload["patch_receipt_sha256"] == "c" * 64


def test_nonzero_candidate_failure_returns_code_error(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("raise ValueError('candidate broke')")
    runner, _ = build_runner(execution_layout, registry)

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "code_error"
    assert result.error_class == "CANDIDATE_CODE_ERROR"
    assert len(result.error_fingerprint) == 64
    assert "ValueError" in execution_layout.store.verify(result.log_artifact).read_text()


def test_failed_process_prediction_never_receives_trusted_execution_seal(
    execution_layout: SimpleNamespace,
) -> None:
    code = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).write_text('row_id,score\\n0,0.5\\n'); sys.exit(2)"
    )
    runner, _ = build_runner(
        execution_layout,
        command_registry(
            code,
            expected=(PREDICTION,),
            extra_arguments=("{prediction_path}",),
        ),
    )

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "code_error"
    attempt = execution_layout.artifact_root / "run_001" / "exp_0001" / "attempt_1"
    assert not (attempt / "execution-seal.json").exists()


def test_zero_exit_without_required_output_is_interface_error(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("print('finished')", expected=(PREDICTION,))
    runner, _ = build_runner(execution_layout, registry)

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "interface_error"
    assert result.error_class == "MISSING_EXPECTED_ARTIFACT"
    assert result.prediction_artifact is None


def test_unknown_command_is_contract_error_without_launch(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("print('must not run')")
    runner, seal = build_runner(execution_layout, registry)

    result = runner.run_sync(
        request(command_id="baseline_full"), ContinuingObserver()
    )

    assert result.outcome == "contract_error"
    assert result.exit_code is None
    assert "UNAPPROVED_COMMAND" in result.error_summary
    assert seal.calls == 1
    assert execution_layout.store.verify(result.telemetry_artifact).read_text() == ""


def test_unapproved_network_is_contract_error(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("print('must not run')")
    runner, _ = build_runner(execution_layout, registry)

    result = runner.run_sync(
        request(network_enabled=True), ContinuingObserver()
    )

    assert result.outcome == "contract_error"
    assert "UNAPPROVED_NETWORK" in result.error_summary


def test_submission_check_requires_and_uses_verified_prior_prediction(
    execution_layout: SimpleNamespace,
) -> None:
    prior = execution_layout.store.write_text(
        "run_001/exp_0001/attempt_1/outputs/predictions.csv",
        "row_id,score\n0,0.5\n",
        kind="predictions",
    )
    registry = CommandRegistry(
        (
            CommandProfile(
                command_id="submission_check",
                executable=str(Path(sys.executable).resolve()),
                arguments=(
                    "-c",
                    "from pathlib import Path; import sys; assert Path(sys.argv[1]).read_text().startswith('row_id')",
                    "{submission_prediction_path}",
                ),
                allowed_fidelities=("final",),
            ),
        )
    )
    missing, _ = build_runner(execution_layout, registry)
    rejected = missing.run_sync(
        request(command_id="submission_check", attempt=2, fidelity="final"),
        ContinuingObserver(),
    )
    assert rejected.outcome == "contract_error"
    assert "prior prediction resolver" in rejected.error_summary

    class Resolver:
        @staticmethod
        def resolve(request_value: Any) -> Any:
            assert request_value.attempt == 3
            return prior

    runner, _ = build_runner(
        execution_layout,
        registry,
        submission_artifact_resolver=Resolver(),
    )
    result = runner.run_sync(
        request(command_id="submission_check", attempt=3, fidelity="final"),
        ContinuingObserver(),
    )
    assert result.outcome == "success"


def test_runner_holds_execution_lease_across_launch_and_both_seal_checks(
    execution_layout: SimpleNamespace,
) -> None:
    active = False

    class LeaseVerifier:
        calls = 0

        @contextmanager
        def acquire_lease(
            self,
            request_value: Any,
            workspace: Path,
            *,
            timeout_seconds: float,
        ) -> Iterator[None]:
            nonlocal active
            del request_value, workspace
            assert timeout_seconds > 0
            assert active is False
            active = True
            try:
                yield
            finally:
                active = False

        def verify(self, request_value: Any, workspace: Path) -> SimpleNamespace:
            del request_value, workspace
            assert active is True
            self.calls += 1
            return SimpleNamespace(receipt_sha256="c" * 64)

    class LeaseCheckingLauncher:
        def __init__(self) -> None:
            from tacorank.execution.process import ProcessLauncher

            self.delegate = ProcessLauncher()

        def launch(self, specification: Any, log_path: Path, limits: Any) -> Any:
            assert active is True
            return self.delegate.launch(specification, log_path, limits)

    verifier = LeaseVerifier()
    runner, _ = build_runner(
        execution_layout,
        command_registry("print('ok')"),
        seal_verifier=verifier,  # type: ignore[arg-type]
        process_launcher=LeaseCheckingLauncher(),
    )
    result = runner.run_sync(request(), ContinuingObserver())
    assert result.outcome == "success"
    assert verifier.calls == 2
    assert active is False


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_workspace_resolver_symlink_is_rejected_before_seal_verification(
    execution_layout: SimpleNamespace,
) -> None:
    runner, seal = build_runner(
        execution_layout,
        command_registry("print('must not run')"),
    )
    linked_workspace = execution_layout.repository / "linked-workspace"
    linked_workspace.symlink_to(execution_layout.workspace, target_is_directory=True)
    runner.workspace_resolver = lambda run_id, experiment_id: linked_workspace

    with pytest.raises(InvalidRunRequest, match="symbolic link"):
        runner.run_sync(request(), ContinuingObserver())
    assert seal.calls == 0


def test_runtime_output_is_redacted_before_artifact_capture(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("print('API_KEY=sk-abcdefghijklmnop')")
    runner, _ = build_runner(execution_layout, registry)

    result = runner.run_sync(request(), ContinuingObserver())

    log = execution_layout.store.verify(result.log_artifact).read_text()
    assert "abcdefghijklmnop" not in log
    assert "[REDACTED]" in log


def test_same_attempt_artifacts_cannot_be_overwritten(
    execution_layout: SimpleNamespace,
) -> None:
    registry = command_registry("print('first')")
    runner, _ = build_runner(execution_layout, registry)
    assert runner.run_sync(request(), ContinuingObserver()).outcome == "success"

    with pytest.raises(Exception, match="immutable|already exists"):
        runner.run_sync(request(), ContinuingObserver())


def test_candidate_precreated_execution_seal_is_rejected(
    execution_layout: SimpleNamespace,
) -> None:
    code = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).write_text('row_id,score\\n0,0.5\\n'); "
        "Path(sys.argv[1]).parent.parent.joinpath('execution-seal.json').write_text('fake')"
    )
    runner, _ = build_runner(
        execution_layout,
        command_registry(
            code,
            expected=(PREDICTION,),
            extra_arguments=("{prediction_path}",),
        ),
    )

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "infrastructure_error"
    assert result.error_class == "EXECUTION_SEAL_WRITE_FAILURE"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_candidate_precreated_execution_seal_symlink_is_rejected(
    execution_layout: SimpleNamespace,
) -> None:
    victim = execution_layout.repository / "victim.txt"
    victim.write_text("untouched\n", encoding="utf-8")
    code = (
        "from pathlib import Path; import sys; "
        "prediction=Path(sys.argv[1]); prediction.write_text('row_id,score\\n0,0.5\\n'); "
        "prediction.parent.parent.joinpath('execution-seal.json').symlink_to(Path(sys.argv[2]))"
    )
    runner, _ = build_runner(
        execution_layout,
        command_registry(
            code,
            expected=(PREDICTION,),
            extra_arguments=("{prediction_path}", str(victim)),
        ),
    )

    result = runner.run_sync(request(), ContinuingObserver())

    assert result.outcome == "infrastructure_error"
    assert result.error_class == "EXECUTION_SEAL_WRITE_FAILURE"
    assert victim.read_text(encoding="utf-8") == "untouched\n"
