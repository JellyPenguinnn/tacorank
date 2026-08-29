from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import pytest

from tacorank.execution.commands import (
    CommandContext,
    CommandPolicyError,
    CommandProfile,
    CommandRegistry,
    ExpectedArtifact,
    PipelineCommandInputs,
    REQUIRED_COMMAND_IDS,
    default_command_registry,
)
from tacorank.execution.solution_cli import (
    ControllerCLIError,
    PipelineInvocation,
    SubmissionCheckInvocation,
    _load_entrypoint,
    execute_pipeline,
    execute_submission_check,
)


PIPELINE_COMMAND_IDS = REQUIRED_COMMAND_IDS.difference({"submission_check"})


def context(
    tmp_path: Path,
    *,
    attempt: int = 1,
    fidelity: str = "smoke",
    submission_prediction_path: Path | None = None,
) -> CommandContext:
    repository = tmp_path / "repo"
    worktree = repository / "worktree"
    artifact_dir = repository / "artifacts"
    worktree.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return CommandContext(
        repository_root=repository.resolve(),
        worktree=worktree.resolve(),
        artifact_dir=artifact_dir.resolve(),
        run_id="run_001",
        experiment_id="exp_0001",
        attempt=attempt,
        fidelity=fidelity,
        seed=3,
        submission_prediction_path=submission_prediction_path,
    )


def pipeline_inputs(tmp_path: Path) -> PipelineCommandInputs:
    contract_root = tmp_path / "contracts"
    contract_root.mkdir(parents=True)
    input_roots = {}
    for command_id in sorted(PIPELINE_COMMAND_IDS):
        root = tmp_path / "input_views" / command_id
        root.mkdir(parents=True)
        input_roots[command_id] = root.resolve()
    return PipelineCommandInputs(
        contract_root=contract_root.resolve(),
        input_roots=input_roots,
        baseline_entrypoint="approved.baseline:run",
        candidate_entrypoint="solution.candidate:run",
        submission_check_entrypoint="approved.submission:check",
    )


def test_default_registry_has_only_symbolic_reviewed_surface(tmp_path: Path) -> None:
    registry = default_command_registry(
        pipeline_inputs(tmp_path / "configuration"),
        container_python_executable="/usr/local/bin/python3"
    )
    assert registry.command_ids == REQUIRED_COMMAND_IDS

    prediction = tmp_path / "prior-prediction.csv"
    prediction.write_text("row_id,score\n", encoding="utf-8")
    resolved = registry.resolve(
        "submission_check",
        context(
            tmp_path,
            fidelity="full",
            submission_prediction_path=prediction.resolve(),
        ),
        network_enabled=False,
    )
    assert resolved.argv[0] == str(Path(sys.executable).resolve())
    assert resolved.command_id == "submission_check"
    assert resolved.container_executable == "/usr/local/bin/python3"


def test_registry_rejects_raw_commands_network_and_zero_attempt(tmp_path: Path) -> None:
    registry = default_command_registry(pipeline_inputs(tmp_path / "configuration"))
    with pytest.raises(CommandPolicyError, match="UNAPPROVED_COMMAND"):
        registry.resolve(
            "python -c 'arbitrary'",
            context(tmp_path),
            network_enabled=False,
        )
    with pytest.raises(CommandPolicyError, match="UNAPPROVED_NETWORK"):
        registry.resolve(
            "candidate_smoke",
            context(tmp_path),
            network_enabled=True,
        )
    with pytest.raises(CommandPolicyError, match="at least one"):
        registry.resolve(
            "candidate_smoke",
            context(tmp_path, attempt=0),
            network_enabled=False,
        )


def test_argument_is_one_argv_element_not_shell_syntax(tmp_path: Path) -> None:
    dangerous_literal = "; touch should-not-exist"
    registry = CommandRegistry(
        (
            CommandProfile(
                command_id="candidate_smoke",
                executable=str(Path(sys.executable).resolve()),
                arguments=("-c", "import sys; print(sys.argv[1])", dangerous_literal),
                allowed_fidelities=("smoke",),
            ),
        )
    )
    resolved = registry.resolve(
        "candidate_smoke", context(tmp_path), network_enabled=False
    )
    assert resolved.argv[-1] == dangerous_literal
    assert not (tmp_path / "should-not-exist").exists()


def test_registry_rejects_credentials_and_unknown_templates() -> None:
    with pytest.raises(CommandPolicyError, match="credential-shaped"):
        CommandProfile(
            command_id="candidate_smoke",
            executable=str(Path(sys.executable).resolve()),
            environment={"API_KEY": "value"},
        )
    with pytest.raises(CommandPolicyError, match="unknown command placeholder"):
        CommandProfile(
            command_id="candidate_smoke",
            executable=str(Path(sys.executable).resolve()),
            arguments=("{raw_shell}",),
        )


def test_pipeline_inputs_require_exact_canonical_views(tmp_path: Path) -> None:
    valid = pipeline_inputs(tmp_path / "valid")
    missing = dict(valid.input_roots)
    missing.pop("candidate_final_infer")
    with pytest.raises(CommandPolicyError, match="missing: candidate_final_infer"):
        PipelineCommandInputs(
            contract_root=valid.contract_root,
            input_roots=missing,
            baseline_entrypoint=valid.baseline_entrypoint,
            candidate_entrypoint=valid.candidate_entrypoint,
            submission_check_entrypoint=valid.submission_check_entrypoint,
        )

    noncanonical_contract = valid.contract_root / ".." / valid.contract_root.name
    with pytest.raises(CommandPolicyError, match="canonical directory"):
        PipelineCommandInputs(
            contract_root=noncanonical_contract,
            input_roots=valid.input_roots,
            baseline_entrypoint=valid.baseline_entrypoint,
            candidate_entrypoint=valid.candidate_entrypoint,
            submission_check_entrypoint=valid.submission_check_entrypoint,
        )

    with pytest.raises(CommandPolicyError, match="invalid candidate entrypoint"):
        PipelineCommandInputs(
            contract_root=valid.contract_root,
            input_roots=valid.input_roots,
            baseline_entrypoint=valid.baseline_entrypoint,
            candidate_entrypoint="arbitrary shell",
            submission_check_entrypoint=valid.submission_check_entrypoint,
        )


def test_registry_profiles_drive_generic_solution_contracts(tmp_path: Path) -> None:
    inputs = pipeline_inputs(tmp_path / "configuration")
    registry = default_command_registry(inputs)

    candidate = registry.resolve(
        "candidate_smoke",
        context(tmp_path / "candidate", fidelity="smoke"),
        network_enabled=False,
    )
    candidate_calls: list[PipelineInvocation] = []

    def candidate_loader(value: str) -> Callable[[object], None]:
        assert value == inputs.candidate_entrypoint

        def run(request: object) -> None:
            assert isinstance(request, PipelineInvocation)
            candidate_calls.append(request)
            request.output_path.write_text("row_id,score\n", encoding="utf-8")

        return run

    candidate_request = execute_pipeline(
        candidate.argv[4:],
        environment=candidate.environment,
        entrypoint_loader=candidate_loader,
    )
    assert candidate_calls == [candidate_request]
    assert candidate_request.input_root == inputs.input_roots["candidate_smoke"]
    assert candidate_request.fidelity == "smoke"
    assert candidate_request.seed == 3

    baseline = registry.resolve(
        "baseline_full",
        context(tmp_path / "baseline", fidelity="full"),
        network_enabled=False,
    )

    def baseline_loader(value: str) -> Callable[[object], None]:
        assert value == inputs.baseline_entrypoint

        def run(request: object) -> None:
            assert isinstance(request, PipelineInvocation)
            request.output_path.write_text("row_id,score\n", encoding="utf-8")

        return run

    baseline_request = execute_pipeline(
        baseline.argv[4:],
        environment=baseline.environment,
        entrypoint_loader=baseline_loader,
    )
    assert baseline_request.mode == "baseline"
    assert baseline_request.input_root == inputs.input_roots["baseline_full"]

    prior_prediction = tmp_path / "prior-prediction.csv"
    prior_prediction.write_text("row_id,score\n", encoding="utf-8")
    submission = registry.resolve(
        "submission_check",
        context(
            tmp_path / "submission",
            fidelity="full",
            submission_prediction_path=prior_prediction.resolve(),
        ),
        network_enabled=False,
    )
    prediction_path = Path(submission.argv[-1])
    submission_calls: list[SubmissionCheckInvocation] = []

    def submission_loader(
        value: str,
    ) -> Callable[[object], None]:
        assert value == inputs.submission_check_entrypoint

        def check(request: object) -> None:
            assert isinstance(request, SubmissionCheckInvocation)
            submission_calls.append(request)

        return check

    submission_request = execute_submission_check(
        submission.argv[4:],
        environment=submission.environment,
        entrypoint_loader=submission_loader,
    )
    assert submission_calls == [submission_request]
    assert submission_request.contract_root == inputs.contract_root
    assert "TACORANK_INPUT_ROOT" not in submission.environment
    assert submission.environment["TACORANK_VERIFIED_PREDICTION_PATH"] == str(
        prediction_path
    )


def test_command_profile_deep_freezes_mutable_inputs() -> None:
    arguments = ["-c", "print('ok')"]
    fidelities = ["smoke"]
    environment = {"TACORANK_MODE": "smoke"}
    artifacts: list[ExpectedArtifact] = []
    profile = CommandProfile(
        command_id="candidate_smoke",
        executable=str(Path(sys.executable).resolve()),
        arguments=arguments,  # type: ignore[arg-type]
        allowed_fidelities=fidelities,  # type: ignore[arg-type]
        environment=environment,
        expected_artifacts=artifacts,  # type: ignore[arg-type]
    )

    arguments.append("changed")
    fidelities.append("full")
    environment["TACORANK_MODE"] = "changed"
    assert profile.arguments == ("-c", "print('ok')")
    assert profile.allowed_fidelities == ("smoke",)
    assert profile.environment == MappingProxyType({"TACORANK_MODE": "smoke"})
    assert profile.expected_artifacts == ()


def test_entrypoint_import_failure_is_bounded_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(module_name: str) -> object:
        del module_name
        raise RuntimeError("API_TOKEN=must-not-escape")

    monkeypatch.setattr("tacorank.execution.solution_cli.importlib.import_module", fail_import)
    with pytest.raises(ControllerCLIError) as captured:
        _load_entrypoint("solution.candidate:run")
    assert str(captured.value) == (
        "configured implementation entrypoint could not be loaded"
    )
    assert "must-not-escape" not in str(captured.value)
