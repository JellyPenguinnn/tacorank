from __future__ import annotations

import asyncio
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tacorank.coding import (
    CandidateIdentity,
    SchemaFactories,
    TraeCodingWorker,
    TraeConfig,
)
from tacorank.execution import (
    CommandProfile,
    CommandRegistry,
    ExecutionRunner,
    ExpectedArtifact,
    ReceiptArtifactBinding,
    RunnerPolicy,
    SandboxPolicy,
    SealedExecutionVerifier,
    TrustedLocalProcessSandbox,
)
from tacorank.git import WorktreeManager
from tacorank.safety import (
    DataAccessPolicy,
    ExecutionSealExpectation,
    OutputColumn,
    OutputContract,
    OutputGate,
    PatchGate,
    ProtectedManifest,
    ReceiptStore,
    SharedSchemaFactories,
)
from tests.execution.store import TestArtifactStore


DATA_MANIFEST_SHA = "d" * 64

FAKE_TRAE = r'''import json
import sys
from pathlib import Path

if "--version" in sys.argv:
    print("trae-cli, version 0.1.0")
    raise SystemExit(0)

def option(name):
    return sys.argv[sys.argv.index(name) + 1]

worktree = Path(option("--working-dir"))
prompt = Path(option("--file")).read_text(encoding="utf-8")
(worktree / "solution" / "candidate.py").write_text(
    "VALUE = 7\n", encoding="utf-8"
)
provider = option("--provider")
model = option("--model")
trajectory = {
    "task": prompt,
    "start_time": "2026-08-29T00:00:00Z",
    "end_time": "2026-08-29T00:00:01Z",
    "provider": provider,
    "model": model,
    "max_steps": int(option("--max-steps")),
    "llm_interactions": [{
        "provider": provider,
        "model": model,
        "response": {
            "content": "patched",
            "usage": {"input_tokens": 8, "output_tokens": 3},
        },
    }],
    "agent_steps": [{"step_number": 1, "state": "completed"}],
    "success": True,
    "final_result": "patched",
    "execution_time": 1.0,
}
Path(option("--trajectory-file")).write_text(
    json.dumps(trajectory), encoding="utf-8"
)
'''


class Record(SimpleNamespace):
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return dict(self.__dict__)


def model_factory(model_name: str, **fields: Any) -> Record:
    return Record(_model_name=model_name, **fields)


def record_factory(**fields: Any) -> Record:
    return Record(**fields)


SHARED_FACTORIES = SharedSchemaFactories(
    check_result=record_factory,
    violation=record_factory,
    patch_check_result=record_factory,
    output_check_result=record_factory,
    artifact_ref=record_factory,
)


class IdentityResolver:
    def for_initial(self, context: Any, spec: Any) -> CandidateIdentity:
        del context, spec
        return CandidateIdentity(1, "event-spec-1")

    def for_repair(self, context: Any, decision: Any) -> CandidateIdentity:
        del context, decision
        return CandidateIdentity(2, "event-spec-1")


class ReceiptResolver:
    def __init__(
        self,
        receipt_id: str,
        artifact: Any,
        patch_attempt: int,
        experiment_root_commit_sha: str,
    ) -> None:
        self.receipt_id = receipt_id
        self.binding = ReceiptArtifactBinding(
            artifact,
            patch_attempt,
            experiment_root_commit_sha,
        )

    def resolve(self, request: Any) -> ReceiptArtifactBinding:
        if request.patch_receipt_id != self.receipt_id:
            raise KeyError("receipt is not controller-recorded")
        return self.binding


class ContinuingObserver:
    def observe(self, sample: Any) -> Record:
        del sample
        return Record(action="continue", reason_code=None, summary=None)


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


@pytest.mark.integration
def test_fake_trae_gate_a_sealed_execution_and_gate_b(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    (repository / "contract").mkdir()
    (repository / "contract" / "COMPETITION.md").write_text(
        "sealed integration contract\n", encoding="utf-8"
    )
    (repository / "solution").mkdir()
    (repository / "solution" / ".gitkeep").write_text("", encoding="utf-8")
    git(repository, "add", "--all")
    git(
        repository,
        "-c",
        "user.name=TacoRank Integration",
        "-c",
        "user.email=tacorank@invalid",
        "commit",
        "-q",
        "-m",
        "base",
    )
    base_commit = git(repository, "rev-parse", "HEAD")

    manifest = ProtectedManifest.capture(
        repository,
        ("contract",),
        contract_paths=("contract",),
        data_manifest_sha256=DATA_MANIFEST_SHA,
        require_minimum=False,
    )
    worktrees = WorktreeManager(repository, tmp_path / "worktrees")
    fake_trae = tmp_path / "fake_trae.py"
    fake_trae.write_text(FAKE_TRAE, encoding="utf-8")
    config_file = tmp_path / "trae.yaml"
    config_file.write_text(
        """agents:
  trae_agent:
    enable_lakeview: false
    model: fake-coder
    max_steps: 4
    tools:
      - str_replace_based_edit_tool
      - task_done
allow_mcp_servers: []
mcp_servers: {}
model_providers:
  fake:
    provider: fake-provider
    api_key: ""
models:
  fake-coder:
    model_provider: fake
    model: fake-model
    max_tokens: 100
    temperature: 0
    top_p: 1
    top_k: 0
    max_retries: 3
    parallel_tool_calls: false
""",
        encoding="utf-8",
    )
    config = TraeConfig(
        command_prefix=(str(Path(sys.executable).resolve()), str(fake_trae)),
        trae_version="0.1.0",
        provider="fake-provider",
        model_id="fake-model",
        config_file=config_file,
        config_sha256=hashlib.sha256(config_file.read_bytes()).hexdigest(),
        max_steps_cap=4,
        max_token_cap=100,
        max_wall_time_seconds_cap=5,
        repair_step_limit=2,
        repair_token_limit=40,
        repair_wall_time_limit_seconds=3,
        repair_allowed_command_ids=("candidate_smoke",),
        solution_revision_step_limit=2,
        solution_revision_wall_time_limit_seconds=3,
        trusted_test_mode=True,
    )
    spec = {
        "schema_version": "1.0",
        "run_id": "run_001",
        "experiment_id": "exp_0001",
        "parent_commit_sha": base_commit,
        "hypothesis": "Create the bounded candidate entry point.",
        "target_files": ["solution/candidate.py"],
    }
    context = Record(
        context_id="context-1",
        run_id="run_001",
        experiment_id="exp_0001",
        contract_sha256=manifest.contract_sha256,
        experiment_spec=spec,
        parent_commit_sha=base_commit,
        target_interface_excerpts={"entrypoint": "solution/candidate.py"},
        editable_roots=("solution",),
        protected_paths=("contract",),
        allowed_command_ids=("candidate_smoke",),
        selected_method_cards=(),
        active_lessons=(),
        step_limit=3,
        token_limit=30,
        wall_time_limit_seconds=3,
        context_artifact={"path": "artifacts/context.json"},
    )
    worker = TraeCodingWorker(
        worktrees=worktrees,
        artifact_repository_root=repository,
        config=config,
        identity_resolver=IdentityResolver(),
        factories=SchemaFactories(
            artifact_ref=record_factory,
            resource_delta=record_factory,
            patch_candidate=record_factory,
        ),
        process_environment={},
    )
    candidate = asyncio.run(worker.create_patch(context, spec))
    candidate_worktree = worktrees.path_for("run_001", "exp_0001")

    receipts = ReceiptStore(repository, SHARED_FACTORIES)
    gate_a = PatchGate(
        repository_root=candidate_worktree,
        artifact_repository_root=repository,
        editable_roots=("solution",),
        protected_manifest=manifest,
        receipt_store=receipts,
        data_access_policy=DataAccessPolicy(
            views=(),
            protected_columns=("protected_target",),
            hidden_path_tokens=("hidden_labels",),
            future_column_patterns=(r"(?:^|_)future(?:_|$)",),
        ),
        allowed_command_ids=("candidate_smoke",),
        artifact_roots=("artifacts", "runs"),
        factories=SHARED_FACTORIES,
        allowed_import_roots=(),
    )
    patch_check = asyncio.run(gate_a.check(candidate))
    assert patch_check.accepted is True
    assert patch_check.receipt_id

    artifact_store = TestArtifactStore(
        repository, repository / "artifacts", model_factory=model_factory
    )
    expected_prediction = ExpectedArtifact(
        role="prediction",
        relative_path="predictions.csv",
        kind="predictions",
        content_type="text/csv",
    )
    prediction_program = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).write_text("
        "'row_id,user_id,video_id,score\\n0,10,100,0.2\\n1,10,100,0.8\\n', "
        "encoding='utf-8')"
    )
    commands = CommandRegistry(
        (
            CommandProfile(
                command_id="candidate_smoke",
                executable=str(Path(sys.executable).resolve()),
                arguments=("-c", prediction_program, "{prediction_path}"),
                allowed_fidelities=("smoke",),
                expected_artifacts=(expected_prediction,),
            ),
        )
    )
    receipt_resolver = ReceiptResolver(
        patch_check.receipt_id,
        patch_check.receipt_artifact,
        candidate.attempt,
        base_commit,
    )
    sealed_verifier = SealedExecutionVerifier(
        worktrees=worktrees,
        receipts=receipts,
        protected_manifest=manifest,
        receipt_artifact_resolver=receipt_resolver,
    )
    sandbox = TrustedLocalProcessSandbox(
        SandboxPolicy(
            allowed_workspace_roots=(worktrees.worktree_root,),
            allowed_artifact_roots=(repository / "artifacts",),
        ),
        allow_unsafe_for_tests=True,
    )
    runner = ExecutionRunner(
        repository_root=repository,
        artifacts=artifact_store,
        commands=commands,
        sandbox=sandbox,
        workspace_resolver=lambda run_id, experiment_id: candidate_worktree,
        seal_verifier=sealed_verifier,
        policy=RunnerPolicy(
            telemetry_interval_seconds=0.02,
            termination_grace_seconds=0.1,
            disk_free_floor_mb=0,
            allow_trusted_local_backend=True,
        ),
        model_factory=model_factory,
    )
    request = Record(
        run_id="run_001",
        experiment_id="exp_0001",
        attempt=1,
        fidelity="smoke",
        command_id="candidate_smoke",
        patch_commit_sha=candidate.patch_commit_sha,
        patch_receipt_id=patch_check.receipt_id,
        seed=7,
        data_manifest_sha256=DATA_MANIFEST_SHA,
        timeout_seconds=3.0,
        memory_limit_mb=4096,
        gpu_memory_limit_mb=0,
        network_enabled=False,
    )
    run_result = runner.run_sync(request, ContinuingObserver())
    assert run_result.outcome == "success"

    gate_b = OutputGate(
        repository_root=repository,
        contract=OutputContract(
            columns=(
                OutputColumn("row_id", "integer"),
                OutputColumn("user_id", "integer"),
                OutputColumn("video_id", "integer"),
                OutputColumn("score", "number"),
            ),
            score_column="score",
            expected_rows=(
                {"row_id": 0, "user_id": 10, "video_id": 100},
                {"row_id": 1, "user_id": 10, "video_id": 100},
            ),
            identity_columns=("user_id", "video_id"),
            forbidden_columns=("protected_target",),
            minimum_unique_scores=2,
        ),
        factories=SHARED_FACTORIES,
    )
    output_check = asyncio.run(
        gate_b.check(
            run_result,
            expected_execution=ExecutionSealExpectation(
                run_id=request.run_id,
                experiment_id=request.experiment_id,
                execution_attempt=request.attempt,
                producer_commit_sha=candidate.patch_commit_sha,
                command_id=request.command_id,
                data_manifest_sha256=request.data_manifest_sha256,
                patch_receipt_id=patch_check.receipt_id,
                patch_receipt_sha256=patch_check.receipt_artifact.sha256,
            ),
        )
    )
    assert output_check.accepted is True
    assert output_check.prediction_artifact.sha256 == run_result.prediction_artifact.sha256
