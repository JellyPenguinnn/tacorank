from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tacorank.coding.trae_adapter import (
    CandidateIdentity,
    CodingWorkerError,
    SchemaFactories,
    SchemaIntegrationError,
    TraeCodingWorker,
    TraeConfig,
    hash_trae_runtime_package,
)
from tacorank.coding.solution_verifier import (
    AcceptingSolutionVerifier,
    SolutionFinding,
    SolutionVerificationResult,
    SolutionVerifierError,
)
from tacorank.git.patches import capture_commit_patch
from tacorank.git.worktrees import WorktreeManager
from tacorank.run_layout import experiment_artifact_prefix


_FAKE_TRAE = r'''from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

if "--version" in sys.argv:
    print("trae-cli, version 0.1.0")
    raise SystemExit(0)

if len(sys.argv) < 2 or sys.argv[1] != "run":
    raise SystemExit(2)

marker = os.environ.get("FAKE_TRAE_RUN_MARKER")
if marker:
    Path(marker).write_text("run\n", encoding="utf-8")

required_flags = {
    "--file", "--provider", "--model", "--max-steps", "--working-dir",
    "--must-patch", "--config-file", "--trajectory-file", "--console-type",
}
if not required_flags.issubset(sys.argv):
    raise SystemExit(3)

def option(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]

behavior = os.environ.get("FAKE_TRAE_BEHAVIOR", "patch")
if behavior == "timeout":
    time.sleep(30)

prompt = Path(option("--file")).read_text(encoding="utf-8")
working_dir = Path(option("--working-dir"))
trajectory_path = Path(option("--trajectory-file"))
max_steps = int(option("--max-steps"))
provider = option("--provider")
model = option("--model")

if behavior != "no_patch":
    candidate = working_dir / "solution" / "candidate.py"
    previous = candidate.read_text(encoding="utf-8") if candidate.exists() else ""
    if behavior == "credential_patch":
        body = "TOKEN = " + repr(os.environ["FAKE_TRAE_SECRET"]) + "\n"
    else:
        body = previous + "VALUE = " + str(previous.count("VALUE") + 1) + "\n"
    candidate.write_text(body, encoding="utf-8")

usage = {"input_tokens": 7, "output_tokens": 4}
if behavior == "over_tokens":
    usage = {"input_tokens": 70, "output_tokens": 40}
response = {"content": "credential=" + os.environ.get("FAKE_TRAE_SECRET", "none")}
if behavior != "missing_usage":
    response["usage"] = usage
success = behavior != "reported_failure"
final_result = (
    "provider rejected continuation for secret="
    + os.environ.get("FAKE_TRAE_SECRET", "none")
    if behavior == "reported_failure"
    else "patched"
)
trajectory = {
    "task": prompt,
    "start_time": "2026-01-01T00:00:00",
    "end_time": "2026-01-01T00:00:01",
    "provider": provider,
    "model": model,
    "max_steps": max_steps,
    "llm_interactions": [{"provider": provider, "model": model, "response": response}],
    "agent_steps": [{"step_number": 1, "state": "completed"}],
    "success": success,
    "final_result": final_result,
    "execution_time": 1.0,
    "fake_cli_arguments": sys.argv[1:],
    "fake_environment": {
        "PYTHON_DOTENV_DISABLED": os.environ.get("PYTHON_DOTENV_DISABLED"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "PYTHONUTF8": os.environ.get("PYTHONUTF8"),
        "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING"),
        "PATH": os.environ.get("PATH"),
        "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG"),
        "HOME": os.environ.get("HOME"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "XDG_CONFIG_HOME": os.environ.get("XDG_CONFIG_HOME"),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
    },
    "fake_cwd": str(Path.cwd()),
}
trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
if behavior == "process_failure_with_trajectory":
    print("provider failed after saving trajectory")
    raise SystemExit(9)
print("finished " + os.environ.get("FAKE_TRAE_SECRET", ""))
'''


_FAKE_DOCKER = r'''from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
log_path = root / "docker.log"
state_path = root / "docker.state"
behavior_path = root / "docker.behavior"
behavior = behavior_path.read_text(encoding="utf-8").strip() if behavior_path.exists() else "ok"
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "argv": sys.argv[1:],
        "environment": dict(os.environ),
    }, sort_keys=True) + "\n")

command = sys.argv[1] if len(sys.argv) > 1 else ""
container_id = "d" * 64
if command == "image":
    print("sha256:" + "a" * 64)
elif command == "create":
    if behavior == "fail_create":
        print("create rejected", file=sys.stderr)
        raise SystemExit(2)
    state_path.write_text(container_id, encoding="ascii")
    cidfile = Path(sys.argv[sys.argv.index("--cidfile") + 1])
    cidfile.write_text(container_id + "\n", encoding="ascii")
    print(container_id)
    if behavior == "create_then_sleep":
        time.sleep(30)
elif command == "start":
    if behavior == "fail_start":
        print("start rejected", file=sys.stderr)
        raise SystemExit(2)
elif command == "stop":
    pass
elif command == "rm":
    if behavior == "fail_remove":
        print("remove rejected", file=sys.stderr)
        raise SystemExit(2)
    state_path.unlink(missing_ok=True)
elif command == "inspect":
    if state_path.exists():
        print("{}")
    else:
        print("Error: No such object: " + container_id, file=sys.stderr)
        raise SystemExit(1)
else:
    print("unexpected command", file=sys.stderr)
    raise SystemExit(64)
'''


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class _IdentityResolver:
    def __init__(self) -> None:
        self.initial: Any = CandidateIdentity(1, "event-spec-1")
        self.repair: Any = CandidateIdentity(2, "event-spec-1")

    def for_initial(self, context: Any, spec: Any) -> Any:
        return self.initial

    def for_repair(self, context: Any, decision: Any) -> Any:
        return self.repair


@pytest.fixture
def adapter_parts(tmp_path: Path) -> SimpleNamespace:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.invalid")
    (repository / "solution").mkdir()
    (repository / "solution" / ".gitkeep").write_text("", encoding="utf-8")
    _git(repository, "add", "solution/.gitkeep")
    _git(repository, "commit", "-q", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")

    fake_cli = tmp_path / "fake_trae.py"
    fake_cli.write_text(_FAKE_TRAE, encoding="utf-8")
    config_file = tmp_path / "trae_config.yaml"
    config_file.write_text(
        """agents:
  trae_agent:
    enable_lakeview: false
    model: fake-coder
    max_steps: 6
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
        command_prefix=(str(Path(sys.executable).resolve()), str(fake_cli)),
        trae_version="0.1.0",
        provider="fake-provider",
        model_id="fake-model",
        config_file=config_file,
        config_sha256=hashlib.sha256(config_file.read_bytes()).hexdigest(),
        max_steps_cap=6,
        max_token_cap=100,
        max_wall_time_seconds_cap=3,
        repair_step_limit=3,
        repair_token_limit=40,
        repair_wall_time_limit_seconds=2,
        repair_allowed_command_ids=("candidate_smoke",),
        solution_revision_step_limit=3,
        solution_revision_wall_time_limit_seconds=2,
        approved_environment_names=(
            "FAKE_TRAE_BEHAVIOR",
            "FAKE_TRAE_SECRET",
            "FAKE_TRAE_RUN_MARKER",
            "PYTHON_DOTENV_DISABLED",
        ),
        credential_environment_names=("FAKE_TRAE_SECRET",),
        solution_verifier_credential_environment_name="FAKE_TRAE_SECRET",
        trae_source_revision="e839e559ac61bdd0e057c375dd1dee391fee797d",
        trusted_test_mode=True,
        version_timeout_seconds=2,
        termination_grace_seconds=1,
    )
    secret = "secret-value-123456"
    environment = {
        "FAKE_TRAE_BEHAVIOR": "patch",
        "FAKE_TRAE_SECRET": secret,
        "FAKE_TRAE_RUN_MARKER": str(tmp_path / "trae-run.marker"),
        "PYTHON_DOTENV_DISABLED": "0",
        "HOME": "/sensitive-real-home",
    }
    spec = {
        "schema_version": "1.0",
        "run_id": "run1",
        "experiment_id": "exp1",
        "parent_commit_sha": base,
        "hypothesis": "Add the candidate module.",
        "target_files": ["solution/candidate.py"],
    }
    context = SimpleNamespace(
        context_id="context-1",
        run_id="run1",
        experiment_id="exp1",
        contract_sha256="a" * 64,
        experiment_spec=spec,
        parent_commit_sha=base,
        target_interface_excerpts={"entrypoint": "predict(rows)"},
        editable_roots=("solution",),
        protected_paths=("contract", "runs"),
        allowed_command_ids=("candidate_smoke",),
        selected_method_cards=(),
        active_lessons=({"summary": f"Never expose {secret}"},),
        step_limit=4,
        token_limit=20,
        wall_time_limit_seconds=2,
        context_artifact={"path": "artifacts/context.json"},
    )
    factories = SchemaFactories(
        artifact_ref=lambda **values: SimpleNamespace(**values),
        resource_delta=lambda **values: SimpleNamespace(**values),
        patch_candidate=lambda **values: SimpleNamespace(**values),
    )
    return SimpleNamespace(
        repository=repository,
        base=base,
        config=config,
        environment=environment,
        secret=secret,
        spec=spec,
        context=context,
        factories=factories,
        resolver=_IdentityResolver(),
        worktrees=WorktreeManager(repository, tmp_path / "worktrees"),
    )


def _worker(parts: SimpleNamespace, solution_verifier: Any = None) -> TraeCodingWorker:
    return TraeCodingWorker(
        worktrees=parts.worktrees,
        artifact_repository_root=parts.repository,
        config=parts.config,
        identity_resolver=parts.resolver,
        factories=parts.factories,
        process_environment=parts.environment,
        solution_verifier=solution_verifier or AcceptingSolutionVerifier(),
    )


class _SequencedVerifier:
    def __init__(self, accepted: list[bool]) -> None:
        self.accepted = list(accepted)
        self.calls = []

    def verify(self, **values: Any) -> SolutionVerificationResult:
        self.calls.append(values)
        accepted = self.accepted.pop(0)
        return SolutionVerificationResult(
            accepted=accepted,
            summary=("implementation matches" if accepted else "mechanism is incomplete"),
            findings=(
                ()
                if accepted
                else (
                    SolutionFinding(
                        "MECHANISM_INCOMPLETE",
                        "error",
                        "solution/candidate.py",
                        "The approved mechanism is not fully wired.",
                    ),
                )
            ),
            required_changes=(
                () if accepted else ("Complete the approved mechanism wiring.",)
            ),
            input_tokens=3,
            output_tokens=2,
            wall_time_ms=1,
            provider_calls=1,
        )


class _UnavailableThenAcceptingVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, **values: Any) -> SolutionVerificationResult:
        self.calls += 1
        if self.calls == 1:
            raise SolutionVerifierError(
                "TRAE_PROVIDER_UNAVAILABLE",
                "solution verifier provider request failed",
                input_tokens=3,
                output_tokens=1,
                wall_time_ms=1,
            )
        return AcceptingSolutionVerifier().verify(**values)


def _production_worker(parts: SimpleNamespace) -> tuple[TraeCodingWorker, Path]:
    root = parts.config.config_file.parent
    install_root = root / "trae-install"
    bin_directory = install_root / "bin"
    identity_directory = (
        install_root / "lib" / "python3.12" / "site-packages" / "trae_agent-0.1.0.dist-info"
    )
    bin_directory.mkdir(parents=True)
    identity_directory.mkdir(parents=True)
    dotenv_identity_directory = (
        install_root / "lib" / "python3.12" / "site-packages" / "python_dotenv-1.2.2.dist-info"
    )
    dotenv_identity_directory.mkdir(parents=True)
    trae_cli = bin_directory / "trae-cli"
    trae_cli.write_text("#!/usr/bin/env python3\n" + _FAKE_TRAE, encoding="utf-8")
    trae_cli.chmod(0o755)
    direct_url = identity_directory / "direct_url.json"
    direct_url.write_text(
        json.dumps(
            {
                "url": "https://github.com/bytedance/trae-agent.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "e839e559ac61bdd0e057c375dd1dee391fee797d",
                    "requested_revision": "e839e559ac61bdd0e057c375dd1dee391fee797d",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dotenv_metadata = dotenv_identity_directory / "METADATA"
    dotenv_metadata.write_text(
        "Metadata-Version: 2.4\nName: python-dotenv\nVersion: 1.2.2\n",
        encoding="utf-8",
    )
    runtime_root = identity_directory.parent
    package_root = runtime_root / "trae_agent"
    assets = package_root / "dist"
    internal = assets / "_internal"
    internal.mkdir(parents=True)
    (internal / "runtime.bin").write_bytes(b"reviewed-runtime")
    for name in ("edit_tool", "json_edit_tool"):
        tool = assets / name
        tool.write_bytes((name + "\n").encode())
        tool.chmod(0o755)
    for relative in (
        "cli.py",
        "agent/base_agent.py",
        "agent/docker_manager.py",
        "agent/trae_agent.py",
        "tools/docker_tool_executor.py",
    ):
        source = package_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# reviewed pinned source fixture\n", encoding="utf-8")
    (package_root / "agent" / "docker_manager.py").write_text(
        "# TacoRank: use cross-platform bounded stateless Docker exec\n"
        "# self._execute_stateless(command, timeout)\n"
        '# ["timeout", "--signal=KILL"]\n',
        encoding="utf-8",
    )
    docker = root / "docker"
    docker.write_text("#!/usr/bin/env python3\n" + _FAKE_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    parts.config = replace(
        parts.config,
        command_prefix=(str(trae_cli),),
        approved_environment_names=tuple(
            name
            for name in parts.config.approved_environment_names
            if name != "PYTHON_DOTENV_DISABLED"
        ),
        trusted_test_mode=False,
        trae_install_root=install_root,
        trae_install_identity_file=direct_url,
        trae_install_identity_sha256=hashlib.sha256(direct_url.read_bytes()).hexdigest(),
        trae_executable_sha256=hashlib.sha256(trae_cli.read_bytes()).hexdigest(),
        trae_runtime_root=runtime_root,
        trae_runtime_manifest_sha256=hash_trae_runtime_package(runtime_root),
        python_dotenv_metadata_file=dotenv_metadata,
        python_dotenv_metadata_sha256=hashlib.sha256(
            dotenv_metadata.read_bytes()
        ).hexdigest(),
        docker_image="tacorank/trae@sha256:" + "a" * 64,
        docker_executable=docker,
    )
    return _worker(parts), docker


def test_real_adapter_seals_patch_and_redacted_evidence(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    candidate = asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))

    assert candidate.attempt == 1
    assert candidate.base_commit_sha == parts.base
    assert candidate.changed_files == ["solution/candidate.py"]
    assert candidate.diff_sha256 == candidate.diff_artifact.sha256
    assert candidate.resource_delta.llm_input_tokens == 7
    assert candidate.resource_delta.llm_output_tokens == 4
    assert candidate.trae_version == parts.config.trae_source_revision
    sealed = capture_commit_patch(
        parts.worktrees.path_for("run1", "exp1"),
        parts.base,
        candidate.patch_commit_sha,
    )
    assert sealed.diff_sha256 == candidate.diff_sha256

    trajectory_bytes = (parts.repository / candidate.trajectory_artifact.path).read_bytes()
    assert parts.secret.encode() not in trajectory_bytes
    trajectory = json.loads(trajectory_bytes)
    assert trajectory["tacorank_adapter"]["config_sha256"] == parts.config.config_sha256
    assert (
        trajectory["tacorank_adapter"]["trae_source_revision"]
        == parts.config.trae_source_revision
    )
    assert trajectory["tacorank_adapter"]["max_provider_tokens"] == 20
    assert trajectory["tacorank_adapter"]["max_provider_tokens"] != parts.config.max_token_cap
    assert trajectory["tacorank_adapter"]["redacted"] is True
    assert trajectory["tacorank_adapter"]["isolation_mode"] == "trusted_test"
    assert trajectory["fake_environment"]["PYTHON_DOTENV_DISABLED"] == "1"
    assert trajectory["fake_environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert trajectory["fake_environment"]["PYTHONUTF8"] == "1"
    assert trajectory["fake_environment"]["PYTHONIOENCODING"] == "utf-8"
    assert trajectory["fake_environment"]["HOME"] != "/sensitive-real-home"
    assert trajectory["fake_environment"]["HOME"].endswith("/home")
    assert trajectory["fake_environment"]["TMPDIR"].endswith("/tmp")
    assert trajectory["fake_environment"]["XDG_CONFIG_HOME"].endswith("/xdg-config")
    assert trajectory["fake_environment"]["XDG_CACHE_HOME"].endswith("/xdg-cache")
    cli_arguments = trajectory["fake_cli_arguments"]
    assert cli_arguments[0] == "run"
    assert cli_arguments[1] == "--file"
    assert cli_arguments[3:9] == [
        "--provider",
        "fake-provider",
        "--model",
        "fake-model",
        "--max-steps",
        "4",
    ]
    assert "--working-dir" in cli_arguments
    assert "--must-patch" in cli_arguments
    assert cli_arguments[-2:] == ["--console-type", "simple"]


def test_real_adapter_emits_person2_canonical_models(
    adapter_parts: SimpleNamespace,
) -> None:
    from tacorank.schemas import PatchCandidate

    parts = adapter_parts
    parts.factories = SchemaFactories.from_shared_schemas()

    candidate = asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))

    assert isinstance(candidate, PatchCandidate)
    assert candidate.diff_artifact.artifact_id == "sha256-" + candidate.diff_sha256


def test_semantic_review_revises_then_accepts_and_records_complete_evidence(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.context.wall_time_limit_seconds = 3
    verifier = _SequencedVerifier([False, True])

    candidate = asyncio.run(
        _worker(parts, verifier).create_patch(parts.context, parts.spec)
    )

    assert len(verifier.calls) == 2
    assert verifier.calls[0]["diff_sha256"] != verifier.calls[1]["diff_sha256"]
    assert candidate.steps_used == 2
    assert candidate.resource_delta.llm_input_tokens == 20
    assert candidate.resource_delta.llm_output_tokens == 12
    trajectory = json.loads(
        (parts.repository / candidate.trajectory_artifact.path).read_text(
            encoding="utf-8"
        )
    )
    assert "# TacoRank bounded implementation-fidelity revision" in trajectory["task"]
    assert "The approved mechanism is not fully wired." in trajectory["task"]
    assert "Complete the approved mechanism wiring." in trajectory["task"]
    verification_meta = trajectory["tacorank_solution_verification"]
    assert verification_meta["accepted"] is True
    assert verification_meta["completed_attempts"] == 2
    index_path = parts.repository / verification_meta["artifact"]["path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["accepted"] is True
    assert [item["accepted"] for item in index["reviews"]] == [False, True]
    assert all(
        (parts.repository / item["artifacts"]["process_log"]["path"]).is_file()
        for item in index["reviews"]
    )


def test_semantic_review_exhaustion_fails_without_sealing_a_patch(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.context.wall_time_limit_seconds = 3
    parts.config = replace(parts.config, solution_verification_max_attempts=2)
    verifier = _SequencedVerifier([False, False])

    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts, verifier).create_patch(parts.context, parts.spec))

    assert failure.value.code == "SOLUTION_VERIFICATION_FAILED"
    assert len(verifier.calls) == 2
    assert failure.value.resource_delta.llm_input_tokens == 20
    assert failure.value.resource_delta.llm_output_tokens == 12
    assert _git(parts.worktrees.path_for("run1", "exp1"), "rev-parse", "HEAD") == parts.base
    assert any(artifact.kind == "report" for artifact in failure.value.diagnostic_artifacts)


def test_transient_verifier_failure_leaves_clean_worktree_for_same_commit_retry(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    verifier = _UnavailableThenAcceptingVerifier()
    worker = _worker(parts, verifier)

    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(worker.create_patch(parts.context, parts.spec))

    assert failure.value.code == "TRAE_PROVIDER_UNAVAILABLE"
    worktree = parts.worktrees.path_for("run1", "exp1")
    assert _git(worktree, "rev-parse", "HEAD") == parts.base
    assert _git(
        worktree,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
    ) == ""

    parts.resolver.initial = CandidateIdentity(2, "event-spec-1")
    candidate = asyncio.run(worker.create_patch(parts.context, parts.spec))

    assert verifier.calls == 2
    assert candidate.attempt == 2
    assert candidate.base_commit_sha == parts.base
    assert (worktree / "solution" / "candidate.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_repair_is_a_direct_commit_on_the_same_branch(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    worker = _worker(parts)
    initial = asyncio.run(worker.create_patch(parts.context, parts.spec))
    recovery = SimpleNamespace(
        context_id="context-repair",
        run_id="run1",
        experiment_id="exp1",
        repair_attempt=1,
        original_experiment_spec=parts.spec,
        current_patch_commit_sha=initial.patch_commit_sha,
        accepted_patch_receipt_id="receipt-1",
        failure_class="code_error",
        error_fingerprint="fingerprint-1",
        error_summary="candidate import failed",
        relevant_trace_tail="candidate.py:1",
        failed_checks=({"name": "smoke", "status": "fail"},),
        previous_repair_fingerprints=(),
        recovery_instructions="Correct the candidate module only.",
        remaining_repair_budget=1,
        target_interface_excerpts={"entrypoint": "predict(rows)"},
        editable_roots=("solution",),
        protected_paths=("contract", "runs"),
    )
    repaired = asyncio.run(
        worker.repair_patch(
            recovery,
            {"action": "trae_repair", "instructions": "correct candidate"},
        )
    )
    assert repaired.attempt == 2
    assert repaired.base_commit_sha == initial.patch_commit_sha
    sealed = capture_commit_patch(
        parts.worktrees.path_for("run1", "exp1"),
        initial.patch_commit_sha,
        repaired.patch_commit_sha,
    )
    assert sealed.patch_commit_sha == repaired.patch_commit_sha
    trajectory = json.loads(
        (parts.repository / repaired.trajectory_artifact.path).read_bytes()
    )
    assert trajectory["tacorank_adapter"]["max_provider_tokens"] == 40


def test_second_repair_rewinds_rejected_tip_to_last_accepted_commit(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    worker = _worker(parts)
    initial = asyncio.run(worker.create_patch(parts.context, parts.spec))
    recovery = SimpleNamespace(
        context_id="context-repair-1",
        run_id="run1",
        experiment_id="exp1",
        repair_attempt=1,
        original_experiment_spec=parts.spec,
        current_patch_commit_sha=initial.patch_commit_sha,
        accepted_patch_receipt_id="receipt-1",
        failure_class="code_error",
        error_fingerprint="fingerprint-1",
        error_summary="candidate import failed",
        relevant_trace_tail="candidate.py:1",
        failed_checks=({"name": "smoke", "status": "fail"},),
        previous_repair_fingerprints=(),
        recovery_instructions="Correct the candidate module only.",
        remaining_repair_budget=1,
        target_interface_excerpts={"entrypoint": "predict(rows)"},
        editable_roots=("solution",),
        protected_paths=("contract", "runs"),
    )
    rejected = asyncio.run(
        worker.repair_patch(
            recovery,
            {"action": "trae_repair", "instructions": "first repair"},
        )
    )

    parts.resolver.repair = CandidateIdentity(3, "event-spec-1")
    retry_context = SimpleNamespace(
        **{
            **vars(recovery),
            "context_id": "context-repair-2",
            "repair_attempt": 2,
            "error_fingerprint": "fingerprint-2",
            "error_summary": "Gate A rejected the first repair",
            "relevant_trace_tail": "UNAPPROVED_TARGET_FILE",
            "remaining_repair_budget": 1,
        }
    )
    repaired = asyncio.run(
        worker.repair_patch(
            retry_context,
            {"action": "trae_repair", "instructions": "second repair"},
        )
    )

    assert repaired.base_commit_sha == initial.patch_commit_sha
    assert repaired.patch_commit_sha != rejected.patch_commit_sha
    worktree = parts.worktrees.path_for("run1", "exp1")
    assert _git(worktree, "rev-parse", "HEAD") == repaired.patch_commit_sha
    assert (worktree / "solution" / "candidate.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\nVALUE = 2\n"


@pytest.mark.parametrize(
    ("behavior", "expected_code"),
    [
        ("no_patch", "NO_PATCH"),
        ("missing_usage", "TOKEN_USAGE_MISSING"),
        ("credential_patch", "CREDENTIAL_IN_PATCH"),
        ("over_tokens", "TOKEN_LIMIT_EXCEEDED"),
        ("timeout", "TRAE_TIMEOUT"),
    ],
)
def test_adapter_fails_closed_for_invalid_coding_results(
    adapter_parts: SimpleNamespace, behavior: str, expected_code: str
) -> None:
    parts = adapter_parts
    parts.environment["FAKE_TRAE_BEHAVIOR"] = behavior
    if behavior == "timeout":
        parts.context.wall_time_limit_seconds = 1
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == expected_code
    assert parts.secret not in str(failure.value)
    assert parts.secret not in (failure.value.output_tail or "")


def test_adapter_allows_all_reported_tokens_when_limit_is_null(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.config = replace(
        parts.config,
        max_token_cap=None,
        repair_token_limit=None,
    )
    parts.context.token_limit = None
    parts.environment["FAKE_TRAE_BEHAVIOR"] = "over_tokens"

    candidate = asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))

    assert candidate.resource_delta.llm_input_tokens == 70
    assert candidate.resource_delta.llm_output_tokens == 40
    trajectory = json.loads(
        (parts.repository / candidate.trajectory_artifact.path).read_bytes()
    )
    assert trajectory["tacorank_adapter"]["max_provider_tokens"] is None


def test_reported_failure_retains_only_redacted_bounded_diagnostic(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.environment["FAKE_TRAE_BEHAVIOR"] = "reported_failure"

    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))

    assert failure.value.code == "TRAE_REPORTED_FAILURE"
    assert parts.secret not in (failure.value.output_tail or "")
    assert "[REDACTED]" in (failure.value.output_tail or "")
    assert failure.value.resource_delta.llm_input_tokens == 7
    assert failure.value.resource_delta.llm_output_tokens == 4
    assert len(failure.value.diagnostic_artifacts) == 2
    assert {artifact.kind for artifact in failure.value.diagnostic_artifacts} == {
        "trajectory",
        "log",
    }
    failure_artifact = parts.repository / experiment_artifact_prefix(
        "run1", "exp1", attempt=1
    ) / "trae_failure_trajectory.json"
    assert failure_artifact.is_file()
    assert parts.secret.encode("utf-8") not in failure_artifact.read_bytes()
    process_log = failure_artifact.with_name("trae_process.log")
    assert process_log.is_file()
    assert parts.secret.encode("utf-8") not in process_log.read_bytes()


def test_process_failure_retains_available_trajectory_usage_and_clean_retry_state(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.environment["FAKE_TRAE_BEHAVIOR"] = "process_failure_with_trajectory"

    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))

    assert failure.value.code == "TRAE_PROCESS_FAILED"
    assert failure.value.resource_delta.llm_input_tokens == 7
    assert failure.value.resource_delta.llm_output_tokens == 4
    assert {artifact.kind for artifact in failure.value.diagnostic_artifacts} == {
        "trajectory",
        "log",
    }
    worktree = parts.worktrees.path_for("run1", "exp1")
    assert _git(worktree, "rev-parse", "HEAD") == parts.base
    assert _git(worktree, "status", "--porcelain=v1") == ""


def test_candidate_identity_is_required_before_worktree_or_trae(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.resolver.initial = None
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "CANDIDATE_IDENTITY_MISSING"
    assert not parts.worktrees.path_for("run1", "exp1").exists()

    with pytest.raises(ValueError, match="positive"):
        CandidateIdentity(0, "event-spec-1")


def test_context_limits_are_checked_before_worktree_creation(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.context.step_limit = parts.config.max_steps_cap + 1
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "CODING_LIMIT_INVALID"
    assert not parts.worktrees.path_for("run1", "exp1").exists()


def test_solution_verifier_output_budget_is_validated_at_startup(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.config = replace(parts.config, solution_verification_max_output_tokens=255)

    with pytest.raises(CodingWorkerError) as failure:
        _worker(parts)

    assert failure.value.code == "TRAE_CONFIG_INVALID"


def test_missing_shared_schema_models_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch, adapter_parts: SimpleNamespace
) -> None:
    from tacorank import schemas

    monkeypatch.delattr(schemas, "ArtifactRef", raising=False)
    monkeypatch.delattr(schemas, "ResourceDelta", raising=False)
    monkeypatch.delattr(schemas, "PatchCandidate", raising=False)
    with pytest.raises(SchemaIntegrationError, match="ArtifactRef, ResourceDelta, PatchCandidate"):
        SchemaFactories.from_shared_schemas()

    parts = adapter_parts
    worker = TraeCodingWorker(
        worktrees=parts.worktrees,
        artifact_repository_root=parts.repository,
        config=parts.config,
        identity_resolver=parts.resolver,
        factories=None,
        process_environment=parts.environment,
    )
    with pytest.raises(SchemaIntegrationError):
        asyncio.run(worker.create_patch(parts.context, parts.spec))
    assert not parts.worktrees.path_for("run1", "exp1").exists()


def test_config_hash_and_cli_version_are_pinned(adapter_parts: SimpleNamespace) -> None:
    parts = adapter_parts
    parts.config = replace(parts.config, config_sha256="0" * 64)
    with pytest.raises(CodingWorkerError) as failure:
        _worker(parts)
    assert failure.value.code == "TRAE_CONFIG_HASH_MISMATCH"

    parts.config = replace(
        parts.config,
        config_sha256=hashlib.sha256(parts.config.config_file.read_bytes()).hexdigest(),
        trae_version="9.9.9",
    )
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "TRAE_VERSION_MISMATCH"

    parts.config.config_file.write_text(
        "provider: fake-provider\napi_key: literal-secret-value\n",
        encoding="utf-8",
    )
    parts.config = replace(
        parts.config,
        trae_version="0.1.0",
        config_sha256=hashlib.sha256(parts.config.config_file.read_bytes()).hexdigest(),
    )
    with pytest.raises(CodingWorkerError) as failure:
        _worker(parts)
    assert failure.value.code == "CREDENTIAL_IN_CONFIG"


def test_production_docker_boundary_has_exact_lifecycle_and_cli(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.context.wall_time_limit_seconds = 10
    parts.config = replace(parts.config, max_wall_time_seconds_cap=10)
    worker, docker = _production_worker(parts)
    candidate = asyncio.run(worker.create_patch(parts.context, parts.spec))

    calls = [json.loads(line) for line in docker.with_name("docker.log").read_text().splitlines()]
    container_id = "d" * 64
    worktree = parts.worktrees.path_for("run1", "exp1")
    expected_create = [
        "create",
        "--cidfile",
        calls[0]["argv"][2],
        "--name",
        "tacorank-trae-" + hashlib.sha256(
            "\0".join(("run1", "exp1", "1", parts.base)).encode()
        ).hexdigest()[:20],
        "--label",
        "com.tacorank.owner=coding-worker",
        "--label",
        "com.tacorank.run-id=run1",
        "--label",
        "com.tacorank.experiment-id=exp1",
        "--label",
        "com.tacorank.attempt=1",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--memory",
        "4096m",
        "--memory-swap",
        "4096m",
        "--cpus",
        "2",
        "--pids-limit",
        "128",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
        "--mount",
        (
            f"type=bind,src={parts.config.trae_runtime_root / 'trae_agent' / 'dist'},"
            "dst=/agent_tools,readonly,bind-propagation=rprivate"
        ),
        "--mount",
        f"type=bind,src={worktree},dst=/workspace,bind-propagation=rprivate",
        "--mount",
        (
            f"type=bind,src={worktree / '.git'},dst=/workspace/.git,"
            "readonly,bind-propagation=rprivate"
        ),
        "--workdir",
        "/workspace",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--init",
        "--ulimit",
        "nofile=256:256",
        "--pull",
        "never",
        "--entrypoint",
        "/bin/sh",
        "tacorank/trae@sha256:" + "a" * 64,
        "-c",
        "exec sleep infinity",
    ]
    assert [call["argv"] for call in calls] == [
        expected_create,
        ["start", container_id],
        ["stop", "--time", "1", container_id],
        ["rm", "--force", "--volumes", container_id],
        ["inspect", "--type", "container", container_id],
    ]
    for call in calls:
        assert "FAKE_TRAE_SECRET" not in call["environment"]
        assert parts.secret not in json.dumps(call)

    trajectory = json.loads(
        (parts.repository / candidate.trajectory_artifact.path).read_bytes()
    )
    adapter = trajectory["tacorank_adapter"]
    assert adapter["isolation_mode"] == "hardened_docker"
    assert adapter["docker_image_digest"] == "sha256:" + "a" * 64
    assert (
        adapter["trae_install_identity"]["verification_mode"]
        == "hashed_direct_url_and_executable"
    )
    assert adapter["trae_install_identity"]["python_dotenv_version"] == "1.2.2"
    assert (
        adapter["trae_runtime_identity"]["verification_mode"]
        == "hashed_runtime_assets_and_dotenv_preflight"
    )
    assert trajectory["fake_cwd"] == str(parts.config.trae_runtime_root)
    args = trajectory["fake_cli_arguments"]
    assert args[-4:] == ["--docker-container-id", container_id, "--docker-keep", "False"]
    assert "--docker-image" not in args
    assert args[args.index("--working-dir") + 1] == str(worktree)
    assert adapter["docker_attach_contract"] == (
        "reviewed-host-project-path-to-mounted-workspace"
    )
    assert trajectory["fake_environment"]["PYTHON_DOTENV_DISABLED"] == "1"
    assert trajectory["fake_environment"]["PATH"] == f"{docker.parent}:/usr/bin:/bin"


def test_production_preflight_executes_read_only_mounted_tool(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    worker, docker = _production_worker(parts)

    worker.preflight()

    calls = [
        json.loads(line)["argv"]
        for line in docker.with_name("docker.log").read_text().splitlines()
    ]
    container_id = "d" * 64
    assert [call[0] for call in calls] == [
        "image",
        "create",
        "start",
        "stop",
        "rm",
        "inspect",
    ]
    assert (
        "type=bind,src=%s,dst=/agent_tools,readonly,bind-propagation=rprivate"
        % (parts.config.trae_runtime_root / "trae_agent" / "dist")
    ) in calls[1]
    entrypoint = calls[1].index("--entrypoint")
    assert calls[1][entrypoint + 1] == "/bin/sh"
    assert calls[1][-2:] == [
        "-c",
        "command -v timeout >/dev/null && /agent_tools/edit_tool --help",
    ]
    assert calls[2] == ["start", "--attach", container_id]


def test_runtime_preflight_requires_cross_platform_docker_bridge(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    worker, _ = _production_worker(parts)
    manager = (
        parts.config.trae_runtime_root
        / "trae_agent"
        / "agent"
        / "docker_manager.py"
    )
    manager.write_text("# unpatched pinned source\n", encoding="utf-8")
    parts.config = replace(
        parts.config,
        trae_runtime_manifest_sha256=hash_trae_runtime_package(
            parts.config.trae_runtime_root
        ),
    )
    worker = _worker(parts)

    with pytest.raises(CodingWorkerError) as failure:
        worker._verify_runtime_root()

    assert failure.value.code == "TRAE_RUNTIME_IDENTITY_MISMATCH"
    assert "cross-platform Docker bridge is missing" in str(failure.value)


def test_local_preflight_does_not_require_provider_credential(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    parts.environment.pop("FAKE_TRAE_SECRET")
    worker, _ = _production_worker(parts)

    worker.preflight_local()

    with pytest.raises(CodingWorkerError) as failure:
        worker.preflight()
    assert failure.value.code == "TRAE_CREDENTIAL_MISSING"


def test_production_preflight_mount_failure_removes_probe(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    worker, docker = _production_worker(parts)
    docker.with_name("docker.behavior").write_text("fail_start", encoding="utf-8")

    with pytest.raises(CodingWorkerError) as failure:
        worker.preflight()

    assert failure.value.code == "TRAE_ISOLATION_SETUP_FAILED"
    assert not docker.with_name("docker.state").exists()
    calls = [
        json.loads(line)["argv"][0]
        for line in docker.with_name("docker.log").read_text().splitlines()
    ]
    assert calls == ["image", "create", "start", "stop", "rm", "inspect"]


def test_isolation_setup_failure_prevents_host_trae_launch(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    worker, docker = _production_worker(parts)
    docker.with_name("docker.behavior").write_text("fail_create", encoding="utf-8")
    marker = Path(parts.environment["FAKE_TRAE_RUN_MARKER"])
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(worker.create_patch(parts.context, parts.spec))
    assert failure.value.code == "TRAE_ISOLATION_SETUP_FAILED"
    assert not marker.exists()
    calls = [json.loads(line)["argv"] for line in docker.with_name("docker.log").read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][0] == "create"


def test_isolation_setup_timeout_removes_any_created_container(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    parts.config = replace(parts.config, docker_cli_timeout_seconds=1)
    docker.with_name("docker.behavior").write_text(
        "create_then_sleep", encoding="utf-8"
    )
    marker = Path(parts.environment["FAKE_TRAE_RUN_MARKER"])
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "TRAE_ISOLATION_SETUP_FAILED"
    assert not marker.exists()
    assert not docker.with_name("docker.state").exists()
    calls = [
        json.loads(line)["argv"][0]
        for line in docker.with_name("docker.log").read_text().splitlines()
    ]
    assert calls == ["create", "stop", "rm", "inspect"]


def test_isolation_is_removed_after_trae_timeout(adapter_parts: SimpleNamespace) -> None:
    parts = adapter_parts
    parts.environment["FAKE_TRAE_BEHAVIOR"] = "timeout"
    parts.context.wall_time_limit_seconds = 1
    worker, docker = _production_worker(parts)
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(worker.create_patch(parts.context, parts.spec))
    assert failure.value.code == "TRAE_TIMEOUT"
    calls = [json.loads(line)["argv"][0] for line in docker.with_name("docker.log").read_text().splitlines()]
    assert calls == ["create", "start", "stop", "rm", "inspect"]


def test_config_execution_surface_and_patch_size_fail_closed(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    config_text = parts.config.config_file.read_text(encoding="utf-8").replace(
        "allow_mcp_servers: []", "allow_mcp_servers: [remote]"
    )
    parts.config.config_file.write_text(config_text, encoding="utf-8")
    parts.config = replace(
        parts.config,
        config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
    )
    with pytest.raises(CodingWorkerError) as failure:
        _worker(parts)
    assert failure.value.code == "TRAE_CONFIG_INVALID"

    parts = adapter_parts
    # Restore the fixture's config bytes after the semantic-mismatch assertion.
    valid_text = config_text.replace("allow_mcp_servers: [remote]", "allow_mcp_servers: []")
    parts.config.config_file.write_text(valid_text, encoding="utf-8")
    parts.config = replace(
        parts.config,
        config_sha256=hashlib.sha256(valid_text.encode()).hexdigest(),
        max_patch_bytes=1,
    )
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "PATCH_TOO_LARGE"


def test_config_rejects_unreviewed_provider_endpoint(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    text = parts.config.config_file.read_text(encoding="utf-8").replace(
        "    api_key: \"\"", "    api_key: \"\"\n    base_url: https://proxy.invalid"
    )
    parts.config.config_file.write_text(text, encoding="utf-8")
    parts.config = replace(
        parts.config,
        config_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    with pytest.raises(CodingWorkerError) as failure:
        _worker(parts)
    assert failure.value.code == "TRAE_CONFIG_INVALID"


def test_production_install_revision_mismatch_fails_before_external_action(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    identity_path = parts.config.trae_install_identity_file
    assert identity_path is not None
    document = json.loads(identity_path.read_text(encoding="utf-8"))
    document["vcs_info"]["commit_id"] = "b" * 40
    identity_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    parts.config = replace(
        parts.config,
        trae_install_identity_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest(),
    )
    marker = Path(parts.environment["FAKE_TRAE_RUN_MARKER"])
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "TRAE_INSTALL_IDENTITY_MISMATCH"
    assert not marker.exists()
    assert not docker.with_name("docker.log").exists()
    assert not parts.worktrees.path_for("run1", "exp1").exists()


def test_production_runtime_rejects_dotenv_before_external_action(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    runtime_root = parts.config.trae_runtime_root
    assert runtime_root is not None
    (runtime_root / ".env").write_text("OPENAI_API_KEY=unreviewed\n", encoding="utf-8")
    marker = Path(parts.environment["FAKE_TRAE_RUN_MARKER"])
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "TRAE_DOTENV_FORBIDDEN"
    assert not marker.exists()
    assert not docker.with_name("docker.log").exists()
    assert not parts.worktrees.path_for("run1", "exp1").exists()


def test_production_runtime_rejects_tampered_docker_tool_assets(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    runtime_root = parts.config.trae_runtime_root
    assert runtime_root is not None
    (runtime_root / "trae_agent" / "dist" / "edit_tool").write_bytes(b"tampered")
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "TRAE_RUNTIME_IDENTITY_MISMATCH"
    assert not docker.with_name("docker.log").exists()
    assert not parts.worktrees.path_for("run1", "exp1").exists()


def test_production_runtime_rejects_unattested_bytecode_cache(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    runtime_root = parts.config.trae_runtime_root
    assert runtime_root is not None
    bytecode = runtime_root / "trae_agent" / "__pycache__" / "cli.cpython-312.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"unreviewed-bytecode")
    with pytest.raises(CodingWorkerError) as failure:
        asyncio.run(_worker(parts).create_patch(parts.context, parts.spec))
    assert failure.value.code == "TRAE_RUNTIME_IDENTITY_MISMATCH"
    assert not docker.with_name("docker.log").exists()


def test_production_config_cannot_be_candidate_editable(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    candidate_config = parts.repository / "trae-agent.yaml"
    candidate_config.write_bytes(parts.config.config_file.read_bytes())
    parts.config = replace(
        parts.config,
        config_file=candidate_config,
        config_sha256=hashlib.sha256(candidate_config.read_bytes()).hexdigest(),
    )
    with pytest.raises(CodingWorkerError) as failure:
        _worker(parts)
    assert failure.value.code == "TRAE_CONFIG_INVALID"
    assert not docker.with_name("docker.log").exists()


def test_production_rejects_code_loader_environment_names(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    parts.config = replace(
        parts.config,
        approved_environment_names=parts.config.approved_environment_names
        + ("PYTHONPATH", "DOCKER_HOST", "BASH_ENV"),
    )
    parts.environment.update(
        {
            "PYTHONPATH": "/unreviewed/code",
            "DOCKER_HOST": "tcp://unreviewed.invalid",
            "BASH_ENV": "/unreviewed/bashrc",
        }
    )
    with pytest.raises(CodingWorkerError) as failure:
        _worker(parts)
    assert failure.value.code == "TRAE_CONFIG_INVALID"
    assert not docker.with_name("docker.log").exists()


def test_production_rejects_docker_path_through_symlinked_parent(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    linked_parent = docker.parent / "docker-parent-link"
    linked_parent.symlink_to(docker.parent, target_is_directory=True)
    parts.config = replace(
        parts.config,
        docker_executable=linked_parent / "docker",
    )
    with pytest.raises(CodingWorkerError) as failure:
        _worker(parts)
    assert failure.value.code == "TRAE_CONFIG_INVALID"
    assert not docker.with_name("docker.log").exists()


@pytest.mark.skipif(os.name != "nt", reason="Docker Desktop uses docker.exe on Windows")
def test_production_accepts_windows_docker_executable_name(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _, docker = _production_worker(parts)
    docker_exe = docker.with_name("docker.exe")
    docker_exe.write_bytes(docker.read_bytes())
    docker_exe.chmod(0o755)
    parts.config = replace(parts.config, docker_executable=docker_exe)

    # Construction performs the complete production configuration validation;
    # no Docker daemon is needed for this boundary test.
    _worker(parts)


def test_production_sanitized_path_uses_host_separator(
    adapter_parts: SimpleNamespace,
) -> None:
    parts = adapter_parts
    _production_worker(parts)
    worker = _worker(parts)

    path_entries = worker._sanitized_environment()["PATH"].split(os.pathsep)
    assert path_entries[0] == str(parts.config.docker_executable.parent)
    if os.name == "nt":
        assert os.pathsep == ";"
        assert path_entries[1:] == os.environ.get("PATH", "").split(os.pathsep)
    else:
        assert path_entries[1:] == ["/usr/bin", "/bin"]


def test_sanitized_environment_forces_utf8_console(
    adapter_parts: SimpleNamespace,
) -> None:
    environment = _worker(adapter_parts)._sanitized_environment()

    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    result = subprocess.run(
        (sys.executable, "-c", "print('\\u2705')"),
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.stdout.decode("utf-8").strip() == "\u2705"
