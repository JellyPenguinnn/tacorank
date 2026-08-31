from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from solution.candidate import run as run_candidate
from benchmarks.kuairand_pure.pipeline import check_submission
from tacorank import deployment as deployment_module
from tacorank.config import ContractError
from tacorank.orchestrator.live import (
    _verify_data_manifest,
    _verify_executable_baseline_parity,
)


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


def test_download_data_uses_certifi_and_extracts_pinned_archive(
    tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "repository").resolve()
    (root / "KuaiRand-Pure").mkdir(parents=True)
    bundle_bytes = io.BytesIO()
    with tarfile.open(fileobj=bundle_bytes, mode="w:gz") as bundle:
        for name in deployment_module.RAW_REQUIRED:
            content = (name + "\n").encode("utf-8")
            member = tarfile.TarInfo("KuaiRand-Pure/data/" + name)
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
    archive = bundle_bytes.getvalue()
    monkeypatch.setattr(
        deployment_module,
        "DATA_ARCHIVE_MD5",
        hashlib.md5(archive, usedforsecurity=False).hexdigest(),
    )
    tls_context = object()
    observed = {}
    monkeypatch.setattr(deployment_module.certifi, "where", lambda: "/trusted/ca.pem")

    def create_context(*, cafile):
        observed["cafile"] = cafile
        return tls_context

    def urlopen(request, *, timeout, context):
        observed.update(url=request.full_url, timeout=timeout, context=context)
        return io.BytesIO(archive)

    monkeypatch.setattr(deployment_module.ssl, "create_default_context", create_context)
    monkeypatch.setattr(deployment_module.urllib.request, "urlopen", urlopen)

    data = root / "KuaiRand-Pure" / "data"
    deployment_module._download_data(root, data)

    assert observed == {
        "cafile": "/trusted/ca.pem",
        "url": deployment_module.DATA_URL,
        "timeout": 120,
        "context": tls_context,
    }
    for name in deployment_module.RAW_REQUIRED:
        assert (data / name).read_text(encoding="utf-8") == name + "\n"


def test_official_split_loader_does_not_dirty_submodule_with_bytecode(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repository").resolve()
    starter = root / "kuairand-starter-kit"
    data = root / "KuaiRand-Pure" / "data"
    starter.mkdir(parents=True)
    data.mkdir(parents=True)
    (starter / "data.py").write_text(
        "def load(path):\n"
        "    return {'train': [path], 'valid': [path], 'test': [path]}\n",
        encoding="utf-8",
    )
    previous = sys.dont_write_bytecode

    result = deployment_module._load_official_splits(root, data)

    assert result["train"] == [str(data)]
    assert sys.dont_write_bytecode is previous
    assert not (starter / "__pycache__").exists()


def test_prepare_data_builds_separate_unlabelled_views_and_attested_labels(
    tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "repository").resolve()
    deployment = root / ".tacorank" / "deployment"
    data = root / "KuaiRand-Pure" / "data"
    (root / "kuairand-starter-kit").mkdir(parents=True)
    (root / "contract").mkdir(parents=True)
    (root / "contract" / "COMPETITION.md").write_text(
        "# Test contract\n", encoding="utf-8"
    )
    (root / "solution").mkdir()
    (root / "solution" / "candidate.py").write_text(
        (Path(__file__).parents[2] / "solution" / "candidate.py").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
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
    monkeypatch.setattr(
        deployment_module,
        "_load_extra_log_columns",
        lambda data: {
            "train": [("1000", "900", "0", "10")] * len(train),
            "valid": [("2000", "1000", "1", "20")] * len(valid),
            "test": [("3000", "1100", "0", "30")] * len(test),
        },
    )

    def fake_run(args, *, cwd, label, capture_output=False):
        del cwd, label, capture_output
        assert args[1:3] == ("-B", "submit.py")
        baseline = Path(args[3])
        split = args[args.index("--split") + 1]
        source_rows = valid if split == "valid" else test
        with baseline.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("row_id", "user_id", "video_id", "score"))
            for row_id, row in enumerate(source_rows):
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
    submission_rows = result["contract_root"] / "submission_rows.csv"
    submission_header = submission_rows.read_text(encoding="utf-8").splitlines()[0]
    assert submission_header == "row_id,user_id,video_id"
    assert "label" not in submission_header
    assert (result["contract_root"] / "COMPETITION.md").read_text(
        encoding="utf-8"
    ) == "# Test contract\n"
    check_submission(
        SimpleNamespace(
            prediction_path=result["baseline_final_prediction_csv"],
            contract_root=result["contract_root"],
        )
    )
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    attested = {record["path"] for record in manifest["files"]}
    assert (
        result["input_roots"]["candidate_full"] / "score.csv"
    ).relative_to(root).as_posix() in attested
    assert result["population_csvs"]["full"].relative_to(root).as_posix() in attested
    assert submission_rows.relative_to(root).as_posix() in attested

    config = SimpleNamespace(
        repository_root=root,
        data_manifest_sha256=hashlib.sha256(
            result["manifest_path"].read_bytes()
        ).hexdigest(),
        research_capabilities=["baseline_parity"],
    )
    live = SimpleNamespace(
        data_manifest_path=result["manifest_path"],
        input_roots=result["input_roots"],
        population_csvs=result["population_csvs"],
        baseline_prediction_csvs=result["baseline_prediction_csvs"],
        baseline_final_prediction_csv=result["baseline_final_prediction_csv"],
        baseline_parity_receipt_path=result["baseline_parity_receipt_path"],
        candidate_entrypoint="solution.candidate:run",
    )
    _verify_data_manifest(config, live)
    _verify_executable_baseline_parity(config, live)

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
    baseline = (
        "row_id,user_id,video_id,score\n"
        "0,u1,v1,0.75\n"
        "1,u1,unknown,0.25\n"
    )
    (input_root / "fm_baseline_predictions.csv").write_text(
        baseline, encoding="utf-8"
    )
    (input_root / "fm_baseline_predictions.sha256").write_text(
        hashlib.sha256(baseline.encode("utf-8")).hexdigest() + "\n",
        encoding="ascii",
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
    assert output.read_text(encoding="utf-8") == baseline


def test_validate_trae_tools_accepts_canonical_bundle(tmp_path: Path) -> None:
    tools = (tmp_path / "tools").resolve()
    internal = tools / "_internal"
    internal.mkdir(parents=True)
    (internal / "libpython.so").write_bytes(b"runtime")
    for name in ("edit_tool", "json_edit_tool"):
        executable = tools / name
        executable.write_bytes(b"tool")
        executable.chmod(0o755)

    deployment_module._validate_trae_tools(tools)


def test_validate_trae_tools_rejects_symlinked_asset(tmp_path: Path) -> None:
    tools = (tmp_path / "tools").resolve()
    internal = tools / "_internal"
    internal.mkdir(parents=True)
    (internal / "libpython.so").write_bytes(b"runtime")
    for name in ("edit_tool", "json_edit_tool"):
        executable = tools / name
        executable.write_bytes(b"tool")
        executable.chmod(0o755)
    (internal / "alias").symlink_to(internal / "libpython.so")

    with pytest.raises(deployment_module.DeploymentError, match="cannot contain symlinks"):
        deployment_module._validate_trae_tools(tools)


def test_patch_trae_read_only_attach_reuses_pre_mounted_tools(
    tmp_path: Path, monkeypatch
) -> None:
    site_packages = tmp_path / "site-packages"
    docker_manager = site_packages / "trae_agent/agent/docker_manager.py"
    docker_manager.parent.mkdir(parents=True)
    docker_manager.write_text(
        '''class DockerManager:
    CONTAINER_TOOLS_PATH = "/agent_tools"

    def _copy_tools_to_container(self):
        print(
            f"Copying tools from '{self.tools_dir}' to container path '{self.CONTAINER_TOOLS_PATH}'..."
        )
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        deployment_module, "_trae_site_packages", lambda runtime: site_packages
    )

    deployment_module._patch_trae_read_only_attach(tmp_path / "runtime")

    patched = docker_manager.read_text(encoding="utf-8")
    assert deployment_module.TRAE_READ_ONLY_TOOL_MARKER in patched
    assert "test -x /agent_tools/edit_tool" in patched
    assert "test -x /agent_tools/json_edit_tool" in patched
    compile(patched, str(docker_manager), "exec")


def test_patch_trae_docker_exec_is_cross_platform_and_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    site_packages = tmp_path / "site-packages"
    docker_manager = site_packages / "trae_agent/agent/docker_manager.py"
    docker_manager.parent.mkdir(parents=True)
    docker_manager.write_text(
        '''class DockerException(Exception):
    pass

class DockerManager:
    def start(self):
            self._copy_tools_to_container()
            # if self.interactive:
            self._start_persistent_shell()

    def execute(self, command: str, timeout: int = 300):
        # if self.interactive:
        return self._execute_interactive(command, timeout)
        # else:
        #     return self._execute_stateless(command)

    def _start_persistent_shell(self):
        pass
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        deployment_module, "_trae_site_packages", lambda runtime: site_packages
    )

    deployment_module._patch_trae_cross_platform_docker_exec(
        tmp_path / "runtime"
    )

    patched = docker_manager.read_text(encoding="utf-8")
    assert deployment_module.TRAE_STATELESS_DOCKER_MARKER in patched
    assert "return self._execute_stateless(command, timeout)" in patched
    assert '"timeout", "--signal=KILL"' in patched
    assert "workdir=self.container_workspace" in patched
    assert "self._start_persistent_shell()" not in patched.split(
        "def _start_persistent_shell", 1
    )[0]
    compile(patched, str(docker_manager), "exec")
    namespace = {}
    exec(compile(patched, str(docker_manager), "exec"), namespace)
    calls = []

    class Container:
        def exec_run(self, argv, *, workdir):
            calls.append((argv, workdir))
            return SimpleNamespace(exit_code=0, output=b"portable output\n")

    manager = namespace["DockerManager"]()
    manager.container = Container()
    manager.container_workspace = "/workspace"
    code, output = manager.execute("printf portable", timeout=7)

    assert code == 0
    assert output == "portable output"
    assert calls == [
        (
            [
                "timeout",
                "--signal=KILL",
                "7s",
                "/bin/bash",
                "-lc",
                "printf portable",
            ],
            "/workspace",
        )
    ]


def test_patch_trae_deepseek_reasoning_is_explicit_and_continuous(
    tmp_path: Path, monkeypatch
) -> None:
    site_packages = tmp_path / "site-packages"
    client = site_packages / "trae_agent/utils/llm_clients/openai_client.py"
    client.parent.mkdir(parents=True)
    client.write_text(
        '''import json

def EasyInputMessageParam(**values):
    return values

class Client:
    def __init__(self):
        self.message_history = []

    def request(self, model_config):
        return self.client.responses.create(
            max_output_tokens=model_config.max_tokens,
        )

    def record(self, response):
        content = ""
        tool_calls = []
        for output_block in response.output:
            if output_block.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        call_id=output_block.call_id,
                        name=output_block.name,
                        arguments=json.loads(output_block.arguments)
                        if output_block.arguments
                        else {},
                        id=output_block.id,
                    )
                )
                tool_call_param = ResponseFunctionToolCallParam(
                    arguments=output_block.arguments,
                    call_id=output_block.call_id,
                    name=output_block.name,
                    type="function_call",
                )
            elif output_block.type == "message":
                content = "".join(
                    content_block.text
                    for content_block in output_block.content
                    if content_block.type == "output_text"
                )

        if content != "":
            self.message_history.append(
                EasyInputMessageParam(content=content, role="assistant", type="message")
            )
        self.last_content = content
        self.last_tool_calls = tool_calls
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        deployment_module, "_trae_site_packages", lambda runtime: site_packages
    )

    deployment_module._patch_trae_deepseek_reasoning(tmp_path / "runtime")

    patched = client.read_text(encoding="utf-8")
    assert deployment_module.TRAE_DEEPSEEK_REASONING_MARKER in patched
    assert deployment_module.TRAE_DEEPSEEK_TOOL_JSON_MARKER in patched
    assert 'reasoning={"effort": "high"}' in patched
    assert 'output_block.type == "reasoning"' in patched
    assert "self.message_history.append(reasoning_item)" in patched
    assert "except (json.JSONDecodeError, TypeError, RecursionError)" in patched
    assert "if not isinstance(tool_arguments, dict)" in patched
    assert "Retry the same operation with one smaller valid JSON tool call" in patched
    assert "arguments=json.dumps(tool_arguments" in patched
    assert "content += message_content" in patched
    assert patched.index("content += message_content") > patched.index(
        'output_block.type == "function_call"'
    )
    assert 'if content != "":' not in patched
    compile(patched, str(client), "exec")
    namespace = {}
    exec(compile(patched, str(client), "exec"), namespace)
    instance = namespace["Client"]()
    instance.record(
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    arguments='{"path":"unterminated',
                    call_id="call_1",
                    name="str_replace_based_edit_tool",
                    id="fc_1",
                )
            ]
        )
    )
    assert instance.last_tool_calls == []
    assert "malformed or truncated" in instance.last_content
    assert instance.message_history[-1]["role"] == "assistant"


def test_patch_trae_docker_edit_tool_filters_and_quotes_arguments(
    tmp_path: Path, monkeypatch
) -> None:
    site_packages = tmp_path / "site-packages"
    executor = site_packages / "trae_agent/tools/docker_tool_executor.py"
    executor.parent.mkdir(parents=True)
    executor.write_text(
        '''import json
import os

class Executor:
    def translate(self, relative_path):
            container_path = os.path.join(self._container_workspace_dir, relative_path)
            return os.path.normpath(container_path)

    def run(self, processed_args, sub_command):
                executable_path = f"{self._docker_manager.CONTAINER_TOOLS_PATH}/edit_tool"
                cmd_parts = [executable_path, sub_command]

                for key, value in processed_args.items():
                    if key == "command" or value is None:
                        continue
                    if isinstance(value, list):
                        str_value = " ".join(map(str, value))
                        cmd_parts.append(f"--{key} {str_value}")
                    else:
                        cmd_parts.append(f"--{key} '{str(value)}'")

                command_to_run = " ".join(cmd_parts)
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        deployment_module, "_trae_site_packages", lambda runtime: site_packages
    )

    deployment_module._patch_trae_docker_edit_tool(tmp_path / "runtime")

    patched = executor.read_text(encoding="utf-8")
    assert deployment_module.TRAE_DOCKER_EDIT_TOOL_MARKER in patched
    assert '"view": ("path", "view_range")' in patched
    assert "for key in command_arguments" in patched
    assert "shlex.join(cmd_parts)" in patched
    assert "posixpath.join(" in patched
    assert 'relative_path.replace(os.sep, "/")' in patched
    compile(patched, str(executor), "exec")
    namespace = {}
    exec(compile(patched, str(executor), "exec"), namespace)
    instance = namespace["Executor"]()
    instance._container_workspace_dir = "/workspace"
    assert instance.translate(os.path.join("solution", "candidate.py")) == (
        "/workspace/solution/candidate.py"
    )


def test_generated_trae_yaml_uses_v4_flash() -> None:
    document = deployment_module._trae_yaml()

    assert "model: deepseek-v4-flash" in document
    assert "max_steps: 64" in document
    assert "deepseek-v4-pro" not in document


def test_generated_live_deployment_caps_repair_at_twenty_steps(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime"
    site_packages = runtime / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    identity = {}
    for key, name in (
        ("executable", "trae.exe"),
        ("direct_url", "direct_url.json"),
        ("dotenv_metadata", "METADATA"),
    ):
        path = runtime / name
        path.write_text(key, encoding="utf-8")
        identity[key] = path
    identity["site_packages"] = site_packages
    trae_yaml = tmp_path / "trae.yaml"
    trae_yaml.write_text("models: {}\n", encoding="utf-8")
    docker = tmp_path / "docker.exe"
    docker.write_text("docker", encoding="utf-8")
    monkeypatch.setattr(
        deployment_module,
        "hash_trae_runtime_package",
        lambda path: "a" * 64,
    )

    payload = deployment_module._trae_payload(
        runtime=runtime,
        runtime_identity=identity,
        trae_yaml=trae_yaml,
        docker=docker,
        docker_host="npipe:////./pipe/dockerDesktopLinuxEngine",
        image="sha256:" + "b" * 64,
    )

    assert payload["repair_step_limit"] == 20


def test_trae_responses_sdk_is_exactly_pinned() -> None:
    requirements = (
        Path(__file__).parents[2] / "requirements-trae.txt"
    ).read_text(encoding="utf-8")

    assert "openai==3.6.0" in requirements.splitlines()


def test_runtime_dockerfile_installs_reviewed_runtime_requirements() -> None:
    dockerfile = (
        Path(__file__).parents[2] / "docker" / "runtime.Dockerfile"
    ).read_text(encoding="utf-8")

    assert "--requirement requirements.txt" in dockerfile
    assert "python -m pip install --no-cache-dir --no-deps ." in dockerfile
    assert dockerfile.index("--requirement requirements.txt") < dockerfile.index(
        "python -m pip install --no-cache-dir --no-deps ."
    )


def test_runtime_import_inventory_is_bound_to_built_image(
    tmp_path: Path, monkeypatch
) -> None:
    observed = {}

    def output(args, *, cwd, label):
        observed.update(args=tuple(args), cwd=cwd, label=label)
        roots = sorted(
            {
                "benchmarks",
                "certifi",
                "numpy",
                "pandas",
                "pydantic",
                "tacorank",
                "yaml",
                "_sysconfigdata__linux_aarch64-linux-gnu",
            }
        )
        return json.dumps(roots)

    monkeypatch.setattr(deployment_module, "_run_output", output)
    roots = deployment_module._runtime_image_import_roots(
        tmp_path, tmp_path / "docker", "sha256:" + "a" * 64
    )

    assert "solution" in roots
    assert "_sysconfigdata__linux_aarch64-linux-gnu" not in roots
    assert set(deployment_module.RUNTIME_REQUIRED_IMPORTS) != set(roots)
    assert observed["label"] == "Docker runtime import verification"
    assert observed["args"][1:9] == (
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
    )
    script = observed["args"][-1]
    for name in deployment_module.RUNTIME_REQUIRED_IMPORTS:
        assert name in script


def test_runtime_import_inventory_rejects_malformed_output(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        deployment_module,
        "_run_output",
        lambda *args, **kwargs: '{"not":"a list"}',
    )

    with pytest.raises(deployment_module.DeploymentError, match="inventory is malformed"):
        deployment_module._runtime_image_import_roots(
            tmp_path, tmp_path / "docker", "sha256:" + "b" * 64
        )
