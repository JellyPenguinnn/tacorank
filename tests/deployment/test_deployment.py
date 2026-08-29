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


def test_patch_trae_deepseek_reasoning_is_explicit_and_continuous(
    tmp_path: Path, monkeypatch
) -> None:
    site_packages = tmp_path / "site-packages"
    client = site_packages / "trae_agent/utils/llm_clients/openai_client.py"
    client.parent.mkdir(parents=True)
    client.write_text(
        '''class Client:
    def request(self, model_config):
        return self.client.responses.create(
            max_output_tokens=model_config.max_tokens,
        )

    def record(self, response):
        for output_block in response.output:
            if output_block.type == "function_call":
                pass
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
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        deployment_module, "_trae_site_packages", lambda runtime: site_packages
    )

    deployment_module._patch_trae_deepseek_reasoning(tmp_path / "runtime")

    patched = client.read_text(encoding="utf-8")
    assert deployment_module.TRAE_DEEPSEEK_REASONING_MARKER in patched
    assert 'reasoning={"effort": "high"}' in patched
    assert 'output_block.type == "reasoning"' in patched
    assert "self.message_history.append(reasoning_item)" in patched
    assert "content += message_content" in patched
    assert patched.index("content += message_content") > patched.index(
        'output_block.type == "function_call"'
    )
    assert 'if content != "":' not in patched
    compile(patched, str(client), "exec")


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
    compile(patched, str(executor), "exec")


def test_generated_trae_yaml_uses_v4_flash() -> None:
    document = deployment_module._trae_yaml()

    assert "model: deepseek-v4-flash" in document
    assert "deepseek-v4-pro" not in document


def test_trae_responses_sdk_is_exactly_pinned() -> None:
    requirements = (
        Path(__file__).parents[2] / "requirements-trae.txt"
    ).read_text(encoding="utf-8")

    assert "openai==3.6.0" in requirements.splitlines()
