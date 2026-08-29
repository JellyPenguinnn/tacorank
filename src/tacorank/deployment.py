"""Fresh-clone production deployment preparation for TacoRank."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .coding import hash_trae_runtime_package
from .evaluation.proxy import split_validation_indices


TRAE_SOURCE_REVISION = "e839e559ac61bdd0e057c375dd1dee391fee797d"
TRAE_READ_ONLY_TOOL_MARKER = "TacoRank: use manifest-verified pre-mounted tools"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DATA_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
DATA_ARCHIVE_MD5 = "0820331067a3784d9691136f772b35a7"
RAW_REQUIRED = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "video_features_basic_pure.csv",
)


class DeploymentError(RuntimeError):
    """A local production prerequisite or generated identity is invalid."""


def setup_live_deployment(
    *,
    repository_root: Path,
    deployment_directory: Path,
    runtime_directory: Path,
    data_directory: Path,
    python312: Path,
    docker_executable: Path,
    run_id: str,
    download_data: bool,
) -> Mapping[str, Any]:
    """Build an exact production deployment and return its generated paths."""

    root = Path(repository_root).resolve(strict=True)
    deployment = _new_directory_inside(root, deployment_directory)
    runtime = _new_external_directory(root, runtime_directory)
    python = _regular_executable(python312, "Python 3.12")
    docker = _regular_executable(docker_executable, "Docker")
    docker_host = _discover_docker_host(docker, root)
    _require_python312(python)
    _require_clean_tracked_checkout(root)
    _run(
        ("git", "submodule", "update", "--init", "--recursive"),
        cwd=root,
        label="Git submodule initialization",
    )

    data = Path(data_directory)
    if not data.is_absolute():
        data = root / data
    if download_data and not _has_raw_data(data):
        _download_data(root, data)
    _require_raw_data(data, root)

    runtime.mkdir(parents=True, exist_ok=False)
    deployment.mkdir(parents=True, exist_ok=False)
    _install_trae(python, runtime, root)
    _patch_trae_read_only_attach(runtime)
    image, image_environment_sha256 = _build_runtime_image(root, docker)
    _install_trae_tools(runtime, docker, image, root)
    runtime_identity = _trae_identity(runtime)
    generated_data = _prepare_data(root, deployment, data)

    trae_yaml = runtime / "trae-agent.yaml"
    _write_text_exclusive(trae_yaml, _trae_yaml())
    trae_yaml.chmod(0o600)
    live_path = deployment / "live-adapters.json"
    run_path = deployment / "run-config.json"
    manifest_path = generated_data["manifest_path"]
    baseline_commit = _git_text(root, ("rev-parse", "--verify", "HEAD^{commit}"))
    evaluator_hash = _sha256_file(root / "kuairand-starter-kit" / "evaluate.py")

    live_payload = {
        "schema_version": "1.0",
        "worktree_root": str(
            (root.parent / ".tacorank-worktrees" / root.name).resolve()
        ),
        "required_submodules": ["kuairand-starter-kit"],
        "trae": {
            "command_prefix": [str(runtime_identity["executable"])],
            "trae_version": "0.1.0",
            "provider": "openai",
            "provider_base_url": DEEPSEEK_BASE_URL,
            "model_id": DEEPSEEK_MODEL,
            "config_file": str(trae_yaml),
            "config_sha256": _sha256_file(trae_yaml),
            "max_steps_cap": 20,
            "max_token_cap": 4096,
            "max_wall_time_seconds_cap": 900,
            "repair_step_limit": 12,
            "repair_token_limit": 2048,
            "repair_wall_time_limit_seconds": 600,
            "repair_allowed_command_ids": ["candidate_smoke"],
            "approved_environment_names": ["DEEPSEEK_API_KEY"],
            "credential_environment_names": ["DEEPSEEK_API_KEY"],
            "credential_environment_aliases": [
                ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
            ],
            "trae_source_revision": TRAE_SOURCE_REVISION,
            "trae_install_root": str(runtime),
            "trae_install_identity_file": str(runtime_identity["direct_url"]),
            "trae_install_identity_sha256": _sha256_file(
                runtime_identity["direct_url"]
            ),
            "trae_executable_sha256": _sha256_file(runtime_identity["executable"]),
            "trae_runtime_root": str(runtime_identity["site_packages"]),
            "trae_runtime_manifest_sha256": hash_trae_runtime_package(
                runtime_identity["site_packages"]
            ),
            "python_dotenv_metadata_file": str(runtime_identity["dotenv_metadata"]),
            "python_dotenv_metadata_sha256": _sha256_file(
                runtime_identity["dotenv_metadata"]
            ),
            "docker_image": image,
            "docker_executable": str(docker),
            "docker_host": docker_host,
        },
        "contract_root": str((root / "contract").resolve(strict=True)),
        "input_roots": {
            key: str(value) for key, value in generated_data["input_roots"].items()
        },
        "baseline_entrypoint": "benchmarks.kuairand_pure.pipeline:run_baseline",
        "candidate_entrypoint": "solution.candidate:run",
        "submission_check_entrypoint": (
            "benchmarks.kuairand_pure.pipeline:check_submission"
        ),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "container_python_executable": "/usr/local/bin/python3",
        "docker_executable": str(docker),
        "docker_host": docker_host,
        "docker_image": image,
        "docker_image_environment_sha256": image_environment_sha256,
        "docker_cpu_count": 2.0,
        "docker_tmpfs_size_mb": 256,
        "output_quota_max_bytes": 2 * 1024 * 1024 * 1024,
        "data_manifest_path": str(manifest_path),
        "population_csvs": {
            key: str(value)
            for key, value in generated_data["population_csvs"].items()
        },
        "baseline_prediction_csvs": {
            key: str(value)
            for key, value in generated_data["baseline_prediction_csvs"].items()
        },
        "candidate_allowed_columns": [
            "date",
            "user_id",
            "video_id",
            "author_id",
            "tab",
            "duration_ms",
            "long_view",
        ],
        "protected_columns": ["label"],
        "hidden_path_tokens": ["hidden_labels", "final_labels", "test_labels"],
        "future_column_patterns": ["(?:^|_)future(?:_|$)"],
        "allowed_import_roots": None,
        "allowed_capability_imports": [],
        "allowed_dependency_changes": [],
    }
    _write_json_exclusive(live_path, live_payload)
    run_payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "repository_root": str(root),
        "contract_path": "contract/COMPETITION.md",
        "protected_paths_path": "PROTECTED_PATHS.md",
        "artifact_roots": ["artifacts", "runs"],
        "command_ids": ["candidate_smoke", "candidate_proxy", "candidate_full"],
        "metric_names": ["GAUC", "nDCG@5"],
        "primary_metric_name": "primary",
        "data_manifest_sha256": _sha256_file(manifest_path),
        "evaluator_sha256": evaluator_hash,
        "baseline_commit_sha": baseline_commit,
        "max_experiments": 50,
        "wall_time_limit_seconds": 21600,
        "convergence_epsilon": 0.002,
        "convergence_patience": 3,
        "max_repairs_per_experiment": 2,
        "allowed_runtime_adjustments": {},
        "timeout_profiles": {"standard": 600, "extended": 900},
        "max_confirmation_attempts": 2,
        "seed_schedule": [11, 22, 33, 44, 55],
        "context_token_limit": 6000,
        "adapter_mode": "live",
        "live_adapter_config_sha256": _sha256_file(live_path),
        "editable_roots": ["solution"],
        "target_interface_excerpts": {
            "candidate": (
                "def run(invocation: PipelineInvocation) -> None; read only "
                "invocation.input_root and write exactly invocation.output_path as "
                "row_id,user_id,video_id,score CSV; use invocation.fidelity and "
                "invocation.seed; return None"
            )
        },
        "coding_step_limit": 20,
        "coding_token_limit": 4096,
        "coding_wall_time_limit_seconds": 900,
        "research_provider": "deepseek",
        "deepseek_model": DEEPSEEK_MODEL,
        "deepseek_base_url": DEEPSEEK_BASE_URL,
        "deepseek_api_key_env": "DEEPSEEK_API_KEY",
        "deepseek_timeout_seconds": 120,
        "deepseek_max_output_tokens": 8192,
        "deepseek_thinking_enabled": True,
        "deepseek_reasoning_effort": "high",
    }
    _write_json_exclusive(run_path, run_payload)
    return {
        "run_config": str(run_path),
        "live_config": str(live_path),
        "data_manifest": str(manifest_path),
        "runtime": str(runtime),
        "docker_image": image,
    }


def _prepare_data(root: Path, deployment: Path, data: Path) -> Mapping[str, Any]:
    views = deployment / "views"
    protected = deployment / "protected"
    views.mkdir(mode=0o700)
    protected.mkdir(mode=0o700)
    splits = _load_official_splits(root, data)
    train = list(splits["train"])
    valid = list(splits["valid"])
    test = list(splits["test"])
    if not train or not valid or not test:
        raise DeploymentError("official KuaiRand split contains an empty population")

    _, proxy_indices = split_validation_indices([row[1] for row in valid])
    if not proxy_indices:
        raise DeploymentError("internal proxy split is empty")
    smoke_indices = proxy_indices[: min(10_000, len(proxy_indices))]
    populations = protected / "populations"
    baselines = protected / "baselines"
    populations.mkdir(mode=0o700)
    baselines.mkdir(mode=0o700)
    population_csvs = {
        "smoke": populations / "smoke.csv",
        "proxy": populations / "proxy.csv",
        "full": populations / "full.csv",
    }
    _write_population(population_csvs["smoke"], (valid[index] for index in smoke_indices))
    _write_population(population_csvs["proxy"], (valid[index] for index in proxy_indices))
    _write_population(population_csvs["full"], valid)

    official_baseline = protected / "official-fm-valid.csv"
    _run(
        (
            sys.executable,
            "submit.py",
            str(official_baseline),
            "--data_dir",
            str(data),
            "--split",
            "valid",
            "--make",
        ),
        cwd=root / "kuairand-starter-kit",
        label="official FM baseline generation",
    )
    baseline_rows = _read_prediction_rows(official_baseline, len(valid))
    baseline_prediction_csvs = {
        "proxy": baselines / "proxy.csv",
        "full": baselines / "full.csv",
    }
    _write_prediction_subset(
        baseline_prediction_csvs["proxy"], baseline_rows, proxy_indices
    )
    _copy_exclusive(official_baseline, baseline_prediction_csvs["full"])

    common = views / "common"
    common.mkdir(mode=0o700)
    train_path = common / "train.csv"
    _write_train(train_path, train)
    command_directories = {
        "candidate_smoke": views / "candidate-smoke",
        "candidate_proxy": views / "candidate-proxy",
        "candidate_full": views / "candidate-full",
        "candidate_final_infer": views / "candidate-final",
    }
    index_sets: Dict[str, Tuple[Sequence[Sequence[Any]], Sequence[int] | None]] = {
        "candidate_smoke": (valid, smoke_indices),
        "candidate_proxy": (valid, proxy_indices),
        "candidate_full": (valid, None),
        "candidate_final_infer": (test, None),
    }
    for command_id, directory in command_directories.items():
        directory.mkdir(mode=0o700)
        os.link(train_path, directory / "train.csv")
        rows, indices = index_sets[command_id]
        selected: Iterable[Sequence[Any]] = rows
        if indices is not None:
            selected = (rows[index] for index in indices)
        _write_score(directory / "score.csv", selected)

    baseline_directory = views / "baseline-full"
    baseline_directory.mkdir(mode=0o700)
    baseline_view = baseline_directory / "baseline_predictions.csv"
    _copy_exclusive(baseline_prediction_csvs["full"], baseline_view)
    _write_text_exclusive(
        baseline_directory / "baseline_predictions.sha256",
        _sha256_file(baseline_view) + "\n",
    )
    input_roots = {
        "baseline_full": baseline_directory,
        **command_directories,
        "clean_reproduce": command_directories["candidate_full"],
    }

    manifest_path = deployment / "data-manifest.json"
    manifest_files = sorted(
        {
            *[path for path in data.rglob("*") if path.is_file()],
            *[
                path
                for path in deployment.rglob("*")
                if path.is_file() and path != manifest_path
            ],
        },
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in manifest_files
    ]
    _write_json_exclusive(
        manifest_path, {"schema_version": "1.0", "files": records}
    )
    return {
        "manifest_path": manifest_path,
        "input_roots": input_roots,
        "population_csvs": population_csvs,
        "baseline_prediction_csvs": baseline_prediction_csvs,
    }


def _load_official_splits(root: Path, data: Path) -> Mapping[str, Any]:
    source = root / "kuairand-starter-kit" / "data.py"
    specification = importlib.util.spec_from_file_location(
        "tacorank_reviewed_kuairand_data", source
    )
    if specification is None or specification.loader is None:
        raise DeploymentError("official KuaiRand data loader could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.load(str(data))


def _write_train(path: Path, rows: Sequence[Sequence[Any]]) -> None:
    with _exclusive_csv(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("date", "user_id", "video_id", "author_id", "tab", "duration_ms", "long_view")
        )
        for row in rows:
            writer.writerow(row)


def _write_score(path: Path, rows: Iterable[Sequence[Any]]) -> None:
    with _exclusive_csv(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("row_id", "date", "user_id", "video_id", "author_id", "tab", "duration_ms")
        )
        for row_id, row in enumerate(rows):
            writer.writerow((row_id, *row[:6]))


def _write_population(path: Path, rows: Iterable[Sequence[Any]]) -> None:
    with _exclusive_csv(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "label"))
        for row_id, row in enumerate(rows):
            writer.writerow((row_id, row[1], row[2], row[6]))


def _read_prediction_rows(path: Path, expected_count: int) -> Sequence[Tuple[str, str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != ["row_id", "user_id", "video_id", "score"]:
            raise DeploymentError("official baseline produced an invalid header")
        for expected, row in enumerate(reader):
            if int(row["row_id"]) != expected:
                raise DeploymentError("official baseline row order is invalid")
            rows.append((row["user_id"], row["video_id"], row["score"]))
    if len(rows) != expected_count:
        raise DeploymentError("official baseline row count is invalid")
    return rows


def _write_prediction_subset(
    path: Path, rows: Sequence[Tuple[str, str, str]], indices: Sequence[int]
) -> None:
    with _exclusive_csv(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, source_index in enumerate(indices):
            user_id, video_id, score = rows[source_index]
            writer.writerow((row_id, user_id, video_id, score))


def _trae_yaml() -> str:
    return """agents:
  trae_agent:
    enable_lakeview: false
    model: tacorank_coder
    max_steps: 20
    tools:
      - str_replace_based_edit_tool
      - task_done

allow_mcp_servers: []
mcp_servers: {}

model_providers:
  openai:
    provider: openai
    api_key: ""
    base_url: https://api.deepseek.com

models:
  tacorank_coder:
    model_provider: openai
    model: deepseek-v4-flash
    max_tokens: 4096
    temperature: 0
    top_p: 1
    top_k: 0
    max_retries: 3
    parallel_tool_calls: false
"""


def _install_trae(python: Path, runtime: Path, root: Path) -> None:
    _run((str(python), "-m", "venv", str(runtime)), cwd=root, label="Trae virtualenv")
    venv_python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        (
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "-r",
            str(root / "requirements-trae.txt"),
        ),
        cwd=root,
        label="pinned Trae installation",
    )


def _patch_trae_read_only_attach(runtime: Path) -> None:
    """Make pinned Trae reuse tools mounted by TacoRank's read-only container."""

    docker_manager = (
        _trae_site_packages(runtime) / "trae_agent" / "agent" / "docker_manager.py"
    )
    try:
        if (
            docker_manager.is_symlink()
            or not docker_manager.is_file()
            or docker_manager.resolve(strict=True) != docker_manager
        ):
            raise DeploymentError("Trae Docker manager is not a canonical source file")
        original = docker_manager.read_text(encoding="utf-8")
    except OSError as exc:
        raise DeploymentError("Trae Docker manager could not be read") from exc
    anchor = '''        print(
            f"Copying tools from '{self.tools_dir}' to container path '{self.CONTAINER_TOOLS_PATH}'..."
        )
'''
    if original.count(anchor) != 1 or TRAE_READ_ONLY_TOOL_MARKER in original:
        raise DeploymentError("pinned Trae read-only attach patch does not apply cleanly")
    pre_mounted_probe = f'''        # {TRAE_READ_ONLY_TOOL_MARKER}.
        if self.container_id and self.container is not None:
            probe = self.container.exec_run(
                [
                    "/bin/sh",
                    "-c",
                    "test -x /agent_tools/edit_tool && "
                    "test -x /agent_tools/json_edit_tool",
                ]
            )
            if probe.exit_code == 0:
                print("Using pre-mounted tools in '/agent_tools'.")
                return

'''
    patched = original.replace(anchor, pre_mounted_probe + anchor)
    try:
        compile(patched, str(docker_manager), "exec")
        docker_manager.write_text(patched, encoding="utf-8")
    except (OSError, SyntaxError) as exc:
        raise DeploymentError("pinned Trae read-only attach patch is invalid") from exc


def _install_trae_tools(
    runtime: Path,
    docker: Path,
    image: str,
    root: Path,
) -> None:
    """Install the pinned Linux tool bundle built into the runtime image."""

    site_packages = _trae_site_packages(runtime)
    package_root = site_packages / "trae_agent"
    destination = package_root / "dist"
    staging = runtime / ".trae-tools-staging"
    if (
        not package_root.is_dir()
        or package_root.is_symlink()
        or package_root.resolve(strict=True) != package_root
        or staging.exists()
        or staging.is_symlink()
    ):
        raise DeploymentError("Trae package cannot receive canonical Docker tools")
    staging.mkdir(mode=0o700)
    container_id = _run_output(
        (str(docker), "create", "--entrypoint", "/bin/true", image),
        cwd=root,
        label="Trae tool extraction container creation",
    )
    if len(container_id) != 64 or any(
        character not in "0123456789abcdef" for character in container_id
    ):
        raise DeploymentError("Docker returned an invalid tool extraction container")
    try:
        _run(
            (
                str(docker),
                "cp",
                container_id + ":/opt/tacorank-trae-tools/.",
                str(staging),
            ),
            cwd=root,
            label="Trae tool extraction",
        )
        _validate_trae_tools(staging)
        if destination.is_symlink() or not destination.is_dir():
            raise DeploymentError("installed Trae tool directory is invalid")
        shutil.rmtree(destination)
        os.replace(staging, destination)
    finally:
        _run(
            (str(docker), "rm", "--force", "--volumes", container_id),
            cwd=root,
            label="Trae tool extraction container cleanup",
        )


def _validate_trae_tools(root: Path) -> None:
    required = (root / "edit_tool", root / "json_edit_tool")
    internal = root / "_internal"
    if (
        root.is_symlink()
        or root.resolve(strict=True) != root
        or internal.is_symlink()
        or not internal.is_dir()
        or internal.resolve(strict=True) != internal
    ):
        raise DeploymentError("built Trae tools have an invalid canonical layout")
    paths = tuple(root.rglob("*"))
    if not paths or len(paths) > 10_000:
        raise DeploymentError("built Trae tool asset count is invalid")
    total_bytes = 0
    for path in paths:
        if path.is_symlink():
            raise DeploymentError("built Trae tools cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise DeploymentError("built Trae tools contain a non-regular asset")
        total_bytes += path.stat().st_size
        if total_bytes > 1024 * 1024 * 1024:
            raise DeploymentError("built Trae tools exceed the deployment byte bound")
    for executable in required:
        if (
            executable.is_symlink()
            or not executable.is_file()
            or executable.resolve(strict=True) != executable
            or not os.access(executable, os.X_OK)
        ):
            raise DeploymentError("built Trae tool executable is invalid")


def _trae_site_packages(runtime: Path) -> Path:
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site_packages_text = _run_output(
        (str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"),
        cwd=runtime,
        label="Trae site-packages discovery",
    )
    return Path(site_packages_text).resolve(strict=True)


def _trae_identity(runtime: Path) -> Mapping[str, Path]:
    executable = runtime / ("Scripts/trae-cli.exe" if os.name == "nt" else "bin/trae-cli")
    site_packages = _trae_site_packages(runtime)
    direct_urls = tuple(site_packages.glob("trae_agent-*.dist-info/direct_url.json"))
    dotenv_metadata = tuple(site_packages.glob("python_dotenv-*.dist-info/METADATA"))
    if len(direct_urls) != 1 or len(dotenv_metadata) != 1:
        raise DeploymentError("Trae installation identity files are ambiguous")
    return {
        "executable": executable.resolve(strict=True),
        "site_packages": site_packages,
        "direct_url": direct_urls[0].resolve(strict=True),
        "dotenv_metadata": dotenv_metadata[0].resolve(strict=True),
    }


def _build_runtime_image(root: Path, docker: Path) -> Tuple[str, str]:
    commit = _git_text(root, ("rev-parse", "--short=12", "HEAD"))
    tag = "tacorank-runtime:" + commit
    platform = _run_output(
        (str(docker), "version", "--format", "{{.Server.Os}}/{{.Server.Arch}}"),
        cwd=root,
        label="Docker server platform",
    )
    if platform not in {"linux/amd64", "linux/arm64"}:
        raise DeploymentError("Docker server platform is not supported: " + platform)
    _run(
        (
            str(docker),
            "build",
            "--pull",
            "--platform",
            platform,
            "--file",
            str(root / "docker" / "runtime.Dockerfile"),
            "--tag",
            tag,
            str(root),
        ),
        cwd=root,
        label="TacoRank runtime image build",
    )
    image_id = _run_output(
        (str(docker), "image", "inspect", "--format", "{{.Id}}", tag),
        cwd=root,
        label="Docker image identity",
    )
    if len(image_id) != 71 or not image_id.startswith("sha256:"):
        raise DeploymentError("Docker returned an invalid image identity")
    environment_text = _run_output(
        (
            str(docker),
            "image",
            "inspect",
            "--format",
            "{{json .Config.Env}}",
            image_id,
        ),
        cwd=root,
        label="Docker image environment identity",
    )
    environment = json.loads(environment_text)
    if environment is None:
        environment = []
    if not isinstance(environment, list) or not all(
        isinstance(value, str) for value in environment
    ):
        raise DeploymentError("Docker image environment is malformed")
    payload = json.dumps(environment, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return image_id, hashlib.sha256(payload).hexdigest()


def _discover_docker_host(docker: Path, root: Path) -> str:
    value = _run_output(
        (
            str(docker),
            "context",
            "inspect",
            "--format",
            "{{.Endpoints.docker.Host}}",
        ),
        cwd=root,
        label="Docker context discovery",
    )
    if not value.startswith("unix://") or "\x00" in value:
        raise DeploymentError("production Docker must use a local Unix socket")
    socket_path = Path(value[len("unix://") :])
    try:
        resolved = socket_path.resolve(strict=True)
    except OSError as error:
        raise DeploymentError("Docker context Unix socket is unavailable") from error
    if not resolved.is_socket():
        raise DeploymentError("Docker context endpoint is not a Unix socket")
    return "unix://" + str(resolved)


def _download_data(root: Path, data: Path) -> None:
    expected = (root / "KuaiRand-Pure" / "data").resolve()
    if data.resolve() != expected:
        raise DeploymentError("automatic download requires KuaiRand-Pure/data")
    archive = root / "KuaiRand-Pure.tar.gz"
    if archive.exists():
        raise DeploymentError("refusing to overwrite existing KuaiRand-Pure.tar.gz")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(archive, flags, 0o600)
    digest = hashlib.md5(usedforsecurity=False)
    total_bytes = 0
    try:
        request = urllib.request.Request(DATA_URL, method="GET")
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(request, timeout=120) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total_bytes += len(block)
                    if total_bytes > 2 * 1024 * 1024 * 1024:
                        raise DeploymentError("downloaded dataset archive exceeds size limit")
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
    except BaseException:
        try:
            archive.unlink()
        except FileNotFoundError:
            pass
        raise
    if digest.hexdigest() != DATA_ARCHIVE_MD5:
        archive.unlink()
        raise DeploymentError("downloaded dataset archive checksum does not match Zenodo")
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise DeploymentError("downloaded dataset archive contains an unsafe path")
            parts = path.parts
            if len(parts) < 2 or parts[:2] != ("KuaiRand-Pure", "data"):
                continue
            target = root.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or target.exists():
                raise DeploymentError("downloaded dataset archive contains an unsafe member")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise DeploymentError("downloaded dataset member is unreadable")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(target, flags, 0o600)
            with source, os.fdopen(descriptor, "wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _require_raw_data(data: Path, root: Path) -> None:
    try:
        resolved = data.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise DeploymentError("data directory must be a real directory inside the repository") from error
    if (
        data.is_symlink()
        or not resolved.is_dir()
        or any(
            (resolved / name).is_symlink() or not (resolved / name).is_file()
            for name in RAW_REQUIRED
        )
    ):
        raise DeploymentError("KuaiRand-Pure raw data is incomplete")


def _has_raw_data(data: Path) -> bool:
    return data.is_dir() and all((data / name).is_file() for name in RAW_REQUIRED)


def _require_clean_tracked_checkout(root: Path) -> None:
    status = _run_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=no"),
        cwd=root,
        label="Git checkout verification",
    )
    if status:
        raise DeploymentError(
            "tracked files must be committed before generating a production deployment"
        )


def _require_python312(python: Path) -> None:
    version = _run_output(
        (str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"),
        cwd=python.parent,
        label="Python version check",
    )
    if version != "3.12":
        raise DeploymentError("the pinned Trae runtime requires Python 3.12")


def _new_directory_inside(root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else root / value
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DeploymentError("deployment directory must be inside repository_root") from error
    if resolved.exists() or resolved.is_symlink():
        raise DeploymentError("deployment directory already exists: %s" % resolved)
    return resolved


def _new_external_directory(root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else root.parent / value
    resolved = candidate.resolve(strict=False)
    if resolved == root or root in resolved.parents:
        raise DeploymentError("Trae runtime must be outside the repository")
    if resolved.exists() or resolved.is_symlink():
        raise DeploymentError("Trae runtime directory already exists: %s" % resolved)
    return resolved


def _regular_executable(value: Path, label: str) -> Path:
    try:
        path = Path(value).resolve(strict=True)
    except OSError as error:
        raise DeploymentError("%s executable is unavailable" % label) from error
    if not path.is_file() or not os.access(path, os.X_OK):
        raise DeploymentError("%s executable is unavailable" % label)
    return path


def _exclusive_csv(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "w", newline="", encoding="utf-8")


def _copy_exclusive(source: Path, destination: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _write_text_exclusive(path: Path, value: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(root: Path, args: Sequence[str]) -> str:
    return _run_output(("git", *args), cwd=root, label="Git identity")


def _run_output(args: Sequence[str], *, cwd: Path, label: str) -> str:
    completed = _run(args, cwd=cwd, label=label, capture_output=True)
    return completed.stdout.strip()


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    label: str,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            shell=False,
            check=False,
        )
    except OSError as error:
        raise DeploymentError("%s could not start" % label) from error
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise DeploymentError("%s failed%s" % (label, ": " + detail[-1000:] if detail else ""))
    return completed
