"""Fresh-clone production deployment preparation for TacoRank."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import certifi

from .coding import (
    TRAE_DEEPSEEK_REASONING_MARKER,
    TRAE_DEEPSEEK_TOOL_JSON_MARKER,
    TRAE_DOCKER_EDIT_TOOL_MARKER,
    TRAE_STATELESS_DOCKER_MARKER,
    hash_trae_runtime_package,
)
from .docker_host import normalize_local_docker_host
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
TRAE_ONLY_DATA_BOUNDARY_SHA256 = hashlib.sha256(
    b"tacorank-trae-only-no-dataset-v1"
).hexdigest()


class DeploymentError(RuntimeError):
    """A local production prerequisite or generated identity is invalid."""


def setup_trae_deployment(
    *,
    repository_root: Path,
    deployment_directory: Path,
    runtime_directory: Path,
    python312: Path,
    docker_executable: Path,
) -> Mapping[str, Any]:
    """Prepare only the production Trae coding path, without benchmark data."""

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

    runtime.mkdir(parents=True, exist_ok=False)
    deployment.mkdir(parents=True, exist_ok=False)
    assets = _prepare_trae_runtime(root, runtime, python, docker)
    config_path = deployment / "trae-deployment.json"
    payload = {
        "schema_version": "1.0",
        "repository_root": str(root),
        "worktree_root": str(
            (root.parent / ".tacorank-worktrees" / root.name).resolve()
        ),
        "required_submodules": ["kuairand-starter-kit"],
        "contract_path": "contract/COMPETITION.md",
        "protected_paths_path": "PROTECTED_PATHS.md",
        "artifact_roots": ["artifacts", "runs"],
        "editable_roots": ["solution"],
        "allowed_command_ids": [
            "candidate_smoke",
            "candidate_proxy",
            "candidate_full",
        ],
        "target_interface_excerpts": {
            "candidate": (
                "def run(invocation: PipelineInvocation) -> None; read only "
                "invocation.input_root and write exactly invocation.output_path as "
                "row_id,user_id,video_id,score CSV; use invocation.fidelity and "
                "invocation.seed; return None. train.csv has the exact columns "
                "date,user_id,video_id,author_id,tab,duration_ms,long_view, where "
                "date is an integer YYYYMMDD value; "
                "score.csv has row_id,date,user_id,video_id,author_id,tab,duration_ms "
                "and never exposes long_view. fm_baseline_predictions.csv is the "
                "setup-verified official FM score for every score.csv row. These are "
                "unconstrained real-valued ranking scores, not probabilities. Never "
                "sigmoid, clip to [0,1], normalize, or rescale the FM parent or a "
                "parent-plus-residual result. Preserve it as the strong parent and "
                "add only a bounded train-only residual on the original score scale "
                "unless the approved hypothesis explicitly replaces the parent. "
"The candidate runs in an offline container with no network: the only "
                "importable third-party packages are numpy, pandas, pydantic, and "
                "PyYAML, plus the standard library. Importing anything else fails "
                "the isolated entrypoint import check and rejects the patch. "
                "The smoke, proxy, and full views share identical training data; only the "
                "scored population differs. Proxy is a terminal decision gate, not a "
                "rehearsal: a proxy regression prunes the experiment and it never "
                "reaches full evaluation. Train at full strength for every fidelity "
                "and use fidelity only to bound scoring cost, never to shrink the "
                "training sample, epochs, or capacity. "
                                "Bound the residual relative to the parent score's own spread rather "
                "than an absolute constant: a cap far below one standard deviation "
                "of the parent scores cannot reorder anything, and a candidate that "
                "leaves within-user ordering essentially unchanged is rejected as a "
                "no-op. Only within-user ordering is measured, so a change that is "
                "constant inside a user's list cannot move the metrics. "
                                "Training dates strictly precede score dates. Preserve contiguous "
                "score row_id order, duplicate rows, finite deterministic scores, "
                "and exclusive output creation. The production loader imports "
                "solution.candidate:run. Keep the implementation in candidate.py "
                "unless every helper path and its import pattern are explicitly "
                "authorized by the ExperimentSpec target_files."
            )
        },
        "coding_step_limit": 64,
        "coding_token_limit": None,
        "coding_wall_time_limit_seconds": 1800,
        "data_boundary_sha256": TRAE_ONLY_DATA_BOUNDARY_SHA256,
        "trae": _trae_payload(
            runtime=runtime,
            runtime_identity=assets["runtime_identity"],
            trae_yaml=assets["trae_yaml"],
            docker=docker,
            docker_host=docker_host,
            image=assets["image"],
        ),
    }
    _write_json_exclusive(config_path, payload)
    return {
        "trae_config": str(config_path),
        "runtime": str(runtime),
        "docker_image": assets["image"],
        "model": DEEPSEEK_MODEL,
        "reasoning_effort": "high",
        "dataset_prepared": False,
    }


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
    assets = _prepare_trae_runtime(root, runtime, python, docker)
    image = str(assets["image"])
    image_environment_sha256 = str(assets["image_environment_sha256"])
    runtime_identity = assets["runtime_identity"]
    generated_data = _prepare_data(root, deployment, data)

    trae_yaml = assets["trae_yaml"]
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
        "trae": _trae_payload(
            runtime=runtime,
            runtime_identity=runtime_identity,
            trae_yaml=trae_yaml,
            docker=docker,
            docker_host=docker_host,
            image=image,
        ),
        "contract_root": str(generated_data["contract_root"]),
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
        "baseline_final_prediction_csv": str(
            generated_data["baseline_final_prediction_csv"]
        ),
        "baseline_parity_receipt_path": str(
            generated_data["baseline_parity_receipt_path"]
        ),
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
        "allowed_research_families": [
            "objective",
            "temporal_history",
            "multitask",
            "duration_bias",
            "features",
            "model",
            "sampling",
            "ensemble",
            "evaluation",
            "other",
        ],
        "allowed_research_data": [
            "train_interactions",
            "public_validation",
            "user_id",
            "video_id",
            "author_id",
            "tab",
            "date",
            "duration_ms",
            "long_view",
            "verified_predictions",
        ],
        "research_capabilities": [
            "baseline_parity",
            "objective_data_frame_verified",
            "verified_best_prediction",
        ],
        "active_research_prohibitions": [],
        "prediction_change_no_op_threshold": 0.001,
        "max_single_score_fraction": 0.5,
        "target_interface_excerpts": {
            "solution/candidate.py": (
                "Required candidate entrypoint: def run(invocation: "
                "PipelineInvocation) -> None; include this file in target_files; read only "
                "invocation.input_root and write exactly invocation.output_path as "
                "row_id,user_id,video_id,score CSV; use invocation.fidelity and "
                "invocation.seed; return None. train.csv has the exact columns "
                "date,user_id,video_id,author_id,tab,duration_ms,long_view, where "
                "date is an integer YYYYMMDD value; "
                "score.csv has row_id,date,user_id,video_id,author_id,tab,duration_ms "
                "and never exposes long_view. fm_baseline_predictions.csv contains "
                "the setup-verified official FM score aligned one-to-one with "
                "score.csv; fm_baseline_predictions.sha256 authenticates it. The "
                "baseline candidate reproduces these bytes exactly. FM scores are "
                "unconstrained real-valued ranking scores, not probabilities. Never "
                "sigmoid, clip to [0,1], normalize, or rescale the FM parent or a "
                "parent-plus-residual result. Keep this FM score as the strong parent "
                "and learn one bounded train-only residual on the original score scale "
                "unless the approved ExperimentSpec explicitly tests replacement. "
"The candidate runs in an offline container with no network: the only "
                "importable third-party packages are numpy, pandas, pydantic, and "
                "PyYAML, plus the standard library. Importing anything else fails "
                "the isolated entrypoint import check and rejects the patch. "
                "The smoke, proxy, and full views share identical training data; only the "
                "scored population differs. Proxy is a terminal decision gate, not a "
                "rehearsal: a proxy regression prunes the experiment and it never "
                "reaches full evaluation. Train at full strength for every fidelity "
                "and use fidelity only to bound scoring cost, never to shrink the "
                "training sample, epochs, or capacity. "
                                "Bound the residual relative to the parent score's own spread rather "
                "than an absolute constant: a cap far below one standard deviation "
                "of the parent scores cannot reorder anything, and a candidate that "
                "leaves within-user ordering essentially unchanged is rejected as a "
                "no-op. Only within-user ordering is measured, so a change that is "
                "constant inside a user's list cannot move the metrics. "
                                "Do not reinterpret duration_ms as watch time: it is video duration. "
                "Training dates strictly precede score dates. Preserve contiguous "
                "score row_id order, duplicate rows, finite deterministic scores, "
                "and exclusive output creation. Use all training rows or report a "
                "deterministic representative sampling fraction in the code. The "
                "production loader imports solution.candidate:run; keep the "
                "implementation in candidate.py unless every helper path and import "
                "pattern are explicitly authorized by target_files."
            )
        },
        "coding_step_limit": 64,
        "coding_token_limit": None,
        "coding_wall_time_limit_seconds": 1800,
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


def _prepare_trae_runtime(
    root: Path,
    runtime: Path,
    python: Path,
    docker: Path,
) -> Mapping[str, Any]:
    """Install, compatibility-patch, and attest the reviewed Trae runtime."""

    _install_trae(python, runtime, root)
    _patch_trae_read_only_attach(runtime)
    _patch_trae_cross_platform_docker_exec(runtime)
    _patch_trae_deepseek_reasoning(runtime)
    _patch_trae_docker_edit_tool(runtime)
    image, image_environment_sha256 = _build_runtime_image(root, docker)
    _install_trae_tools(runtime, docker, image, root)
    runtime_identity = _trae_identity(runtime)
    trae_yaml = runtime / "trae-agent.yaml"
    _write_text_exclusive(trae_yaml, _trae_yaml())
    trae_yaml.chmod(0o600)
    return {
        "image": image,
        "image_environment_sha256": image_environment_sha256,
        "runtime_identity": runtime_identity,
        "trae_yaml": trae_yaml,
    }


def _trae_payload(
    *,
    runtime: Path,
    runtime_identity: Mapping[str, Path],
    trae_yaml: Path,
    docker: Path,
    docker_host: str,
    image: str,
) -> Mapping[str, Any]:
    return {
        "command_prefix": [str(runtime_identity["executable"])],
        "trae_version": "0.1.0",
        "provider": "openai",
        "provider_base_url": DEEPSEEK_BASE_URL,
        "model_id": DEEPSEEK_MODEL,
        "reasoning_effort": "high",
        "config_file": str(trae_yaml),
        "config_sha256": _sha256_file(trae_yaml),
        "max_steps_cap": 64,
        "max_token_cap": None,
        "max_wall_time_seconds_cap": 1800,
        "repair_step_limit": 20,
        "repair_token_limit": None,
        "repair_wall_time_limit_seconds": 1200,
        "repair_allowed_command_ids": ["candidate_smoke"],
        "solution_verification_max_attempts": 5,
        "solution_verification_timeout_seconds": 120,
        "solution_verification_max_output_tokens": 4096,
        "solution_verification_max_source_bytes": 524288,
        "solution_revision_step_limit": 32,
        "solution_revision_wall_time_limit_seconds": 600,
        "solution_verifier_credential_environment_name": "DEEPSEEK_API_KEY",
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
    contract_view = deployment / "contract"
    contract_view.mkdir(mode=0o700)
    _copy_exclusive(
        root / "contract" / "COMPETITION.md",
        contract_view / "COMPETITION.md",
    )
    _write_submission_rows(contract_view / "submission_rows.csv", test)
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
            "-B",
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
    smoke_baseline_prediction_csv = baselines / "smoke.csv"
    _write_prediction_subset(
        smoke_baseline_prediction_csv, baseline_rows, smoke_indices
    )
    _write_prediction_subset(
        baseline_prediction_csvs["proxy"], baseline_rows, proxy_indices
    )
    _copy_exclusive(official_baseline, baseline_prediction_csvs["full"])
    baseline_final_prediction_csv = protected / "official-fm-test.csv"
    _run(
        (
            sys.executable,
            "-B",
            "submit.py",
            str(baseline_final_prediction_csv),
            "--data_dir",
            str(data),
            "--split",
            "test",
            "--make",
        ),
        cwd=root / "kuairand-starter-kit",
        label="official FM final submission generation",
    )

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
    fm_prediction_sources = {
        "candidate_smoke": smoke_baseline_prediction_csv,
        "candidate_proxy": baseline_prediction_csvs["proxy"],
        "candidate_full": baseline_prediction_csvs["full"],
        "candidate_final_infer": baseline_final_prediction_csv,
    }
    for command_id, directory in command_directories.items():
        directory.mkdir(mode=0o700)
        os.link(train_path, directory / "train.csv")
        rows, indices = index_sets[command_id]
        selected: Iterable[Sequence[Any]] = rows
        if indices is not None:
            selected = (rows[index] for index in indices)
        _write_score(directory / "score.csv", selected)
        fm_view = directory / "fm_baseline_predictions.csv"
        _copy_exclusive(fm_prediction_sources[command_id], fm_view)
        _write_text_exclusive(
            directory / "fm_baseline_predictions.sha256",
            _sha256_file(fm_view) + "\n",
        )

    parity_receipt = deployment / "baseline-parity-receipt.json"
    _write_json_exclusive(
        parity_receipt,
        _candidate_baseline_parity_receipt(root, command_directories),
    )

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
        "contract_root": contract_view,
        "input_roots": input_roots,
        "population_csvs": population_csvs,
        "baseline_prediction_csvs": baseline_prediction_csvs,
        "baseline_final_prediction_csv": baseline_final_prediction_csv,
        "baseline_parity_receipt_path": parity_receipt,
    }


def _candidate_baseline_parity_receipt(
    root: Path,
    input_roots: Mapping[str, Path],
) -> Mapping[str, Any]:
    """Execute the editable parent and prove exact FM bytes at every route."""

    source = root / "solution" / "candidate.py"
    specification = importlib.util.spec_from_file_location(
        "tacorank_setup_verified_candidate", source
    )
    if specification is None or specification.loader is None:
        raise DeploymentError("candidate baseline entrypoint could not be loaded")
    module = importlib.util.module_from_spec(specification)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    implementation = getattr(module, "run", None)
    if not callable(implementation):
        raise DeploymentError("candidate baseline does not define callable run")

    routes: Dict[str, Mapping[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="tacorank-baseline-parity-") as temporary:
        temporary_root = Path(temporary).resolve(strict=True)
        for command_id in sorted(input_roots):
            input_root = input_roots[command_id].resolve(strict=True)
            expected = input_root / "fm_baseline_predictions.csv"
            output = temporary_root / (command_id + ".csv")
            try:
                result = implementation(
                    SimpleNamespace(input_root=input_root, output_path=output)
                )
            except Exception as error:
                raise DeploymentError(
                    "candidate baseline parity execution failed for %s" % command_id
                ) from error
            if result is not None or not output.is_file() or output.is_symlink():
                raise DeploymentError(
                    "candidate baseline parity produced an invalid output for %s"
                    % command_id
                )
            expected_sha = _sha256_file(expected)
            output_sha = _sha256_file(output)
            if output_sha != expected_sha:
                raise DeploymentError(
                    "candidate baseline does not reproduce official FM for %s"
                    % command_id
                )
            routes[command_id] = {
                "fm_prediction_sha256": expected_sha,
                "candidate_output_sha256": output_sha,
                "exact_bytes_match": True,
            }
    return {
        "schema_version": "1.0",
        "candidate_entrypoint": "solution.candidate:run",
        "candidate_source_path": "solution/candidate.py",
        "candidate_source_sha256": _sha256_file(source),
        "routes": routes,
    }


def _load_official_splits(root: Path, data: Path) -> Mapping[str, Any]:
    source = root / "kuairand-starter-kit" / "data.py"
    specification = importlib.util.spec_from_file_location(
        "tacorank_reviewed_kuairand_data", source
    )
    if specification is None or specification.loader is None:
        raise DeploymentError("official KuaiRand data loader could not be loaded")
    module = importlib.util.module_from_spec(specification)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
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


def _write_submission_rows(path: Path, rows: Iterable[Sequence[Any]]) -> None:
    with _exclusive_csv(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id"))
        for row_id, row in enumerate(rows):
            writer.writerow((row_id, row[1], row[2]))


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
    max_steps: 64
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
    max_tokens: 32768
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


def _patch_trae_cross_platform_docker_exec(runtime: Path) -> None:
    """Replace Trae's Unix-only interactive pexpect shell with Docker exec.

    The reviewed container remains the security boundary. Each tool call starts
    in ``/workspace`` and is bounded inside the container by coreutils
    ``timeout``. This works through Docker Desktop's named pipe on Windows and
    its Unix socket on macOS/Linux without relying on a host pseudo-terminal.
    """

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

    startup_anchor = '''            self._copy_tools_to_container()
            # if self.interactive:
            self._start_persistent_shell()
'''
    execute_anchor = '''        # if self.interactive:
        return self._execute_interactive(command, timeout)
        # else:
        #     return self._execute_stateless(command)
'''
    helper_anchor = '''    def _start_persistent_shell(self):
'''
    if (
        original.count(startup_anchor) != 1
        or original.count(execute_anchor) != 1
        or original.count(helper_anchor) != 1
        or TRAE_STATELESS_DOCKER_MARKER in original
    ):
        raise DeploymentError(
            "pinned Trae cross-platform Docker patch does not apply cleanly"
        )

    startup_replacement = f'''            self._copy_tools_to_container()
            # {TRAE_STATELESS_DOCKER_MARKER}.
            probe = self.container.exec_run(
                ["/bin/sh", "-c", "command -v timeout >/dev/null"],
                workdir=self.container_workspace,
            )
            if probe.exit_code != 0:
                raise DockerException(
                    "Container does not provide the reviewed timeout command."
                )
            print("Stateless Docker execution is ready.")
'''
    stateless_helper = '''    def _execute_stateless(self, command: str, timeout_seconds: int) -> tuple[int, str]:
        """Execute one bounded command without a host pseudo-terminal."""
        if not self.container:
            raise RuntimeError("Container is not running. Call start() first.")
        try:
            bounded_timeout = max(1, int(timeout_seconds))
        except (TypeError, ValueError) as exc:
            raise ValueError("Docker command timeout must be a positive integer.") from exc
        result = self.container.exec_run(
            [
                "timeout", "--signal=KILL", f"{bounded_timeout}s",
                "/bin/bash", "-lc", command,
            ],
            workdir=self.container_workspace,
        )
        output = result.output
        if isinstance(output, tuple):
            output = b"".join(part or b"" for part in output)
        if isinstance(output, bytes):
            rendered = output.decode("utf-8", errors="replace")
        else:
            rendered = str(output or "")
        return int(result.exit_code), rendered.strip()

'''
    patched = (
        original.replace(startup_anchor, startup_replacement)
        .replace(
            execute_anchor,
            "        return self._execute_stateless(command, timeout)\n",
        )
        .replace(helper_anchor, stateless_helper + helper_anchor)
    )
    try:
        compile(patched, str(docker_manager), "exec")
        docker_manager.write_text(patched, encoding="utf-8")
    except (OSError, SyntaxError) as exc:
        raise DeploymentError(
            "pinned Trae cross-platform Docker patch is invalid"
        ) from exc


def _patch_trae_deepseek_reasoning(runtime: Path) -> None:
    """Make the pinned Responses client explicit and continuous for DeepSeek thinking."""

    client = (
        _trae_site_packages(runtime)
        / "trae_agent"
        / "utils"
        / "llm_clients"
        / "openai_client.py"
    )
    try:
        if (
            client.is_symlink()
            or not client.is_file()
            or client.resolve(strict=True) != client
        ):
            raise DeploymentError("Trae OpenAI client is not a canonical source file")
        original = client.read_text(encoding="utf-8")
    except OSError as exc:
        raise DeploymentError("Trae OpenAI client could not be read") from exc

    request_anchor = """            max_output_tokens=model_config.max_tokens,
"""
    response_anchor = """        for output_block in response.output:
            if output_block.type == "function_call":
"""
    function_call_anchor = '''            if output_block.type == "function_call":
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
'''
    message_anchor = '''            elif output_block.type == "message":
                content = "".join(
                    content_block.text
                    for content_block in output_block.content
                    if content_block.type == "output_text"
                )

        if content != "":
            self.message_history.append(
                EasyInputMessageParam(content=content, role="assistant", type="message")
            )
'''
    if (
        original.count(request_anchor) != 1
        or original.count(response_anchor) != 1
        or original.count(function_call_anchor) != 1
        or original.count(message_anchor) != 1
        or TRAE_DEEPSEEK_REASONING_MARKER in original
        or TRAE_DEEPSEEK_TOOL_JSON_MARKER in original
    ):
        raise DeploymentError("pinned Trae DeepSeek patch does not apply cleanly")

    ordered_message = '''            elif output_block.type == "message":
                message_content = "".join(
                    content_block.text
                    for content_block in output_block.content
                    if content_block.type == "output_text"
                )
                content += message_content
                if message_content:
                    self.message_history.append(
                        EasyInputMessageParam(
                            content=message_content,
                            role="assistant",
                            type="message",
                        )
                    )
'''
    safe_function_call = f'''            if output_block.type == "function_call":
                # {TRAE_DEEPSEEK_TOOL_JSON_MARKER}.
                try:
                    tool_arguments = (
                        json.loads(output_block.arguments)
                        if output_block.arguments
                        else {{}}
                    )
                except (json.JSONDecodeError, TypeError, RecursionError):
                    tool_arguments = None
                if not isinstance(tool_arguments, dict):
                    diagnostic = (
                        "TacoRank rejected malformed or truncated tool arguments. "
                        "Retry the same operation with one smaller valid JSON tool call."
                    )
                    content += diagnostic
                    self.message_history.append(
                        EasyInputMessageParam(
                            content=diagnostic,
                            role="assistant",
                            type="message",
                        )
                    )
                    continue
                tool_calls.append(
                    ToolCall(
                        call_id=output_block.call_id,
                        name=output_block.name,
                        arguments=tool_arguments,
                        id=output_block.id,
                    )
                )
                tool_call_param = ResponseFunctionToolCallParam(
                    arguments=json.dumps(tool_arguments, separators=(",", ":")),
                    call_id=output_block.call_id,
                    name=output_block.name,
                    type="function_call",
                )
'''
    patched = original.replace(
        request_anchor,
        request_anchor
        + """            reasoning={"effort": "high"}
            if model_config.model.startswith("deepseek-")
            else openai.NOT_GIVEN,
""",
    ).replace(function_call_anchor, safe_function_call).replace(
        response_anchor,
        f'''        for output_block in response.output:
            # {TRAE_DEEPSEEK_REASONING_MARKER}.
            if output_block.type == "reasoning":
                reasoning_item = output_block.model_dump(exclude_none=True)
                reasoning_item.pop("summary", None)
                reasoning_item.pop("encrypted_content", None)
                self.message_history.append(reasoning_item)
            elif output_block.type == "function_call":
''',
    ).replace(message_anchor, ordered_message)
    try:
        compile(patched, str(client), "exec")
        client.write_text(patched, encoding="utf-8")
    except (OSError, SyntaxError) as exc:
        raise DeploymentError("pinned Trae DeepSeek patch is invalid") from exc


def _patch_trae_docker_edit_tool(runtime: Path) -> None:
    """Normalize DeepSeek edit calls before the pinned Docker CLI boundary."""

    executor = (
        _trae_site_packages(runtime)
        / "trae_agent"
        / "tools"
        / "docker_tool_executor.py"
    )
    try:
        if (
            executor.is_symlink()
            or not executor.is_file()
            or executor.resolve(strict=True) != executor
        ):
            raise DeploymentError("Trae Docker edit executor is not a canonical source file")
        original = executor.read_text(encoding="utf-8")
    except OSError as exc:
        raise DeploymentError("Trae Docker edit executor could not be read") from exc

    import_anchor = "import os\n"
    path_anchor = '''            container_path = os.path.join(self._container_workspace_dir, relative_path)
            return os.path.normpath(container_path)
'''
    block_anchor = '''                executable_path = f"{self._docker_manager.CONTAINER_TOOLS_PATH}/edit_tool"
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
'''
    if (
        original.count(import_anchor) != 1
        or original.count(path_anchor) != 1
        or original.count(block_anchor) != 1
        or TRAE_DOCKER_EDIT_TOOL_MARKER in original
    ):
        raise DeploymentError("pinned Trae Docker edit patch does not apply cleanly")

    replacement = f'''                # {TRAE_DOCKER_EDIT_TOOL_MARKER}.
                command_arguments = {{
                    "view": ("path", "view_range"),
                    "create": ("path", "file_text"),
                    "str_replace": ("path", "old_str", "new_str"),
                    "insert": ("path", "insert_line", "new_str"),
                }}.get(sub_command)
                if command_arguments is None:
                    raise ValueError(f"Unsupported edit sub-command: {{sub_command}}")

                executable_path = f"{{self._docker_manager.CONTAINER_TOOLS_PATH}}/edit_tool"
                cmd_parts = [executable_path, sub_command]
                for key in command_arguments:
                    value = processed_args.get(key)
                    if value is None:
                        continue
                    cmd_parts.append(f"--{{key}}")
                    if isinstance(value, list):
                        cmd_parts.extend(str(item) for item in value)
                    else:
                        cmd_parts.append(str(value))

                command_to_run = shlex.join(cmd_parts)
'''
    patched = (
        original.replace(import_anchor, import_anchor + "import posixpath\nimport shlex\n")
        .replace(
            path_anchor,
            '''            container_path = posixpath.join(
                self._container_workspace_dir, relative_path.replace(os.sep, "/")
            )
            return posixpath.normpath(container_path)
''',
        )
        .replace(block_anchor, replacement)
    )
    try:
        compile(patched, str(executor), "exec")
        executor.write_text(patched, encoding="utf-8")
    except (OSError, SyntaxError) as exc:
        raise DeploymentError("pinned Trae Docker edit patch is invalid") from exc


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
        if destination.is_symlink() or (
            destination.exists() and not destination.is_dir()
        ):
            raise DeploymentError("installed Trae tool directory is invalid")
        if destination.exists():
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
    try:
        return normalize_local_docker_host(value)
    except ValueError as error:
        raise DeploymentError(str(error)) from error


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
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(
                request,
                timeout=120,
                context=tls_context,
            ) as response:
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
