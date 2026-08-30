"""Data-independent production validation for the Trae coding boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import Field, field_validator, model_validator

from ..artifacts import ArtifactStore
from ..git import WorktreeManager, resolve_commit
from ..git.refs import validate_identifier
from ..providers import DeepSeekResearchProvider
from ..research.duplicate_detection import compute_duplicate_key
from ..safety import (
    DataAccessPolicy,
    DockerEntrypointSmokeCheck,
    PatchGate,
    ProtectedManifest,
    ReceiptStore,
    parse_protected_paths_markdown,
    path_is_within,
)
from ..schemas import (
    ArtifactKind,
    CoderContext,
    ExperimentSpec,
    SHA256_RE,
    StrictModel,
    normalize_relative_path,
)
from .trae_adapter import (
    CandidateIdentity,
    CodingWorkerError,
    TraeCodingWorker,
    TraeConfig,
)


class TraeStandaloneError(RuntimeError):
    """A focused Trae validation input or result violated its contract."""


class TraeStandaloneConfig(StrictModel):
    """Generated configuration needed to run Trae without preparing ML data."""

    schema_version: Literal["1.0"] = "1.0"
    repository_root: Path
    worktree_root: Path
    required_submodules: List[str] = Field(default_factory=list)
    contract_path: str
    protected_paths_path: str
    artifact_roots: List[str]
    editable_roots: List[str]
    allowed_command_ids: List[str]
    allowed_import_roots: List[str]
    target_interface_excerpts: Dict[str, str] = Field(default_factory=dict)
    coding_step_limit: int = Field(gt=0)
    coding_token_limit: Optional[int] = Field(default=None, gt=0)
    coding_wall_time_limit_seconds: int = Field(gt=0)
    data_boundary_sha256: str
    trae: Dict[str, Any]
    candidate_entrypoint: str = "solution.candidate:run"
    container_python_executable: str = "/usr/local/bin/python3"

    @field_validator("contract_path", "protected_paths_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return normalize_relative_path(value)

    @field_validator(
        "required_submodules",
        "artifact_roots",
        "editable_roots",
    )
    @classmethod
    def validate_path_lists(cls, values: List[str]) -> List[str]:
        normalized = [normalize_relative_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("path lists must not contain duplicates")
        return normalized

    @field_validator("allowed_command_ids")
    @classmethod
    def validate_commands(cls, values: List[str]) -> List[str]:
        if not values or any(not value.strip() for value in values):
            raise ValueError("allowed_command_ids must be non-empty")
        if len(values) != len(set(values)):
            raise ValueError("allowed_command_ids must be unique")
        return values

    @field_validator("allowed_import_roots")
    @classmethod
    def validate_import_roots(cls, values: List[str]) -> List[str]:
        if not values or any(not value.isidentifier() for value in values):
            raise ValueError("allowed_import_roots must contain import identifiers")
        if values != sorted(set(values)):
            raise ValueError("allowed_import_roots must be sorted and unique")
        return values

    @field_validator("data_boundary_sha256")
    @classmethod
    def validate_data_boundary(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("data_boundary_sha256 must be lowercase sha256")
        return value

    @model_validator(mode="after")
    def validate_required_roots(self) -> "TraeStandaloneConfig":
        if not self.artifact_roots or not self.editable_roots:
            raise ValueError("artifact_roots and editable_roots must be non-empty")
        return self

    @classmethod
    def load(cls, path: Path) -> "TraeStandaloneConfig":
        source = Path(path).resolve(strict=True)
        raw = json.loads(source.read_text(encoding="utf-8"))
        base = source.parent
        root_value = Path(raw["repository_root"])
        raw["repository_root"] = str(
            (root_value if root_value.is_absolute() else base / root_value).resolve(
                strict=True
            )
        )
        worktree_value = Path(raw["worktree_root"])
        raw["worktree_root"] = str(
            (worktree_value if worktree_value.is_absolute() else base / worktree_value)
            .expanduser()
            .resolve(strict=False)
        )
        trae = dict(raw["trae"])
        for key in (
            "config_file",
            "docker_executable",
            "trae_install_root",
            "trae_install_identity_file",
            "trae_runtime_root",
            "python_dotenv_metadata_file",
        ):
            if trae.get(key) is not None:
                value = Path(trae[key])
                trae[key] = str(
                    (value if value.is_absolute() else base / value).resolve(strict=True)
                )
        if trae.get("command_prefix"):
            command = list(trae["command_prefix"])
            executable = Path(command[0])
            command[0] = str(
                (
                    executable
                    if executable.is_absolute()
                    else base / executable
                ).resolve(strict=True)
            )
            trae["command_prefix"] = command
        raw["trae"] = trae
        return cls.model_validate(raw)

    def coding_config(self) -> TraeConfig:
        try:
            config = TraeConfig.from_mapping(self.trae)
        except (KeyError, TypeError, ValueError) as exc:
            raise TraeStandaloneError("invalid generated Trae configuration") from exc
        if (
            config.provider != "openai"
            or config.provider_base_url != "https://api.deepseek.com"
            or config.model_id != "deepseek-v4-flash"
            or config.reasoning_effort != "high"
        ):
            raise TraeStandaloneError(
                "standalone coding requires DeepSeek V4 Flash with high reasoning"
            )
        return config


class _StandaloneIdentityResolver:
    def __init__(self, event_id: str) -> None:
        self.event_id = event_id

    def for_initial(self, context: Any, spec: Any) -> CandidateIdentity:
        if context.experiment_id != spec.experiment_id:
            raise TraeStandaloneError("coder context and experiment identities differ")
        return CandidateIdentity(1, self.event_id)

    def for_repair(self, context: Any, decision: Any) -> CandidateIdentity:
        del context, decision
        raise TraeStandaloneError("standalone example runs do not perform recovery")


def load_example_spec(
    path: Path,
    *,
    base_commit_sha: str,
    run_id: str,
    experiment_id: str,
) -> ExperimentSpec:
    """Resolve only the documented runtime placeholders and validate Person 1 output."""

    validate_identifier(run_id, "run_id")
    validate_identifier(experiment_id, "experiment_id")
    source = Path(path).resolve(strict=True)
    raw = json.loads(source.read_text(encoding="utf-8"))
    replacements = {
        "run_id": ("$RUN_ID", run_id),
        "experiment_id": ("$EXPERIMENT_ID", experiment_id),
        "parent_commit_sha": ("$BASE_COMMIT", base_commit_sha),
    }
    for field, (placeholder, value) in replacements.items():
        if raw.get(field) != placeholder:
            raise TraeStandaloneError(
                "example input field %s must contain %s" % (field, placeholder)
            )
        raw[field] = value
    if raw.get("duplicate_key") != "$DUPLICATE_KEY":
        raise TraeStandaloneError(
            "example input field duplicate_key must contain $DUPLICATE_KEY"
        )
    raw["duplicate_key"] = compute_duplicate_key(raw)
    try:
        return ExperimentSpec.model_validate(raw)
    except ValueError as exc:
        raise TraeStandaloneError("example ExperimentSpec is invalid") from exc


def build_example_context(
    config: TraeStandaloneConfig,
    spec: ExperimentSpec,
    *,
    contract_sha256: str,
) -> CoderContext:
    """Build the canonical Person 2-to-Person 3 handoff without a ledger run."""

    protected_text = (
        config.repository_root / config.protected_paths_path
    ).read_text(encoding="utf-8")
    protected_paths = list(parse_protected_paths_markdown(protected_text))
    for target in spec.target_files:
        if not any(path_is_within(target, root) for root in config.editable_roots):
            raise TraeStandaloneError("example target is outside editable_roots: " + target)
        if any(path_is_within(target, root) for root in protected_paths):
            raise TraeStandaloneError("example target is protected: " + target)

    method_cards = []
    for method_id in spec.method_card_ids:
        card_path = config.repository_root / "research" / "methods" / (method_id + ".md")
        if card_path.is_symlink() or not card_path.is_file():
            raise TraeStandaloneError("example method card is unavailable: " + method_id)
        method_cards.append(
            {
                "method_id": method_id,
                "path": card_path.relative_to(config.repository_root).as_posix(),
                "content": card_path.read_text(encoding="utf-8"),
            }
        )

    target_interfaces = {
        target: config.target_interface_excerpts[target]
        for target in spec.target_files
        if target in config.target_interface_excerpts
    }
    missing_target_interfaces = sorted(
        set(spec.target_files) - set(target_interfaces)
    )
    if missing_target_interfaces:
        raise TraeStandaloneError(
            "example target interface is unavailable: "
            + missing_target_interfaces[0]
        )

    context_seed = {
        "role": "coder",
        "run_id": spec.run_id,
        "experiment_id": spec.experiment_id,
        "contract_sha256": contract_sha256,
        "experiment_spec": spec.model_dump(mode="json"),
        "target_interface_excerpts": target_interfaces,
        "editable_roots": config.editable_roots,
        "protected_paths": protected_paths,
        "allowed_command_ids": config.allowed_command_ids,
        "selected_method_cards": method_cards,
        "step_limit": config.coding_step_limit,
        "token_limit": config.coding_token_limit,
        "wall_time_limit_seconds": config.coding_wall_time_limit_seconds,
    }
    canonical = json.dumps(
        context_seed,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    context_id = "ctx_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    artifacts = ArtifactStore(config.repository_root, config.artifact_roots)
    artifact = artifacts.write(
        artifact_id="artifact_" + context_id,
        kind=ArtifactKind.CONTEXT,
        relative_path="runs/%s/contexts/%s.json" % (spec.run_id, context_id),
        content=canonical.encode("utf-8"),
        content_type="application/json",
    )
    return CoderContext(
        context_id=context_id,
        role="coder",
        run_id=spec.run_id,
        experiment_id=spec.experiment_id,
        snapshot_event_id=None,
        source_event_ids=[],
        excluded_source_ids={},
        content=canonical,
        estimated_tokens=max(1, (len(canonical) + 3) // 4),
        artifact=artifact,
        contract_sha256=contract_sha256,
        experiment_spec=spec,
        parent_commit_sha=spec.parent_commit_sha,
        target_interface_excerpts=target_interfaces,
        editable_roots=config.editable_roots,
        protected_paths=protected_paths,
        allowed_command_ids=config.allowed_command_ids,
        selected_method_cards=method_cards,
        active_lessons=[],
        step_limit=config.coding_step_limit,
        token_limit=config.coding_token_limit,
        wall_time_limit_seconds=config.coding_wall_time_limit_seconds,
    )


def preflight_trae(
    config: TraeStandaloneConfig,
    *,
    local_only: bool = False,
) -> Mapping[str, Any]:
    """Verify local isolation and, unless excluded, live DeepSeek model access."""

    base_commit = _clean_head(config.repository_root)
    worker, _ = _worker(config, base_commit, "evt_trae_preflight")
    if local_only:
        worker.preflight_local()
    else:
        worker.preflight()
        _provider(config).preflight()
    coding = config.coding_config()
    return {
        "status": "passed",
        "scope": "local_runtime" if local_only else "local_runtime_and_provider",
        "base_commit_sha": base_commit,
        "model": coding.model_id,
        "reasoning_effort": coding.reasoning_effort,
        "docker_isolation": True,
        "credential_checked": not local_only,
        "provider_checked": not local_only,
        "dataset_required": False,
        "ledger_created": False,
    }


async def run_trae_example(
    config: TraeStandaloneConfig,
    example_path: Path,
    *,
    run_id: str,
    experiment_id: str,
) -> Mapping[str, Any]:
    """Run one real Trae patch and Gate A, deliberately stopping before ML."""

    base_commit = _clean_head(config.repository_root)
    spec = load_example_spec(
        example_path,
        base_commit_sha=base_commit,
        run_id=run_id,
        experiment_id=experiment_id,
    )
    contract_bytes = (config.repository_root / config.contract_path).read_bytes()
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    manifest = ProtectedManifest.from_markdown(
        config.repository_root / config.protected_paths_path,
        config.repository_root,
        contract_paths=(config.contract_path,),
        data_manifest_sha256=config.data_boundary_sha256,
    )
    context = build_example_context(config, spec, contract_sha256=contract_sha256)
    event_id = "evt_trae_" + hashlib.sha256(
        (context.context_id + base_commit).encode("utf-8")
    ).hexdigest()[:16]
    worker, worktrees = _worker(config, base_commit, event_id)
    worker.preflight()
    _provider(config).preflight()
    try:
        candidate = await worker.create_patch(context, spec)
    except CodingWorkerError as exc:
        diagnostic = (exc.output_tail or "").strip()[:2_048]
        suffix = ": " + diagnostic if diagnostic else ""
        raise TraeStandaloneError(
            "%s: %s%s" % (exc.code, exc.summary, suffix)
        ) from exc

    coding = config.coding_config()
    if (
        coding.docker_executable is None
        or coding.docker_host is None
        or coding.docker_image is None
    ):
        raise TraeStandaloneError("Gate A requires the reviewed Docker boundary")
    gate = PatchGate(
        repository_root=worktrees.path_for(run_id, experiment_id),
        artifact_repository_root=config.repository_root,
        editable_roots=config.editable_roots,
        protected_manifest=manifest,
        receipt_store=ReceiptStore(config.repository_root),
        data_access_policy=DataAccessPolicy(
            views=(),
            protected_columns=("label",),
            hidden_path_tokens=("hidden_labels", "final_labels", "test_labels"),
            future_column_patterns=(r"(?:^|_)future(?:_|$)",),
        ),
        allowed_command_ids=config.allowed_command_ids,
        artifact_roots=config.artifact_roots,
        allowed_import_roots=config.allowed_import_roots,
        allowed_capability_imports=(),
        allowed_dependency_changes=(),
        smoke_check=DockerEntrypointSmokeCheck(
            docker_executable=coding.docker_executable,
            docker_host=coding.docker_host,
            image=coding.docker_image,
            container_python_executable=config.container_python_executable,
            entrypoint=config.candidate_entrypoint,
        ),
    )
    checked = await gate.check(
        candidate,
        experiment_root_commit_sha=base_commit,
        authorized_changed_files=spec.target_files,
    )
    if not checked.accepted:
        violations = [violation.code for violation in checked.violations]
        raise TraeStandaloneError(
            "Trae created a patch, but Gate A rejected it: " + ", ".join(violations)
        )
    return {
        "status": "passed",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "base_commit_sha": base_commit,
        "patch_commit_sha": candidate.patch_commit_sha,
        "changed_files": candidate.changed_files,
        "diff_artifact": candidate.diff_artifact.model_dump(mode="json"),
        "trajectory_artifact": candidate.trajectory_artifact.model_dump(mode="json"),
        "gate_a_receipt_id": checked.receipt_id,
        "gate_a_receipt_artifact": checked.receipt_artifact.model_dump(mode="json"),
        "model": coding.model_id,
        "reasoning_effort": coding.reasoning_effort,
        "docker_isolation": True,
        "ml_training_started": False,
    }


def _worker(
    config: TraeStandaloneConfig,
    base_commit: str,
    event_id: str,
) -> tuple[TraeCodingWorker, WorktreeManager]:
    worktrees = WorktreeManager(
        config.repository_root,
        config.worktree_root,
        required_submodules=config.required_submodules,
    )
    worktrees.preflight(base_commit)
    worker = TraeCodingWorker(
        worktrees=worktrees,
        artifact_repository_root=config.repository_root,
        config=config.coding_config(),
        identity_resolver=_StandaloneIdentityResolver(event_id),
    )
    return worker, worktrees


def _provider(config: TraeStandaloneConfig) -> DeepSeekResearchProvider:
    coding = config.coding_config()
    credential_names = coding.credential_environment_names
    if credential_names != ("DEEPSEEK_API_KEY",):
        raise TraeStandaloneError("standalone DeepSeek credential contract is invalid")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise TraeStandaloneError(
            "set DEEPSEEK_API_KEY in the process environment before live Trae validation"
        )
    return DeepSeekResearchProvider(
        api_key=api_key,
        model=coding.model_id,
        base_url=coding.provider_base_url or "https://api.deepseek.com",
        thinking_enabled=True,
        reasoning_effort="high",
    )


def _clean_head(repository_root: Path) -> str:
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    if status.returncode != 0:
        raise TraeStandaloneError("Git checkout status could not be verified")
    if status.stdout:
        raise TraeStandaloneError(
            "tracked files must be committed before a real Trae example run"
        )
    resolved = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "--verify", "HEAD^{commit}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    if resolved.returncode != 0:
        raise TraeStandaloneError("Git HEAD could not be resolved")
    return resolve_commit(repository_root, resolved.stdout.strip())


def run_example_sync(
    config: TraeStandaloneConfig,
    example_path: Path,
    *,
    run_id: str,
    experiment_id: str,
) -> Mapping[str, Any]:
    return asyncio.run(
        run_trae_example(
            config,
            example_path,
            run_id=run_id,
            experiment_id=experiment_id,
        )
    )
