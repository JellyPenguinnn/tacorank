"""Bounded Trae Agent adapter that seals exact Git patches and evidence.

The adapter owns coding mechanics only.  It does not evaluate a patch, promote
an experiment, append memory, or select recovery policy.  Shared handoff models
are injected (or loaded from ``tacorank.schemas``) so this module does not
redefine Person 2's schema authority.
"""

from __future__ import annotations

import asyncio
import email.parser
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple

import yaml  # type: ignore[import-untyped]

from tacorank.git.patches import (
    WrittenArtifact,
    commit_staged_patch,
    stage_and_capture,
    write_artifact,
)
from tacorank.git.refs import GitOperationError, require_ancestor, resolve_commit
from tacorank.docker_host import normalize_local_docker_host
from tacorank.git.worktrees import WorktreeManager, WorktreeRecord
from tacorank.run_layout import experiment_artifact_prefix

from .output_parser import ParsedTrajectory, TrajectoryParseError, parse_trajectory_file
from .prompts import build_coding_prompt, build_repair_prompt, prompt_sha256
from .redaction import SecretRedactor


_REVIEWED_TRAE_SOURCE_REVISION = "e839e559ac61bdd0e057c375dd1dee391fee797d"
_REVIEWED_TRAE_SOURCE_URL = "https://github.com/bytedance/trae-agent.git"
_REVIEWED_DOTENV_VERSION = "1.2.2"
TRAE_DEEPSEEK_REASONING_MARKER = (
    "TacoRank: force DeepSeek Responses reasoning effort and preserve reasoning items"
)
TRAE_DEEPSEEK_TOOL_JSON_MARKER = (
    "TacoRank: recover malformed DeepSeek tool arguments inside the bounded agent loop"
)
TRAE_DOCKER_EDIT_TOOL_MARKER = (
    "TacoRank: normalize and shell-quote command-specific edit arguments"
)
_PINNED_IMAGE_RE = re.compile(r"^(?:[^\s@]+@)?sha256:([0-9a-f]{64})$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_REVIEWED_TOOLS = (
    "str_replace_based_edit_tool",
    "task_done",
)
_FORBIDDEN_ENVIRONMENT_NAMES = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "HOME",
        "PATH",
        "SHELL",
        "SHELLOPTS",
        "TMP",
        "TEMP",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
    }
)
_FORBIDDEN_ENVIRONMENT_PREFIXES = ("PYTHON", "LD_", "DYLD_", "GIT_", "DOCKER_")


class CodingWorkerError(RuntimeError):
    """Safe classified failure returned across the coding adapter boundary."""

    def __init__(
        self,
        code: str,
        summary: str,
        *,
        output_tail: Optional[str] = None,
        resource_delta: Any = None,
        diagnostic_artifacts: Sequence[Any] = (),
    ) -> None:
        self.code = code
        self.summary = summary
        self.output_tail = output_tail
        self.resource_delta = resource_delta
        self.diagnostic_artifacts = tuple(diagnostic_artifacts)
        super().__init__(f"{code}: {summary}")


class SchemaIntegrationError(RuntimeError):
    """Shared Person 2 schema factories are missing or incomplete."""


@dataclass(frozen=True)
class CandidateIdentity:
    """Ledger identity allocated by Person 2 before a coding action."""

    attempt: int
    experiment_spec_event_id: str

    def __post_init__(self) -> None:
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("candidate attempt must be a positive integer")
        if (
            not isinstance(self.experiment_spec_event_id, str)
            or not self.experiment_spec_event_id.strip()
        ):
            raise ValueError("experiment_spec_event_id must be non-empty")


class CandidateIdentityResolver(Protocol):
    """Resolve controller-owned ledger identities without guessing defaults."""

    def for_initial(self, context: Any, spec: Any) -> CandidateIdentity:
        ...

    def for_repair(self, context: Any, decision: Any) -> CandidateIdentity:
        ...


@dataclass(frozen=True)
class SchemaFactories:
    """Construct Person 2's canonical shared models by keyword arguments."""

    artifact_ref: Callable[..., Any]
    resource_delta: Callable[..., Any]
    patch_candidate: Callable[..., Any]

    @classmethod
    def from_shared_schemas(cls) -> "SchemaFactories":
        """Load the canonical factories, failing explicitly while absent."""

        try:
            from tacorank import schemas
        except ImportError as exc:
            raise SchemaIntegrationError("cannot import tacorank.schemas") from exc
        missing = [
            name
            for name in ("ArtifactRef", "ResourceDelta", "PatchCandidate")
            if not callable(getattr(schemas, name, None))
        ]
        if missing:
            raise SchemaIntegrationError(
                "tacorank.schemas must define callable shared models: "
                + ", ".join(missing)
            )
        return cls(
            artifact_ref=getattr(schemas, "ArtifactRef"),
            resource_delta=getattr(schemas, "ResourceDelta"),
            patch_candidate=getattr(schemas, "PatchCandidate"),
        )


@dataclass(frozen=True)
class TraeConfig:
    """Operator-reviewed and pinned Trae invocation configuration."""

    command_prefix: Tuple[str, ...]
    trae_version: str
    provider: str
    model_id: str
    config_file: Path
    config_sha256: str
    max_steps_cap: int
    max_token_cap: Optional[int]
    max_wall_time_seconds_cap: int
    repair_step_limit: int
    repair_token_limit: Optional[int]
    repair_wall_time_limit_seconds: int
    repair_allowed_command_ids: Tuple[str, ...]
    approved_environment_names: Tuple[str, ...] = ()
    credential_environment_names: Tuple[str, ...] = ()
    credential_environment_aliases: Tuple[Tuple[str, str], ...] = ()
    provider_base_url: Optional[str] = None
    reasoning_effort: str = "high"
    trae_source_revision: Optional[str] = None
    trae_install_root: Optional[Path] = None
    trae_install_identity_file: Optional[Path] = None
    trae_install_identity_sha256: Optional[str] = None
    trae_executable_sha256: Optional[str] = None
    trae_runtime_root: Optional[Path] = None
    trae_runtime_manifest_sha256: Optional[str] = None
    python_dotenv_metadata_file: Optional[Path] = None
    python_dotenv_metadata_sha256: Optional[str] = None
    docker_image: Optional[str] = None
    docker_executable: Optional[Path] = None
    docker_host: Optional[str] = None
    trusted_test_mode: bool = False
    reviewed_tool_names: Tuple[str, ...] = _DEFAULT_REVIEWED_TOOLS
    docker_memory_limit_mb: int = 4096
    docker_pids_limit: int = 128
    docker_cpu_limit: float = 2.0
    docker_tmpfs_limit_mb: int = 256
    docker_agent_tools_size_limit_mb: int = 64
    docker_cli_timeout_seconds: int = 30
    version_timeout_seconds: int = 10
    termination_grace_seconds: int = 5
    max_process_output_bytes: int = 8 * 1024 * 1024
    max_trajectory_bytes: int = 50 * 1024 * 1024
    max_patch_bytes: int = 10 * 1024 * 1024
    worktree_lease_timeout_seconds: float = 30.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TraeConfig":
        """Normalize a JSON-shaped deployment mapping at the adapter boundary."""

        normalized = dict(values)
        normalized["command_prefix"] = tuple(normalized["command_prefix"])
        for key in (
            "config_file",
            "docker_executable",
            "trae_install_root",
            "trae_install_identity_file",
            "trae_runtime_root",
            "python_dotenv_metadata_file",
        ):
            if normalized.get(key) is not None:
                normalized[key] = Path(normalized[key])
        for key in (
            "repair_allowed_command_ids",
            "approved_environment_names",
            "credential_environment_names",
        ):
            if key in normalized:
                normalized[key] = tuple(normalized[key])
        if "credential_environment_aliases" in normalized:
            normalized["credential_environment_aliases"] = tuple(
                tuple(item) for item in normalized["credential_environment_aliases"]
            )
        return cls(**normalized)


@dataclass(frozen=True)
class _ProcessResult:
    exit_code: int
    timed_out: bool
    output_limited: bool
    output_tail: str


@dataclass(frozen=True)
class _IsolationSession:
    mode: str
    container_id: Optional[str]
    image: Optional[str]
    image_digest: Optional[str]


class TraeCodingWorker:
    """Real CodingWorker adapter for initial and repair patch creation.

    Trae's CLI exposes a hard step limit and its approved config controls the
    provider's per-response maximum.  The action-level provider-token budget is
    also stated in the prompt and is fail-closed against exact trajectory totals
    after the run; Trae currently exposes no live total-token cutoff flag.
    """

    def __init__(
        self,
        *,
        worktrees: WorktreeManager,
        artifact_repository_root: Path,
        config: TraeConfig,
        identity_resolver: CandidateIdentityResolver,
        factories: Optional[SchemaFactories] = None,
        process_environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.worktrees = worktrees
        self.artifact_repository_root = Path(artifact_repository_root).resolve(strict=True)
        if self.artifact_repository_root != self.worktrees.repository:
            raise CodingWorkerError(
                "ARTIFACT_ROOT_INVALID",
                "artifact_repository_root must be the controller repository root",
            )
        self.config = config
        self.identity_resolver = identity_resolver
        self._factories = factories
        self._source_environment = dict(
            os.environ if process_environment is None else process_environment
        )
        self._validate_config()
        self.redactor = SecretRedactor.from_environment(
            self._source_environment,
            self.config.credential_environment_names,
        )
        self._environment = self._sanitized_environment()
        self._verify_config_file()

    def preflight(self) -> None:
        """Verify credentials and all local production capabilities."""

        missing_credentials = [
            name
            for name in self.config.credential_environment_names
            if not self._source_environment.get(name, "").strip()
        ]
        if missing_credentials:
            raise CodingWorkerError(
                "TRAE_CREDENTIAL_MISSING",
                "required coding credential environment is missing: "
                + ", ".join(sorted(missing_credentials)),
            )
        self.preflight_local()

    def preflight_local(self) -> None:
        """Verify the pinned Trae and Docker boundary without using credentials."""

        self._schema_factories()
        self._verify_config_file()
        self._verify_install_identity()
        runtime_root, _ = self._verify_runtime_root()
        self._verify_cli_version(runtime_root)
        image = self.config.docker_image
        if image is None:
            raise CodingWorkerError(
                "TRAE_ISOLATION_REQUIRED", "production Docker image is unavailable"
            )
        match = _PINNED_IMAGE_RE.fullmatch(image)
        if match is None:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Docker image is not pinned by an exact digest"
            )
        with tempfile.TemporaryDirectory(prefix="tacorank-trae-preflight-") as temporary:
            docker_config_root = Path(temporary) / "docker-config"
            docker_config_root.mkdir(mode=0o700)
            inspected = self._run_docker_cli(
                ("image", "inspect", "--format", "{{.Id}}", image),
                docker_config_root=docker_config_root,
                check=True,
            )
            image_id = inspected.stdout.decode("ascii", errors="ignore").strip()
            if image_id != "sha256:" + match.group(1):
                raise CodingWorkerError(
                    "TRAE_IMAGE_IDENTITY_MISMATCH",
                    "Docker resolved a different image than the reviewed digest",
                )
            self._preflight_tool_mount(
                runtime_root,
                image,
                docker_config_root=docker_config_root,
            )

    async def create_patch(self, context: Any, spec: Any) -> Any:
        """Create an initial patch on a new experiment branch."""

        self._schema_factories()
        self._verify_install_identity()
        self._verify_runtime_root()
        identity = self._resolve_identity(
            self.identity_resolver.for_initial(context, spec)
        )
        self._validate_context_bounds(context)
        prompt = build_coding_prompt(context, spec, redactor=self.redactor)
        run_id = _context_text(context, "run_id")
        experiment_id = _context_text(context, "experiment_id")
        parent = _context_text(context, "parent_commit_sha")
        try:
            target = self.worktrees.path_for(run_id, experiment_id)
            if identity.attempt == 1:
                record = self.worktrees.create(run_id, experiment_id, parent)
            elif target.exists():
                record = self.worktrees.attach(
                    run_id, experiment_id, parent, require_clean=True
                )
            else:
                record = self.worktrees.create(
                    run_id,
                    experiment_id,
                    parent,
                    reuse_existing_branch=True,
                )
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc
        return await asyncio.to_thread(
            self._produce_candidate,
            context,
            record,
            parent,
            identity,
            prompt,
            _context_int(context, "step_limit"),
            _context_optional_positive_int(context, "token_limit"),
            _context_int(context, "wall_time_limit_seconds"),
        )

    async def repair_patch(self, context: Any, decision: Any) -> Any:
        """Create a direct repair commit on the same experiment branch."""

        self._schema_factories()
        self._verify_install_identity()
        self._verify_runtime_root()
        identity = self._resolve_identity(
            self.identity_resolver.for_repair(context, decision)
        )
        repair_attempt = _context_int(context, "repair_attempt")
        if identity.attempt != repair_attempt + 1:
            raise CodingWorkerError(
                "CANDIDATE_IDENTITY_MISMATCH",
                "resolved coding attempt does not follow the recovery attempt",
            )
        current_commit = _context_text(context, "current_patch_commit_sha")
        original_spec = getattr(context, "original_experiment_spec", None)
        if original_spec is None:
            raise CodingWorkerError(
                "RECOVERY_CONTEXT_INVALID", "missing original_experiment_spec"
            )
        parent = _model_field(original_spec, "parent_commit_sha")
        try:
            require_ancestor(self.worktrees.repository, parent, current_commit)
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc
        prompt = build_repair_prompt(
            context,
            decision,
            step_limit=self.config.repair_step_limit,
            token_limit=self.config.repair_token_limit,
            wall_time_limit_seconds=self.config.repair_wall_time_limit_seconds,
            allowed_command_ids=self.config.repair_allowed_command_ids,
            redactor=self.redactor,
        )
        run_id = _context_text(context, "run_id")
        experiment_id = _context_text(context, "experiment_id")
        target = self.worktrees.path_for(run_id, experiment_id)
        try:
            if target.exists():
                record = self.worktrees.attach(
                    run_id, experiment_id, current_commit, require_clean=True
                )
            else:
                record = self.worktrees.create(
                    run_id,
                    experiment_id,
                    parent,
                    reuse_existing_branch=True,
                )
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc
        if not target.exists():
            raise CodingWorkerError(
                "REPAIR_WORKTREE_MISSING", "repair worktree was not created"
            )
        if record.commit_sha != current_commit:
            raise CodingWorkerError(
                "REPAIR_COMMIT_MISMATCH",
                "experiment branch tip differs from recovery context commit",
            )
        return await asyncio.to_thread(
            self._produce_candidate,
            context,
            record,
            current_commit,
            identity,
            prompt,
            self.config.repair_step_limit,
            self.config.repair_token_limit,
            self.config.repair_wall_time_limit_seconds,
        )

    def _produce_candidate(
        self,
        context: Any,
        record: WorktreeRecord,
        base_commit_sha: str,
        identity: CandidateIdentity,
        prompt: str,
        step_limit: int,
        token_limit: Optional[int],
        wall_time_limit_seconds: int,
    ) -> Any:
        try:
            with self.worktrees.acquire_lease(
                record,
                timeout_seconds=self.config.worktree_lease_timeout_seconds,
            ):
                return self._produce_candidate_with_lease(
                    context,
                    record,
                    base_commit_sha,
                    identity,
                    prompt,
                    step_limit,
                    token_limit,
                    wall_time_limit_seconds,
                )
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc

    def _produce_candidate_with_lease(
        self,
        context: Any,
        record: WorktreeRecord,
        base_commit_sha: str,
        identity: CandidateIdentity,
        prompt: str,
        step_limit: int,
        token_limit: Optional[int],
        wall_time_limit_seconds: int,
    ) -> Any:
        started = time.monotonic()
        try:
            base = resolve_commit(record.path, base_commit_sha)
            self.worktrees.verify(
                record, expected_commit_sha=base, require_clean=True
            )
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc
        self._verify_config_file()
        install_identity = self._verify_install_identity()
        runtime_root, runtime_identity = self._verify_runtime_root()
        self._verify_cli_version(runtime_root)
        artifact_prefix = experiment_artifact_prefix(
            record.run_id,
            record.experiment_id,
            attempt=identity.attempt,
        )

        with tempfile.TemporaryDirectory(prefix="tacorank-trae-") as temporary:
            temporary_root = Path(temporary)
            prompt_path = temporary_root / "task.md"
            prompt_path.write_text(prompt, encoding="utf-8")
            prompt_path.chmod(0o600)
            raw_trajectory_path = temporary_root / "trajectory.json"
            process_output_path = temporary_root / "process.log"
            docker_config_root = temporary_root / "docker-config"
            docker_config_root.mkdir(mode=0o700)
            action_environment = self._action_environment(
                temporary_root,
                docker_config_root=docker_config_root,
            )
            isolation = self._start_isolation(
                record,
                identity,
                docker_config_root=docker_config_root,
            )
            try:
                command = self._build_command(
                    prompt_path,
                    record.path,
                    raw_trajectory_path,
                    step_limit,
                    isolation,
                )
                process = _run_bounded_process(
                    command,
                    cwd=runtime_root,
                    environment=action_environment,
                    timeout_seconds=wall_time_limit_seconds,
                    termination_grace_seconds=self.config.termination_grace_seconds,
                    output_path=process_output_path,
                    max_output_bytes=self.config.max_process_output_bytes,
                    redactor=self.redactor,
                )
                if process.timed_out:
                    raise self._process_failure(
                        "TRAE_TIMEOUT",
                        f"Trae exceeded the {wall_time_limit_seconds}s wall limit",
                        output_tail=process.output_tail,
                        started=started,
                        process_output_path=process_output_path,
                        artifact_prefix=artifact_prefix,
                    )
                if process.output_limited:
                    raise self._process_failure(
                        "TRAE_OUTPUT_LIMIT",
                        "Trae process output exceeded its hard byte limit",
                        output_tail=process.output_tail,
                        started=started,
                        process_output_path=process_output_path,
                        artifact_prefix=artifact_prefix,
                    )
                if process.exit_code != 0:
                    raise self._process_failure(
                        "TRAE_PROCESS_FAILED",
                        f"Trae exited with code {process.exit_code}",
                        output_tail=process.output_tail,
                        started=started,
                        process_output_path=process_output_path,
                        artifact_prefix=artifact_prefix,
                    )

                try:
                    parsed = parse_trajectory_file(
                        raw_trajectory_path,
                        max_bytes=self.config.max_trajectory_bytes,
                    )
                except TrajectoryParseError as exc:
                    raise self._process_failure(
                        exc.code,
                        str(exc),
                        output_tail=process.output_tail,
                        started=started,
                        process_output_path=process_output_path,
                        artifact_prefix=artifact_prefix,
                    ) from exc
                trajectory_bytes = self._redacted_trajectory_bytes(
                    parsed,
                    prompt,
                    token_limit=token_limit,
                    isolation=isolation,
                    install_identity=install_identity,
                    runtime_identity=runtime_identity,
                )
                try:
                    self._validate_trajectory(parsed, step_limit, token_limit)
                except CodingWorkerError as exc:
                    try:
                        failure_written = write_artifact(
                            self.artifact_repository_root,
                            f"{artifact_prefix}/trae_failure_trajectory.json",
                            trajectory_bytes,
                            content_type="application/json",
                        )
                        process_written = self._retain_process_log(
                            process_output_path,
                            f"{artifact_prefix}/trae_process.log",
                        )
                    except (
                        CodingWorkerError,
                        GitOperationError,
                        OSError,
                    ) as artifact_error:
                        raise CodingWorkerError(
                            "TRAE_FAILURE_EVIDENCE_WRITE_FAILED",
                            "validated Trae failure evidence could not be retained",
                            resource_delta=self._resource_delta(started, parsed),
                        ) from artifact_error
                    diagnostic = (exc.output_tail or "").strip()
                    raise CodingWorkerError(
                        exc.code,
                        exc.summary,
                        output_tail=diagnostic or None,
                        resource_delta=self._resource_delta(started, parsed),
                        diagnostic_artifacts=(
                            self._artifact_ref(failure_written, "trajectory"),
                            self._artifact_ref(process_written, "log"),
                        ),
                    ) from exc
            except BaseException as primary_error:
                try:
                    self._close_isolation(
                        isolation,
                        docker_config_root=docker_config_root,
                    )
                except CodingWorkerError as cleanup_error:
                    raise cleanup_error from primary_error
                raise
            else:
                self._close_isolation(
                    isolation,
                    docker_config_root=docker_config_root,
                )

        try:
            self.worktrees.verify(
                record, expected_commit_sha=base, require_clean=False
            )
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc
        try:
            staged = stage_and_capture(
                record.path,
                base,
                max_diff_bytes=self.config.max_patch_bytes,
            )
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc
        if staged.is_empty:
            raise CodingWorkerError("NO_PATCH", "must-patch task produced no Git diff")
        if len(staged.diff) > self.config.max_patch_bytes:
            raise CodingWorkerError(
                "PATCH_TOO_LARGE",
                "candidate patch exceeds the configured artifact byte limit",
            )
        if self.redactor.contains_known_secret(staged.diff):
            raise CodingWorkerError(
                "CREDENTIAL_IN_PATCH",
                "candidate patch contains a credential from the approved environment",
            )
        try:
            sealed = commit_staged_patch(
                record.path,
                staged,
                message=(
                    f"tacorank: {record.run_id}/{record.experiment_id} "
                    f"attempt {identity.attempt}"
                ),
            )
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc

        try:
            diff_written = write_artifact(
                self.artifact_repository_root,
                f"{artifact_prefix}/patch.diff",
                sealed.diff,
                content_type="text/x-diff",
            )
            trajectory_written = write_artifact(
                self.artifact_repository_root,
                f"{artifact_prefix}/trae_trajectory.json",
                trajectory_bytes,
                content_type="application/json",
            )
        except GitOperationError as exc:
            raise CodingWorkerError(exc.code, str(exc)) from exc

        diff_artifact = self._artifact_ref(diff_written, "diff")
        trajectory_artifact = self._artifact_ref(trajectory_written, "trajectory")
        factories = self._schema_factories()
        resource_delta = self._resource_delta(started, parsed)
        return factories.patch_candidate(
            schema_version="1.0",
            run_id=record.run_id,
            experiment_id=record.experiment_id,
            attempt=identity.attempt,
            experiment_spec_event_id=identity.experiment_spec_event_id,
            context_id=_context_text(context, "context_id"),
            base_commit_sha=sealed.base_commit_sha,
            patch_commit_sha=sealed.patch_commit_sha,
            diff_sha256=sealed.diff_sha256,
            changed_files=list(sealed.changed_files),
            diff_artifact=diff_artifact,
            trajectory_artifact=trajectory_artifact,
            trae_version=(
                self.config.trae_source_revision or self.config.trae_version
            ),
            model_id=self.config.model_id,
            steps_used=parsed.steps_used,
            resource_delta=resource_delta,
        )

    def _resource_delta(
        self, started: float, parsed: Optional[ParsedTrajectory] = None
    ) -> Any:
        usage = parsed.usage if parsed is not None else None
        return self._schema_factories().resource_delta(
            llm_input_tokens=usage.input_tokens if usage is not None else 0,
            llm_output_tokens=usage.output_tokens if usage is not None else 0,
            token_measurement="provider" if usage is not None else "none",
            wall_time_ms=max(0, int((time.monotonic() - started) * 1000)),
            cpu_time_ms=0,
            gpu_time_ms=0,
            gpu_count=0,
            peak_rss_mb=None,
            peak_gpu_memory_mb=None,
            manual_interventions=0,
        )

    def _retain_process_log(self, source: Path, relative_path: str) -> WrittenArtifact:
        try:
            raw = source.read_bytes()
        except OSError as error:
            raise CodingWorkerError(
                "TRAE_FAILURE_EVIDENCE_WRITE_FAILED",
                "Trae process output could not be retained",
            ) from error
        if len(raw) > self.config.max_process_output_bytes:
            raw = raw[-self.config.max_process_output_bytes :]
        rendered = self.redactor.redact(
            raw.decode("utf-8", errors="replace")
        ).encode("utf-8")
        if self.redactor.contains_known_secret(rendered):
            raise CodingWorkerError(
                "TRAE_FAILURE_EVIDENCE_WRITE_FAILED",
                "Trae process output redaction failed",
            )
        return write_artifact(
            self.artifact_repository_root,
            relative_path,
            rendered,
            content_type="text/plain; charset=utf-8",
        )

    def _process_failure(
        self,
        code: str,
        summary: str,
        *,
        output_tail: str,
        started: float,
        process_output_path: Path,
        artifact_prefix: str,
    ) -> CodingWorkerError:
        try:
            written = self._retain_process_log(
                process_output_path,
                f"{artifact_prefix}/trae_process.log",
            )
        except (CodingWorkerError, GitOperationError, OSError):
            return CodingWorkerError(
                "TRAE_FAILURE_EVIDENCE_WRITE_FAILED",
                "Trae process failure evidence could not be retained",
                resource_delta=self._resource_delta(started),
            )
        reference = self._artifact_ref(written, "log")
        return CodingWorkerError(
            code,
            summary,
            output_tail=output_tail.strip() or None,
            resource_delta=self._resource_delta(started),
            diagnostic_artifacts=(reference,),
        )

    def _artifact_ref(self, artifact: WrittenArtifact, kind: str) -> Any:
        return self._schema_factories().artifact_ref(
            artifact_id=f"sha256-{artifact.sha256}",
            kind=kind,
            path=artifact.path,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            content_type=artifact.content_type,
        )

    def _redacted_trajectory_bytes(
        self,
        parsed: ParsedTrajectory,
        prompt: str,
        *,
        token_limit: Optional[int],
        isolation: _IsolationSession,
        install_identity: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
    ) -> bytes:
        document = self.redactor.redact_object(parsed.raw)
        if not isinstance(document, dict):
            raise CodingWorkerError(
                "TRAJECTORY_MALFORMED", "redacted trajectory root is not an object"
            )
        if "tacorank_adapter" in document:
            raise CodingWorkerError(
                "TRAJECTORY_RESERVED_KEY",
                "Trae trajectory uses reserved tacorank_adapter metadata key",
            )
        document["tacorank_adapter"] = {
            "schema_version": "1.0",
            "trae_version": self.config.trae_version,
            "trae_source_revision": self.config.trae_source_revision,
            "trae_install_identity": dict(install_identity),
            "trae_runtime_identity": dict(runtime_identity),
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "reasoning_effort": self.config.reasoning_effort,
            "config_sha256": self.config.config_sha256,
            "prompt_sha256": prompt_sha256(prompt),
            "max_steps": parsed.max_steps,
            "max_provider_tokens": token_limit,
            "isolation_mode": isolation.mode,
            "docker_mode": isolation.mode == "hardened_docker",
            "docker_image": isolation.image,
            "docker_image_digest": isolation.image_digest,
            "docker_attach_contract": (
                "reviewed-host-project-path-to-mounted-workspace"
                if isolation.mode == "hardened_docker"
                else None
            ),
            "redacted": True,
        }
        rendered = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        result = (rendered + "\n").encode("utf-8")
        if len(result) > self.config.max_trajectory_bytes:
            raise CodingWorkerError(
                "TRAJECTORY_TOO_LARGE",
                "redacted trajectory exceeds the configured artifact size limit",
            )
        if self.redactor.contains_known_secret(result):
            raise CodingWorkerError(
                "TRAJECTORY_REDACTION_FAILED", "known credential remained in trajectory"
            )
        return result

    def _build_command(
        self,
        prompt_path: Path,
        worktree: Path,
        trajectory_path: Path,
        step_limit: int,
        isolation: _IsolationSession,
    ) -> Tuple[str, ...]:
        args = [
            *self.config.command_prefix,
            "run",
            "--file",
            str(prompt_path),
            "--provider",
            self.config.provider,
            "--model",
            self.config.model_id,
            "--max-steps",
            str(step_limit),
            "--working-dir",
            str(worktree),
            "--must-patch",
            "--config-file",
            str(Path(self.config.config_file).resolve(strict=True)),
            "--trajectory-file",
            str(trajectory_path),
            "--console-type",
            "simple",
        ]
        if isolation.container_id is not None:
            args.extend(
                (
                    "--docker-container-id",
                    isolation.container_id,
                    "--docker-keep",
                    "False",
                )
            )
        return tuple(args)

    def _start_isolation(
        self,
        record: WorktreeRecord,
        identity: CandidateIdentity,
        *,
        docker_config_root: Path,
    ) -> _IsolationSession:
        if self.config.trusted_test_mode:
            return _IsolationSession(
                mode="trusted_test",
                container_id=None,
                image=None,
                image_digest=None,
            )

        image = self.config.docker_image
        docker_executable = self.config.docker_executable
        if image is None or docker_executable is None:
            raise CodingWorkerError(
                "TRAE_ISOLATION_REQUIRED",
                "production Trae requires a reviewed hardened Docker boundary",
            )
        image_match = _PINNED_IMAGE_RE.fullmatch(image)
        if image_match is None:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Docker image is not pinned by an exact digest"
            )
        asset_source = self._reviewed_tool_mount_source()

        worktree = record.path.resolve(strict=True)
        if any(character in str(worktree) for character in (",", "\n", "\r", "\x00")):
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED",
                "worktree path cannot be represented by the reviewed Docker mount",
            )
        identity_material = (
            f"{record.run_id}\0{record.experiment_id}\0{identity.attempt}\0"
            f"{record.commit_sha}"
        ).encode("utf-8")
        suffix = hashlib.sha256(identity_material).hexdigest()[:20]
        name = f"tacorank-trae-{suffix}"
        cidfile = docker_config_root / "container.cid"
        cpu_limit = format(self.config.docker_cpu_limit, ".3f").rstrip("0").rstrip(".")
        memory = f"{self.config.docker_memory_limit_mb}m"
        nested_admin_mounts: list[str] = []
        for submodule in self.worktrees.required_submodules:
            if "," in submodule:
                raise CodingWorkerError(
                    "TRAE_ISOLATION_SETUP_FAILED",
                    "submodule path cannot be represented by the reviewed Docker mount",
                )
            nested_admin_mounts.extend(
                (
                    "--mount",
                    (
                        f"type=bind,src={worktree / submodule / '.git'},"
                        f"dst=/workspace/{submodule}/.git,readonly,"
                        "bind-propagation=rprivate"
                    ),
                )
            )
        create_args = (
            "create",
            "--cidfile",
            str(cidfile),
            "--name",
            name,
            "--label",
            "com.tacorank.owner=coding-worker",
            "--label",
            f"com.tacorank.run-id={record.run_id}",
            "--label",
            f"com.tacorank.experiment-id={record.experiment_id}",
            "--label",
            f"com.tacorank.attempt={identity.attempt}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            memory,
            "--memory-swap",
            memory,
            "--cpus",
            cpu_limit,
            "--pids-limit",
            str(self.config.docker_pids_limit),
            "--tmpfs",
            (
                "/tmp:rw,noexec,nosuid,nodev,"
                f"size={self.config.docker_tmpfs_limit_mb}m,mode=1777"
            ),
            "--mount",
            (
                f"type=bind,src={asset_source},dst=/agent_tools,"
                "readonly,bind-propagation=rprivate"
            ),
            "--mount",
            (
                f"type=bind,src={worktree},dst=/workspace,"
                "bind-propagation=rprivate"
            ),
            "--mount",
            (
                f"type=bind,src={worktree / '.git'},dst=/workspace/.git,"
                "readonly,bind-propagation=rprivate"
            ),
            *nested_admin_mounts,
            "--workdir",
            "/workspace",
            "--user",
            _container_user(),
            "--init",
            "--ulimit",
            "nofile=256:256",
            "--pull",
            "never",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "exec sleep infinity",
        )
        try:
            created = self._run_docker_cli(
                create_args,
                docker_config_root=docker_config_root,
                check=False,
            )
        except CodingWorkerError as exc:
            cidfile_id = self._read_container_id(cidfile)
            if cidfile_id is not None:
                leaked_session = _IsolationSession(
                    mode="hardened_docker",
                    container_id=cidfile_id,
                    image=image,
                    image_digest=f"sha256:{image_match.group(1)}",
                )
                try:
                    self._close_isolation(
                        leaked_session,
                        docker_config_root=docker_config_root,
                    )
                except CodingWorkerError as cleanup_error:
                    raise cleanup_error from exc
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED",
                "hardened Docker container creation failed",
            ) from exc
        raw_id = created.stdout.decode("ascii", errors="ignore").strip()
        cidfile_id = self._read_container_id(cidfile) or ""
        usable_id = cidfile_id if _CONTAINER_ID_RE.fullmatch(cidfile_id) else raw_id
        if (
            created.returncode != 0
            or _CONTAINER_ID_RE.fullmatch(raw_id) is None
            or raw_id != cidfile_id
        ):
            if _CONTAINER_ID_RE.fullmatch(usable_id):
                leaked_session = _IsolationSession(
                    mode="hardened_docker",
                    container_id=usable_id,
                    image=image,
                    image_digest=f"sha256:{image_match.group(1)}",
                )
                try:
                    self._close_isolation(
                        leaked_session,
                        docker_config_root=docker_config_root,
                    )
                except CodingWorkerError as cleanup_error:
                    raise cleanup_error
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED",
                "hardened Docker container creation failed",
            )
        session = _IsolationSession(
            mode="hardened_docker",
            container_id=raw_id,
            image=image,
            image_digest=f"sha256:{image_match.group(1)}",
        )
        try:
            started = self._run_docker_cli(
                ("start", raw_id),
                docker_config_root=docker_config_root,
                check=False,
            )
        except CodingWorkerError as exc:
            try:
                self._close_isolation(
                    session,
                    docker_config_root=docker_config_root,
                )
            except CodingWorkerError as cleanup_error:
                raise cleanup_error from exc
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED",
                "hardened Docker container failed to start",
            ) from exc
        if started.returncode != 0:
            try:
                self._close_isolation(
                    session,
                    docker_config_root=docker_config_root,
                )
            except CodingWorkerError as cleanup_error:
                raise cleanup_error
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED",
                "hardened Docker container failed to start",
            )
        return session

    def _reviewed_tool_mount_source(
        self, runtime_root: Optional[Path] = None
    ) -> Path:
        configured_root = runtime_root or self.config.trae_runtime_root
        if configured_root is None:
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED", "Trae runtime assets are unavailable"
            )
        try:
            source = (
                Path(configured_root) / "trae_agent" / "dist"
            ).resolve(strict=True)
        except OSError as exc:
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED", "Trae runtime assets are unavailable"
            ) from exc
        if not source.is_dir() or any(
            character in str(source) for character in (",", "\n", "\r", "\x00")
        ):
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED",
                "Trae tool path cannot be represented by the reviewed Docker mount",
            )
        return source

    def _preflight_tool_mount(
        self,
        runtime_root: Path,
        image: str,
        *,
        docker_config_root: Path,
    ) -> None:
        """Execute a reviewed tool through the production read-only mount."""

        source = self._reviewed_tool_mount_source(runtime_root)
        cidfile = docker_config_root / "tool-mount.cid"
        suffix = hashlib.sha256(str(docker_config_root).encode("utf-8")).hexdigest()[:20]
        name = "tacorank-trae-preflight-" + suffix
        create_args = (
            "create",
            "--cidfile",
            str(cidfile),
            "--name",
            name,
            "--label",
            "com.tacorank.owner=coding-worker-preflight",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            "256m",
            "--memory-swap",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "32",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
            "--mount",
            (
                f"type=bind,src={source},dst=/agent_tools,"
                "readonly,bind-propagation=rprivate"
            ),
            "--user",
            _container_user(),
            "--pull",
            "never",
            "--entrypoint",
            "/agent_tools/edit_tool",
            image,
            "--help",
        )
        try:
            created = self._run_docker_cli(
                create_args,
                docker_config_root=docker_config_root,
                check=False,
            )
        except CodingWorkerError as exc:
            container_id = self._read_container_id(cidfile)
            if container_id is not None:
                try:
                    self._close_isolation(
                        _IsolationSession(
                            mode="preflight",
                            container_id=container_id,
                            image=image,
                            image_digest=None,
                        ),
                        docker_config_root=docker_config_root,
                    )
                except CodingWorkerError as cleanup_error:
                    raise cleanup_error from exc
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED",
                "reviewed Trae tool mount could not be created",
            ) from exc
        raw_id = created.stdout.decode("ascii", errors="ignore").strip()
        cidfile_id = self._read_container_id(cidfile) or ""
        usable_id = cidfile_id if _CONTAINER_ID_RE.fullmatch(cidfile_id) else raw_id
        if (
            created.returncode != 0
            or _CONTAINER_ID_RE.fullmatch(raw_id) is None
            or raw_id != cidfile_id
        ):
            if _CONTAINER_ID_RE.fullmatch(usable_id):
                self._close_isolation(
                    _IsolationSession(
                        mode="preflight",
                        container_id=usable_id,
                        image=image,
                        image_digest=None,
                    ),
                    docker_config_root=docker_config_root,
                )
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED",
                "reviewed Trae tool mount could not be created",
            )
        session = _IsolationSession(
            mode="preflight",
            container_id=raw_id,
            image=image,
            image_digest=None,
        )
        try:
            executed = self._run_docker_cli(
                ("start", "--attach", raw_id),
                docker_config_root=docker_config_root,
                check=False,
            )
            if executed.returncode != 0:
                raise CodingWorkerError(
                    "TRAE_ISOLATION_SETUP_FAILED",
                    "reviewed Trae tools are not executable in the Docker boundary",
                )
        finally:
            self._close_isolation(
                session,
                docker_config_root=docker_config_root,
            )

    def _close_isolation(
        self,
        session: _IsolationSession,
        *,
        docker_config_root: Path,
    ) -> None:
        container_id = session.container_id
        if container_id is None:
            return
        try:
            self._run_docker_cli(
                (
                    "stop",
                    "--time",
                    str(self.config.termination_grace_seconds),
                    container_id,
                ),
                docker_config_root=docker_config_root,
                check=False,
            )
        except CodingWorkerError:
            # Removal is still mandatory and authoritative absence is checked below.
            pass
        try:
            self._run_docker_cli(
                ("rm", "--force", "--volumes", container_id),
                docker_config_root=docker_config_root,
                check=False,
            )
        except CodingWorkerError:
            # An interrupted remove may still have succeeded; inspect decides.
            pass
        try:
            inspected = self._run_docker_cli(
                ("inspect", "--type", "container", container_id),
                docker_config_root=docker_config_root,
                check=False,
            )
        except CodingWorkerError as exc:
            raise CodingWorkerError(
                "TRAE_ISOLATION_CLEANUP_FAILED",
                "container absence could not be verified after removal",
            ) from exc
        inspect_error = inspected.stderr.decode("utf-8", errors="replace").lower()
        if inspected.returncode == 0 or not any(
            marker in inspect_error
            for marker in ("no such object", "no such container")
        ):
            raise CodingWorkerError(
                "TRAE_ISOLATION_CLEANUP_FAILED",
                "container absence could not be verified after removal",
            )

    @staticmethod
    def _read_container_id(cidfile: Path) -> Optional[str]:
        try:
            if (
                not cidfile.is_file()
                or cidfile.is_symlink()
                or cidfile.stat().st_size > 128
            ):
                return None
            value = cidfile.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            return None
        return value if _CONTAINER_ID_RE.fullmatch(value) else None

    def _run_docker_cli(
        self,
        args: Sequence[str],
        *,
        docker_config_root: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        executable = self.config.docker_executable
        if executable is None:
            raise CodingWorkerError(
                "TRAE_ISOLATION_REQUIRED", "Docker executable is not configured"
            )
        docker_path = str(executable.parent)
        environment = {
            "PATH": (
                os.pathsep.join((docker_path, os.environ.get("PATH", "")))
                if os.name == "nt"
                else f"{docker_path}:/usr/bin:/bin"
            ),
            "HOME": str(docker_config_root),
            "DOCKER_CONFIG": str(docker_config_root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        if self.config.docker_host is not None:
            environment["DOCKER_HOST"] = self.config.docker_host
        try:
            result = subprocess.run(
                (str(executable), *args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                shell=False,
                env=environment,
                timeout=self.config.docker_cli_timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED", "Docker isolation command failed"
            ) from exc
        if check and result.returncode != 0:
            raise CodingWorkerError(
                "TRAE_ISOLATION_SETUP_FAILED", "Docker isolation command was rejected"
            )
        return result

    def _verify_cli_version(self, runtime_root: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="tacorank-trae-version-") as temporary:
            temporary_root = Path(temporary)
            environment = self._action_environment(temporary_root)
            result = _run_bounded_process(
                (*self.config.command_prefix, "--version"),
                cwd=runtime_root,
                environment=environment,
                timeout_seconds=self.config.version_timeout_seconds,
                termination_grace_seconds=self.config.termination_grace_seconds,
                output_path=temporary_root / "version.log",
                max_output_bytes=64 * 1024,
                redactor=self.redactor,
            )
        if result.timed_out or result.output_limited or result.exit_code != 0:
            raise CodingWorkerError(
                "TRAE_VERSION_CHECK_FAILED", "unable to verify the pinned Trae version"
            )
        words = result.output_tail.strip().replace(",", " ").split()
        if self.config.trae_version not in words:
            raise CodingWorkerError(
                "TRAE_VERSION_MISMATCH",
                "Trae CLI version output does not match the pinned version",
            )

    def _validate_trajectory(
        self,
        parsed: ParsedTrajectory,
        step_limit: int,
        token_limit: Optional[int],
    ) -> None:
        if parsed.provider != self.config.provider or parsed.model != self.config.model_id:
            raise CodingWorkerError(
                "TRAE_IDENTITY_MISMATCH",
                "trajectory provider/model differs from the pinned configuration",
            )
        if parsed.max_steps != step_limit or parsed.steps_used > step_limit:
            raise CodingWorkerError(
                "STEP_LIMIT_MISMATCH", "trajectory step limit differs from the request"
            )
        total_tokens = parsed.usage.input_tokens + parsed.usage.output_tokens
        if token_limit is not None and total_tokens > token_limit:
            raise CodingWorkerError(
                "TOKEN_LIMIT_EXCEEDED",
                f"provider usage {total_tokens} exceeded the {token_limit} token limit",
            )
        if not parsed.success:
            detail = self.redactor.redact(parsed.final_result or "").strip()
            detail = "".join(
                character
                if character in {"\n", "\t"} or ord(character) >= 32
                else " "
                for character in detail
            )[:2_048]
            raise CodingWorkerError(
                "TRAE_REPORTED_FAILURE",
                "Trae trajectory reports an unsuccessful task",
                output_tail=detail or None,
            )

    def _validate_context_bounds(self, context: Any) -> None:
        limits = (
            ("step_limit", self.config.max_steps_cap),
            ("wall_time_limit_seconds", self.config.max_wall_time_seconds_cap),
        )
        for field, cap in limits:
            value = _context_int(context, field)
            if value < 1 or value > cap:
                raise CodingWorkerError(
                    "CODING_LIMIT_INVALID", f"{field} must be in [1, {cap}]"
                )
        token_limit = _context_optional_positive_int(context, "token_limit")
        if (
            token_limit is not None
            and self.config.max_token_cap is not None
            and token_limit > self.config.max_token_cap
        ):
            raise CodingWorkerError(
                "CODING_LIMIT_INVALID",
                "token_limit exceeds the reviewed coding token cap",
            )

    def _validate_config(self) -> None:
        config = self.config
        if not config.command_prefix or any(
            not isinstance(part, str) or not part or "\x00" in part
            for part in config.command_prefix
        ):
            raise CodingWorkerError("TRAE_CONFIG_INVALID", "invalid command_prefix")
        executable = Path(config.command_prefix[0])
        if (
            not executable.is_absolute()
            or not executable.is_file()
            or executable.is_symlink()
            or executable.resolve(strict=True) != executable
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID",
                "command_prefix executable must be an absolute regular non-symlink file",
            )
        for field in ("trae_version", "provider", "model_id"):
            value = getattr(config, field)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise CodingWorkerError("TRAE_CONFIG_INVALID", f"invalid {field}")
        if (
            config.provider_base_url == "https://api.deepseek.com"
            and config.reasoning_effort != "high"
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID",
                "the reviewed DeepSeek coding path requires high reasoning effort",
            )
        if (
            not isinstance(config.config_sha256, str)
            or len(config.config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in config.config_sha256)
        ):
            raise CodingWorkerError("TRAE_CONFIG_INVALID", "invalid config_sha256")
        if len(set(config.approved_environment_names)) != len(
            config.approved_environment_names
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "approved environment names contain duplicates"
            )
        if not set(config.credential_environment_names).issubset(
            config.approved_environment_names
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID",
                "credential names must be explicitly approved environment names",
            )
        if any(
            not isinstance(item, (tuple, list)) or len(item) != 2
            for item in config.credential_environment_aliases
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "invalid credential environment alias"
            )
        alias_sources = [source for source, _ in config.credential_environment_aliases]
        alias_targets = [target for _, target in config.credential_environment_aliases]
        if (
            len(alias_sources) != len(set(alias_sources))
            or len(alias_targets) != len(set(alias_targets))
            or not set(alias_sources).issubset(config.credential_environment_names)
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID",
                "credential environment aliases must be unique approved credentials",
            )
        for source, target in config.credential_environment_aliases:
            if any(
                not isinstance(name, str)
                or re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) is None
                for name in (source, target)
            ):
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID", "invalid credential environment alias"
                )
        if config.provider_base_url is not None:
            if (
                not isinstance(config.provider_base_url, str)
                or re.fullmatch(r"https://[^\s/?#]+(?::[0-9]+)?", config.provider_base_url)
                is None
            ):
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID", "provider base URL must be an HTTPS origin"
                )
        if not config.trusted_test_mode:
            forbidden = sorted(
                name
                for name in config.approved_environment_names
                if name in _FORBIDDEN_ENVIRONMENT_NAMES
                or name.startswith(_FORBIDDEN_ENVIRONMENT_PREFIXES)
            )
            if forbidden:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID",
                    "approved environment includes a code-loader/runtime injection variable",
                )
        integer_fields = (
            "max_steps_cap",
            "max_wall_time_seconds_cap",
            "repair_step_limit",
            "repair_wall_time_limit_seconds",
            "version_timeout_seconds",
            "termination_grace_seconds",
            "max_process_output_bytes",
            "max_trajectory_bytes",
            "max_patch_bytes",
        )
        for field in integer_fields:
            value = getattr(config, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CodingWorkerError("TRAE_CONFIG_INVALID", f"invalid {field}")
        for field in ("max_token_cap", "repair_token_limit"):
            value = getattr(config, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise CodingWorkerError("TRAE_CONFIG_INVALID", f"invalid {field}")
        if config.repair_step_limit > config.max_steps_cap:
            raise CodingWorkerError("TRAE_CONFIG_INVALID", "repair step limit exceeds cap")
        if (
            config.max_token_cap is not None
            and (
                config.repair_token_limit is None
                or config.repair_token_limit > config.max_token_cap
            )
        ):
            raise CodingWorkerError("TRAE_CONFIG_INVALID", "repair token limit exceeds cap")
        if config.repair_wall_time_limit_seconds > config.max_wall_time_seconds_cap:
            raise CodingWorkerError("TRAE_CONFIG_INVALID", "repair wall limit exceeds cap")
        if not config.repair_allowed_command_ids:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "repair command allowlist cannot be empty"
            )
        if not isinstance(config.trusted_test_mode, bool):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "trusted_test_mode must be a boolean"
            )
        if config.reviewed_tool_names != _DEFAULT_REVIEWED_TOOLS:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID",
                "production Trae tools must be exactly edit and task_done",
            )
        if config.trusted_test_mode:
            if config.docker_image is not None or config.docker_executable is not None:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID",
                    "trusted test mode cannot be combined with production Docker settings",
                )
        else:
            if config.trae_source_revision != _REVIEWED_TRAE_SOURCE_REVISION:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID",
                    "production Docker attachment requires the reviewed Trae source revision",
                )
            if len(config.command_prefix) != 1:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID",
                    "production command_prefix must name only the pinned Trae executable",
                )
            if any(
                value is None
                for value in (
                    config.trae_install_root,
                    config.trae_install_identity_file,
                    config.trae_install_identity_sha256,
                    config.trae_executable_sha256,
                    config.trae_runtime_root,
                    config.trae_runtime_manifest_sha256,
                    config.python_dotenv_metadata_file,
                    config.python_dotenv_metadata_sha256,
                )
            ):
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID", "production Trae install identity is required"
                )
            install_root = Path(config.trae_install_root)  # type: ignore[arg-type]
            identity_path = Path(config.trae_install_identity_file)  # type: ignore[arg-type]
            runtime_root = Path(config.trae_runtime_root)  # type: ignore[arg-type]
            dotenv_metadata = Path(
                config.python_dotenv_metadata_file  # type: ignore[arg-type]
            )
            if (
                not install_root.is_absolute()
                or install_root.is_symlink()
                or not install_root.is_dir()
                or install_root.resolve(strict=True) != install_root
                or not identity_path.is_absolute()
                or identity_path.is_symlink()
                or not identity_path.is_file()
                or identity_path.resolve(strict=True) != identity_path
                or identity_path.name != "direct_url.json"
                or not identity_path.parent.name.endswith(".dist-info")
                or not runtime_root.is_absolute()
                or runtime_root.is_symlink()
                or not runtime_root.is_dir()
                or runtime_root.resolve(strict=True) != runtime_root
                or not dotenv_metadata.is_absolute()
                or dotenv_metadata.is_symlink()
                or not dotenv_metadata.is_file()
                or dotenv_metadata.resolve(strict=True) != dotenv_metadata
                or dotenv_metadata.name != "METADATA"
                or not dotenv_metadata.parent.name.startswith("python_dotenv-")
                or not dotenv_metadata.parent.name.endswith(".dist-info")
            ):
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID", "invalid Trae install identity path"
                )
            try:
                executable.relative_to(install_root)
                identity_path.relative_to(install_root)
                runtime_root.relative_to(install_root)
                dotenv_metadata.relative_to(install_root)
            except ValueError as exc:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID",
                    "Trae executable and identity must share the canonical install root",
                ) from exc
            for field in (
                "trae_install_identity_sha256",
                "trae_executable_sha256",
                "trae_runtime_manifest_sha256",
                "python_dotenv_metadata_sha256",
            ):
                value = getattr(config, field)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise CodingWorkerError("TRAE_CONFIG_INVALID", f"invalid {field}")
            if config.docker_image is None or _PINNED_IMAGE_RE.fullmatch(
                config.docker_image
            ) is None:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID", "Docker image must be pinned by sha256 digest"
                )
            if config.docker_executable is None:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID", "an explicit Docker executable is required"
                )
            docker_path = Path(config.docker_executable)
            if (
                not docker_path.is_absolute()
                or not docker_path.is_file()
                or docker_path.is_symlink()
                or docker_path.resolve(strict=True) != docker_path
                or docker_path.name
                not in (("docker", "docker.exe") if os.name == "nt" else ("docker",))
                or not os.access(docker_path, os.X_OK)
            ):
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID",
                    "docker_executable must be an absolute executable regular file named docker",
                )
            if config.docker_host is not None:
                try:
                    normalize_local_docker_host(config.docker_host)
                except ValueError as exc:
                    raise CodingWorkerError(
                        "TRAE_CONFIG_INVALID",
                        str(exc),
                    ) from exc
        if config.docker_image is not None and _PINNED_IMAGE_RE.fullmatch(
            config.docker_image
        ) is None:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Docker image must be pinned by sha256 digest"
            )
        if config.trae_source_revision is not None and (
            len(config.trae_source_revision) not in (40, 64)
            or any(
                character not in "0123456789abcdef"
                for character in config.trae_source_revision
            )
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID",
                "trae_source_revision must be a full lowercase Git object ID",
            )
        docker_integer_fields = (
            "docker_memory_limit_mb",
            "docker_pids_limit",
            "docker_tmpfs_limit_mb",
            "docker_agent_tools_size_limit_mb",
            "docker_cli_timeout_seconds",
        )
        for field in docker_integer_fields:
            value = getattr(config, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CodingWorkerError("TRAE_CONFIG_INVALID", f"invalid {field}")
        if (
            isinstance(config.docker_cpu_limit, bool)
            or not isinstance(config.docker_cpu_limit, (int, float))
            or not 0 < float(config.docker_cpu_limit) <= 64
        ):
            raise CodingWorkerError("TRAE_CONFIG_INVALID", "invalid docker_cpu_limit")
        if (
            isinstance(config.worktree_lease_timeout_seconds, bool)
            or not isinstance(config.worktree_lease_timeout_seconds, (int, float))
            or not 0 < float(config.worktree_lease_timeout_seconds) <= 300
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "invalid worktree_lease_timeout_seconds"
            )

    def _verify_install_identity(self) -> Mapping[str, Any]:
        """Bind the runtime executable to the reviewed VCS installation evidence."""

        if self.config.trusted_test_mode:
            return {"verification_mode": "trusted_test_bypass"}
        executable = Path(self.config.command_prefix[0])
        identity_path = self.config.trae_install_identity_file
        expected_identity_hash = self.config.trae_install_identity_sha256
        expected_executable_hash = self.config.trae_executable_sha256
        dotenv_metadata_path = self.config.python_dotenv_metadata_file
        expected_dotenv_hash = self.config.python_dotenv_metadata_sha256
        if (
            identity_path is None
            or expected_identity_hash is None
            or expected_executable_hash is None
            or dotenv_metadata_path is None
            or expected_dotenv_hash is None
        ):
            raise CodingWorkerError(
                "TRAE_INSTALL_IDENTITY_MISMATCH",
                "production Trae install identity is incomplete",
            )
        try:
            if executable.stat().st_size > 1024 * 1024:
                raise CodingWorkerError(
                    "TRAE_INSTALL_IDENTITY_MISMATCH",
                    "Trae console executable exceeds the reviewed size bound",
                )
            executable_bytes = executable.read_bytes()
            if identity_path.stat().st_size > 64 * 1024:
                raise CodingWorkerError(
                    "TRAE_INSTALL_IDENTITY_MISMATCH",
                    "Trae direct-url evidence exceeds the reviewed size bound",
                )
            identity_bytes = identity_path.read_bytes()
            if dotenv_metadata_path.stat().st_size > 256 * 1024:
                raise CodingWorkerError(
                    "TRAE_INSTALL_IDENTITY_MISMATCH",
                    "python-dotenv metadata exceeds the reviewed size bound",
                )
            dotenv_metadata_bytes = dotenv_metadata_path.read_bytes()
        except OSError as exc:
            raise CodingWorkerError(
                "TRAE_INSTALL_IDENTITY_MISMATCH",
                "Trae install identity could not be read",
            ) from exc
        executable_hash = hashlib.sha256(executable_bytes).hexdigest()
        identity_hash = hashlib.sha256(identity_bytes).hexdigest()
        dotenv_metadata_hash = hashlib.sha256(dotenv_metadata_bytes).hexdigest()
        if (
            executable_hash != expected_executable_hash
            or identity_hash != expected_identity_hash
            or dotenv_metadata_hash != expected_dotenv_hash
        ):
            raise CodingWorkerError(
                "TRAE_INSTALL_IDENTITY_MISMATCH",
                "Trae installed bytes differ from the reviewed hashes",
            )
        try:
            dotenv_metadata = email.parser.Parser().parsestr(
                dotenv_metadata_bytes.decode("utf-8")
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise CodingWorkerError(
                "TRAE_INSTALL_IDENTITY_MISMATCH",
                "python-dotenv install metadata is malformed",
            ) from exc
        if (
            dotenv_metadata.get("Name") != "python-dotenv"
            or dotenv_metadata.get("Version") != _REVIEWED_DOTENV_VERSION
        ):
            raise CodingWorkerError(
                "TRAE_INSTALL_IDENTITY_MISMATCH",
                "python-dotenv is not the reviewed disable-guard version",
            )
        try:
            direct_url = json.loads(identity_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CodingWorkerError(
                "TRAE_INSTALL_IDENTITY_MISMATCH", "Trae direct-url evidence is malformed"
            ) from exc
        if not isinstance(direct_url, Mapping) or set(direct_url) != {"url", "vcs_info"}:
            raise CodingWorkerError(
                "TRAE_INSTALL_IDENTITY_MISMATCH",
                "Trae direct-url evidence has an unreviewed shape",
            )
        vcs_info = direct_url.get("vcs_info")
        if (
            direct_url.get("url") != _REVIEWED_TRAE_SOURCE_URL
            or not isinstance(vcs_info, Mapping)
            or set(vcs_info) != {"vcs", "commit_id", "requested_revision"}
            or vcs_info.get("vcs") != "git"
            or vcs_info.get("commit_id") != self.config.trae_source_revision
            or vcs_info.get("requested_revision") != self.config.trae_source_revision
        ):
            raise CodingWorkerError(
                "TRAE_INSTALL_IDENTITY_MISMATCH",
                "Trae installation is not the reviewed VCS revision",
            )
        return {
            "verification_mode": "hashed_direct_url_and_executable",
            "source_url": _REVIEWED_TRAE_SOURCE_URL,
            "source_revision": self.config.trae_source_revision,
            "direct_url_sha256": identity_hash,
            "executable_sha256": executable_hash,
            "python_dotenv_version": _REVIEWED_DOTENV_VERSION,
            "python_dotenv_metadata_sha256": dotenv_metadata_hash,
        }

    def _verify_runtime_root(self) -> Tuple[Path, Mapping[str, Any]]:
        """Verify the trusted CLI cwd, Docker tool assets, and dotenv boundary."""

        if self.config.trusted_test_mode:
            return self.worktrees.repository, {
                "verification_mode": "trusted_test_bypass",
                "cwd_mode": "controller_repository",
            }
        configured_root = self.config.trae_runtime_root
        expected_manifest = self.config.trae_runtime_manifest_sha256
        if configured_root is None or expected_manifest is None:
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "trusted Trae runtime root is not configured",
            )
        try:
            root = Path(configured_root).resolve(strict=True)
        except OSError as exc:
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "trusted Trae runtime root is unavailable",
            ) from exc
        try:
            root.relative_to(self.worktrees.repository)
        except ValueError:
            pass
        else:
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "trusted Trae runtime root cannot be candidate-editable",
            )
        manifest = hash_trae_runtime_package(root)
        if manifest != expected_manifest:
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "Trae Docker tool assets differ from the reviewed manifest",
            )
        if self.config.provider_base_url == "https://api.deepseek.com":
            reasoning_client = (
                root / "trae_agent" / "utils" / "llm_clients" / "openai_client.py"
            )
            try:
                reasoning_source = reasoning_client.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise CodingWorkerError(
                    "TRAE_RUNTIME_IDENTITY_MISMATCH",
                    "reviewed DeepSeek Trae client patch is unavailable",
                ) from exc
            if (
                TRAE_DEEPSEEK_REASONING_MARKER not in reasoning_source
                or TRAE_DEEPSEEK_TOOL_JSON_MARKER not in reasoning_source
                or 'reasoning={"effort": "high"}' not in reasoning_source
                or 'output_block.type == "reasoning"' not in reasoning_source
                or "content += message_content" not in reasoning_source
                or "except (json.JSONDecodeError, TypeError, RecursionError)"
                not in reasoning_source
                or "if not isinstance(tool_arguments, dict)" not in reasoning_source
            ):
                raise CodingWorkerError(
                    "TRAE_RUNTIME_IDENTITY_MISMATCH",
                    "reviewed DeepSeek reasoning continuity patch is missing",
                )
            edit_executor = root / "trae_agent" / "tools" / "docker_tool_executor.py"
            try:
                edit_source = edit_executor.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise CodingWorkerError(
                    "TRAE_RUNTIME_IDENTITY_MISMATCH",
                    "reviewed DeepSeek edit-tool compatibility patch is unavailable",
                ) from exc
            if (
                TRAE_DOCKER_EDIT_TOOL_MARKER not in edit_source
                or "command_arguments =" not in edit_source
                or "shlex.join(cmd_parts)" not in edit_source
            ):
                raise CodingWorkerError(
                    "TRAE_RUNTIME_IDENTITY_MISMATCH",
                    "reviewed DeepSeek edit-tool compatibility patch is missing",
                )
        asset_bytes = _trae_runtime_asset_bytes(root)
        if asset_bytes > self.config.docker_agent_tools_size_limit_mb * 1024 * 1024:
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "reviewed agent-tools bundle exceeds the Docker mount size limit",
            )
        _reject_dotenv_candidates(root)
        return root, {
            "verification_mode": "hashed_runtime_assets_and_dotenv_preflight",
            "runtime_package_manifest_sha256": manifest,
            "runtime_asset_bytes": asset_bytes,
            "dotenv_candidates_present": False,
            "cwd_mode": "trusted_runtime_root",
        }

    def _verify_config_file(self) -> None:
        path = Path(self.config.config_file)
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise CodingWorkerError("TRAE_CONFIG_MISSING", "Trae config file is missing") from exc
        if path.is_symlink() or not resolved.is_file():
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae config must be a regular non-symlink file"
            )
        if not self.config.trusted_test_mode:
            for editable_root in (
                self.worktrees.repository,
                self.worktrees.worktree_root,
            ):
                try:
                    resolved.relative_to(editable_root)
                except ValueError:
                    continue
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID",
                    "production Trae config must be outside candidate-editable roots",
                )
        raw = resolved.read_bytes()
        if len(raw) > 1024 * 1024:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae config exceeds the reviewed size limit"
            )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.config.config_sha256:
            raise CodingWorkerError(
                "TRAE_CONFIG_HASH_MISMATCH", "Trae config bytes changed after approval"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae config must be UTF-8"
            ) from exc
        if self.redactor.contains_credential_material(text):
            raise CodingWorkerError(
                "CREDENTIAL_IN_CONFIG",
                "Trae config contains credential material; use approved environment variables",
            )
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae config is not valid safe YAML"
            ) from exc
        self._validate_config_document(document)

    def _validate_config_document(self, document: Any) -> None:
        """Fail closed if the hashed YAML enables an unreviewed execution path."""

        if not isinstance(document, Mapping):
            raise CodingWorkerError("TRAE_CONFIG_INVALID", "Trae config must be a mapping")
        expected_root_keys = {
            "agents",
            "allow_mcp_servers",
            "mcp_servers",
            "model_providers",
            "models",
        }
        if set(document) != expected_root_keys:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae config contains an unreviewed root setting"
            )
        if document.get("allow_mcp_servers") != [] or document.get("mcp_servers") != {}:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae MCP servers must be completely disabled"
            )

        agents = document.get("agents")
        if not isinstance(agents, Mapping) or set(agents) != {"trae_agent"}:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae config must define only trae_agent"
            )
        agent = agents.get("trae_agent")
        if not isinstance(agent, Mapping) or set(agent) != {
            "enable_lakeview",
            "model",
            "max_steps",
            "tools",
        }:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae agent settings differ from the reviewed surface"
            )
        if agent.get("enable_lakeview") is not False:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae lakeview must remain disabled"
            )
        if agent.get("max_steps") != self.config.max_steps_cap:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae YAML max_steps must equal max_steps_cap"
            )
        tools = agent.get("tools")
        if not isinstance(tools, list) or tuple(tools) != self.config.reviewed_tool_names:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae tools differ from the reviewed Docker-routed set"
            )

        model_alias = agent.get("model")
        models = document.get("models")
        if (
            not isinstance(model_alias, str)
            or not isinstance(models, Mapping)
            or set(models) != {model_alias}
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae model alias is not uniquely defined"
            )
        model = models.get(model_alias)
        if not isinstance(model, Mapping):
            raise CodingWorkerError("TRAE_CONFIG_INVALID", "Trae model must be a mapping")
        allowed_model_keys = {
            "model_provider",
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "max_retries",
            "parallel_tool_calls",
        }
        if set(model) != allowed_model_keys:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae model settings differ from the reviewed surface"
            )
        if model.get("model") != self.config.model_id:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae YAML model differs from TraeConfig"
            )
        max_tokens = model.get("max_tokens")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
            or (
                self.config.max_token_cap is not None
                and max_tokens > self.config.max_token_cap
            )
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "model max_tokens exceeds the reviewed token cap"
            )
        if model.get("parallel_tool_calls") is not False:
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "parallel tool calls must remain disabled"
            )

        provider_alias = model.get("model_provider")
        providers = document.get("model_providers")
        if (
            not isinstance(provider_alias, str)
            or not isinstance(providers, Mapping)
            or set(providers) != {provider_alias}
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae provider alias is not uniquely defined"
            )
        provider = providers.get(provider_alias)
        expected_provider_keys = {"provider", "api_key"}
        if self.config.provider_base_url is not None:
            expected_provider_keys.add("base_url")
        if (
            not isinstance(provider, Mapping)
            or set(provider) != expected_provider_keys
            or provider.get("provider") != self.config.provider
            or provider.get("base_url") != self.config.provider_base_url
        ):
            raise CodingWorkerError(
                "TRAE_CONFIG_INVALID", "Trae YAML provider differs from TraeConfig"
            )
        if provider.get("api_key") not in (None, ""):
            raise CodingWorkerError(
                "CREDENTIAL_IN_CONFIG", "Trae provider key must come only from the environment"
            )

    def _sanitized_environment(self) -> Mapping[str, str]:
        environment = {}
        for name in self.config.approved_environment_names:
            if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID", "invalid approved environment name"
                )
            value = self._source_environment.get(name)
            if value is not None:
                if not isinstance(value, str) or "\x00" in value:
                    raise CodingWorkerError(
                        "TRAE_CONFIG_INVALID", "approved environment value is invalid"
                    )
                environment[name] = value
        for source, target in self.config.credential_environment_aliases:
            value = environment.get(source)
            if value is not None:
                environment[target] = value
        environment["LANG"] = "C.UTF-8"
        environment["LC_ALL"] = "C.UTF-8"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHON_DOTENV_DISABLED"] = "1"
        environment["GIT_TERMINAL_PROMPT"] = "0"
        if os.name == "nt":
            # Python's Windows runtime needs these system values to load DLLs
            # and resolve subprocesses after the rest of the environment is
            # intentionally reduced to approved names.
            for name in (
                "SystemRoot",
                "WINDIR",
                "ComSpec",
                "PATHEXT",
                "SystemDrive",
                "ProgramData",
            ):
                value = os.environ.get(name)
                if value:
                    environment[name] = value
        if not self.config.trusted_test_mode:
            docker_executable = self.config.docker_executable
            if docker_executable is None:
                raise CodingWorkerError(
                    "TRAE_CONFIG_INVALID", "Docker executable is required in production"
                )
            if os.name == "nt":
                environment["PATH"] = os.pathsep.join(
                    (str(docker_executable.parent), os.environ.get("PATH", ""))
                )
            else:
                environment["PATH"] = f"{docker_executable.parent}:/usr/bin:/bin"
            if self.config.docker_host is not None:
                environment["DOCKER_HOST"] = self.config.docker_host
        return environment

    def _action_environment(
        self,
        temporary_root: Path,
        *,
        docker_config_root: Optional[Path] = None,
    ) -> Mapping[str, str]:
        """Create an isolated home/cache/temp environment for one host Trae action."""

        roots = {
            "HOME": temporary_root / "home",
            "TMPDIR": temporary_root / "tmp",
            "TMP": temporary_root / "tmp",
            "TEMP": temporary_root / "tmp",
            "XDG_CONFIG_HOME": temporary_root / "xdg-config",
            "XDG_CACHE_HOME": temporary_root / "xdg-cache",
        }
        if os.name == "nt":
            # pathlib.Path.home() uses USERPROFILE on Windows and ignores
            # HOME. Keep Trae's home isolated from the operator's profile.
            roots["USERPROFILE"] = temporary_root / "home"
        for path in set(roots.values()):
            path.mkdir(mode=0o700)
        selected_docker_config = docker_config_root or temporary_root / "docker-config"
        selected_docker_config.mkdir(mode=0o700, exist_ok=True)
        environment = dict(self._environment)
        environment.update({name: str(path) for name, path in roots.items()})
        environment["DOCKER_CONFIG"] = str(selected_docker_config)
        return environment

    @staticmethod
    def _resolve_identity(value: Any) -> CandidateIdentity:
        if not isinstance(value, CandidateIdentity):
            raise CodingWorkerError(
                "CANDIDATE_IDENTITY_MISSING",
                "identity resolver must return CandidateIdentity before Trae invocation",
            )
        return value


    def _schema_factories(self) -> SchemaFactories:
        """Resolve shared models lazily, always before an external coding action."""

        if self._factories is None:
            self._factories = SchemaFactories.from_shared_schemas()
        return self._factories


def hash_trae_runtime_package(runtime_root: Path) -> str:
    """Hash the complete installed Trae package, excluding bytecode caches."""

    try:
        root = Path(runtime_root).resolve(strict=True)
        package_root = root / "trae_agent"
        assets = package_root / "dist"
        internal = assets / "_internal"
        if (
            package_root.is_symlink()
            or assets.is_symlink()
            or internal.is_symlink()
            or not package_root.is_dir()
            or not assets.is_dir()
            or not internal.is_dir()
            or package_root.resolve(strict=True) != package_root
            or assets.resolve(strict=True) != assets
            or internal.resolve(strict=True) != internal
        ):
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "Trae runtime does not contain canonical packaged Docker assets",
            )
        paths = sorted(package_root.rglob("*"), key=lambda path: path.as_posix())
    except OSError as exc:
        raise CodingWorkerError(
            "TRAE_RUNTIME_IDENTITY_MISMATCH",
            "Trae runtime assets could not be enumerated",
        ) from exc
    if len(paths) > 50_000:
        raise CodingWorkerError(
            "TRAE_RUNTIME_IDENTITY_MISMATCH", "Trae runtime asset count is unbounded"
        )
    records = []
    total_bytes = 0
    for path in paths:
        relative_path = path.relative_to(package_root)
        relative = relative_path.as_posix()
        if path.is_symlink():
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "Trae runtime assets cannot contain symlinks",
            )
        if "__pycache__" in relative_path.parts or path.suffix in {".pyc", ".pyo"}:
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "Trae runtime cannot contain executable bytecode caches",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "Trae runtime contains a non-regular asset",
            )
        metadata = path.stat()
        total_bytes += metadata.st_size
        if total_bytes > 2 * 1024 * 1024 * 1024:
            raise CodingWorkerError(
                "TRAE_RUNTIME_IDENTITY_MISMATCH",
                "Trae runtime assets exceed the reviewed byte bound",
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        records.append(
            {
                "path": relative,
                "sha256": digest.hexdigest(),
                "size_bytes": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        )
    required_paths = {
        "cli.py",
        "agent/base_agent.py",
        "agent/docker_manager.py",
        "agent/trae_agent.py",
        "tools/docker_tool_executor.py",
        "dist/edit_tool",
        "dist/json_edit_tool",
    }
    recorded_paths = {record["path"] for record in records}
    if not records or not required_paths.issubset(recorded_paths):
        raise CodingWorkerError(
            "TRAE_RUNTIME_IDENTITY_MISMATCH",
            "Trae runtime package is missing a reviewed attach-contract component",
        )
    manifest = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(manifest).hexdigest()


def _reject_dotenv_candidates(runtime_root: Path) -> None:
    """Reject every `.env` the pinned import or trusted cwd could discover."""

    try:
        for candidate in runtime_root.rglob(".env"):
            if candidate.exists() or candidate.is_symlink():
                raise CodingWorkerError(
                    "TRAE_DOTENV_FORBIDDEN",
                    "an unreviewed .env exists in the trusted Trae runtime tree",
                )
        current = runtime_root
        while True:
            candidate = current / ".env"
            if candidate.exists() or candidate.is_symlink():
                raise CodingWorkerError(
                    "TRAE_DOTENV_FORBIDDEN",
                    "an unreviewed .env exists on the Trae runtime search path",
                )
            if current.parent == current:
                break
            current = current.parent
    except OSError as exc:
        raise CodingWorkerError(
            "TRAE_DOTENV_FORBIDDEN", "Trae dotenv preflight could not be completed"
        ) from exc


def _trae_runtime_asset_bytes(runtime_root: Path) -> int:
    assets = runtime_root / "trae_agent" / "dist"
    try:
        return sum(path.stat().st_size for path in assets.rglob("*") if path.is_file())
    except OSError as exc:
        raise CodingWorkerError(
            "TRAE_RUNTIME_IDENTITY_MISMATCH",
            "Trae runtime asset sizes could not be verified",
        ) from exc


class FakeCodingWorker:
    """Deterministic shared-model test double for Person 2 integration tests."""

    def __init__(self, *, create_result: Any, repair_result: Any) -> None:
        self.create_result = create_result
        self.repair_result = repair_result
        self.calls: list[Tuple[str, Any, Any]] = []

    async def create_patch(self, context: Any, spec: Any) -> Any:
        self.calls.append(("create_patch", context, spec))
        return _fake_result(self.create_result)

    async def repair_patch(self, context: Any, decision: Any) -> Any:
        self.calls.append(("repair_patch", context, decision))
        return _fake_result(self.repair_result)


def _run_bounded_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    termination_grace_seconds: int,
    output_path: Path,
    max_output_bytes: int,
    redactor: SecretRedactor,
) -> _ProcessResult:
    """Run one argv-only child in its own process group with hard bounds."""

    started = time.monotonic()
    timed_out = False
    output_limited = False
    with Path(output_path).open("wb") as output:
        try:
            process_options = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt"
                else {"start_new_session": True}
            )
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                shell=False,
                close_fds=True,
                **process_options,
            )
        except OSError as exc:
            raise CodingWorkerError(
                "TRAE_LAUNCH_FAILED", "failed to launch the pinned Trae executable"
            ) from exc
        while process.poll() is None:
            elapsed = time.monotonic() - started
            output.flush()
            try:
                output_size = Path(output_path).stat().st_size
            except FileNotFoundError:
                output_size = 0
            if output_size > max_output_bytes:
                output_limited = True
                _terminate_process_group(process, termination_grace_seconds)
                break
            if elapsed >= timeout_seconds:
                timed_out = True
                _terminate_process_group(process, termination_grace_seconds)
                break
            time.sleep(0.02)
        exit_code = process.wait()
        if not timed_out and not output_limited:
            _cleanup_process_group(process.pid, termination_grace_seconds)
        output.flush()
        if Path(output_path).stat().st_size > max_output_bytes:
            output_limited = True
    tail = _read_tail(Path(output_path), min(max_output_bytes, 64 * 1024))
    return _ProcessResult(
        exit_code=exit_code,
        timed_out=timed_out,
        output_limited=output_limited,
        output_tail=redactor.redact(tail),
    )


def _fake_result(value: Any) -> Any:
    if isinstance(value, BaseException):
        raise value
    return value


def _container_user() -> str:
    """Return a portable numeric uid:gid for the Linux Trae container."""

    get_uid = getattr(os, "getuid", lambda: 65534)
    get_gid = getattr(os, "getgid", lambda: 65534)
    return f"{get_uid()}:{get_gid()}"


def _terminate_process_group(process: subprocess.Popen[Any], grace_seconds: int) -> None:
    if process.poll() is None:
        if os.name == "nt":
            _windows_terminate_tree(process.pid)
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    _cleanup_process_group(process.pid, grace_seconds)


def _cleanup_process_group(process_group_id: int, grace_seconds: int) -> None:
    """Terminate descendants left in the dedicated coding process group."""

    if os.name == "nt":
        _windows_terminate_tree(process_group_id)
        return

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _windows_terminate_tree(process_id: int) -> None:
    """Terminate a Windows child tree (the native equivalent of ``killpg``)."""

    try:
        subprocess.run(
            ("taskkill", "/PID", str(process_id), "/T", "/F"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
    except OSError:
        # The child may already have exited, or taskkill may be unavailable in
        # a constrained test environment. The parent wait/kill path remains the
        # final bound.
        pass


def _read_tail(path: Path, maximum_bytes: int) -> str:
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - maximum_bytes))
        value = handle.read(maximum_bytes)
    return value.decode("utf-8", errors="replace")


def _context_text(context: Any, field: str) -> str:
    value = getattr(context, field, None)
    if not isinstance(value, str) or not value.strip():
        raise CodingWorkerError("CONTEXT_INVALID", f"missing or invalid {field}")
    return value


def _context_int(context: Any, field: str) -> int:
    value = getattr(context, field, None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodingWorkerError("CONTEXT_INVALID", f"missing or invalid {field}")
    return value


def _context_optional_positive_int(context: Any, field: str) -> Optional[int]:
    try:
        value = getattr(context, field)
    except AttributeError as exc:
        raise CodingWorkerError(
            "CONTEXT_INVALID", f"missing or invalid {field}"
        ) from exc
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CodingWorkerError(
            "CONTEXT_INVALID", f"{field} must be null or a positive integer"
        )
    return value


def _model_field(model: Any, field: str) -> str:
    if isinstance(model, Mapping):
        value = model.get(field)
    else:
        value = getattr(model, field, None)
    if not isinstance(value, str) or not value:
        raise CodingWorkerError("CONTEXT_INVALID", f"spec is missing {field}")
    return value
