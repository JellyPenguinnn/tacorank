"""Reviewed symbolic command registry.

Run requests carry a ``command_id`` only.  They never carry executable text,
arguments, working directories, or environment variables supplied by an LLM.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple


REQUIRED_COMMAND_IDS = frozenset(
    {
        "baseline_full",
        "candidate_smoke",
        "candidate_proxy",
        "candidate_full",
        "candidate_final_infer",
        "submission_check",
        "clean_reproduce",
    }
)

_PIPELINE_COMMAND_IDS = REQUIRED_COMMAND_IDS.difference({"submission_check"})

_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_SAFE_IDENT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_PIPELINE_ENTRYPOINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)
_ALLOWED_PLACEHOLDERS = frozenset(
    {
        "repository_root",
        "worktree",
        "artifact_dir",
        "prediction_path",
        "checkpoint_path",
        "submission_prediction_path",
        "run_id",
        "experiment_id",
        "attempt",
        "fidelity",
        "seed",
    }
)


class CommandPolicyError(ValueError):
    """Raised when a symbolic command or its context violates policy."""


@dataclass(frozen=True)
class PipelineCommandInputs:
    """Controller-reviewed integration values for the generic solution CLIs.

    Input roots are keyed by symbolic command so the controller can expose a
    different legal data view at each fidelity.  Docker translates these host
    paths to its exact read-only mount targets; the trusted local test backend
    uses the canonical host paths directly.
    """

    contract_root: Path
    input_roots: Mapping[str, Path]
    baseline_entrypoint: str
    candidate_entrypoint: str
    submission_check_entrypoint: str

    def __post_init__(self) -> None:
        contract_root = _canonical_pipeline_root(
            self.contract_root, "pipeline contract root"
        )
        supplied_ids = set(self.input_roots)
        if supplied_ids != _PIPELINE_COMMAND_IDS:
            missing = sorted(_PIPELINE_COMMAND_IDS.difference(supplied_ids))
            extra = sorted(supplied_ids.difference(_PIPELINE_COMMAND_IDS))
            details = []
            if missing:
                details.append("missing: {0}".format(", ".join(missing)))
            if extra:
                details.append("unexpected: {0}".format(", ".join(extra)))
            raise CommandPolicyError(
                "pipeline input roots must match pipeline command IDs ({0})".format(
                    "; ".join(details)
                )
            )
        input_roots = {
            command_id: _canonical_pipeline_root(
                self.input_roots[command_id],
                "pipeline input root for {0}".format(command_id),
            )
            for command_id in sorted(_PIPELINE_COMMAND_IDS)
        }
        for label, value in (
            ("baseline entrypoint", self.baseline_entrypoint),
            ("candidate entrypoint", self.candidate_entrypoint),
            ("submission-check entrypoint", self.submission_check_entrypoint),
        ):
            if _PIPELINE_ENTRYPOINT.fullmatch(value) is None:
                raise CommandPolicyError("invalid {0}".format(label))
        object.__setattr__(self, "contract_root", contract_root)
        object.__setattr__(self, "input_roots", MappingProxyType(input_roots))


@dataclass(frozen=True)
class ExpectedArtifact:
    role: str
    relative_path: str
    kind: str
    content_type: Optional[str] = None
    required: bool = True

    def __post_init__(self) -> None:
        if self.role not in {"prediction", "checkpoint"}:
            raise CommandPolicyError("unsupported artifact role")
        _validate_relative_path(self.relative_path, "expected artifact")


@dataclass(frozen=True)
class CommandProfile:
    """A code-reviewed executable template registered by the controller."""

    command_id: str
    executable: str
    arguments: Tuple[str, ...] = ()
    working_directory: str = "worktree"
    environment: Mapping[str, str] = field(default_factory=dict)
    allowed_fidelities: Tuple[str, ...] = ()
    allow_network: bool = False
    expected_artifacts: Tuple[ExpectedArtifact, ...] = ()
    heartbeat_interval_seconds: Optional[float] = None
    gpu_count: int = 0
    container_executable: Optional[str] = None

    def __post_init__(self) -> None:
        arguments = tuple(self.arguments)
        allowed_fidelities = tuple(self.allowed_fidelities)
        expected_artifacts = tuple(self.expected_artifacts)
        if not _SAFE_IDENT.fullmatch(self.command_id):
            raise CommandPolicyError("invalid command_id")
        executable = Path(self.executable)
        if not executable.is_absolute():
            raise CommandPolicyError("command executable must be an absolute path")
        if "\x00" in self.executable:
            raise CommandPolicyError("command executable contains NUL")
        if self.working_directory not in {"worktree", "artifact_dir"}:
            raise CommandPolicyError("unsupported working directory selector")
        if self.gpu_count < 0:
            raise CommandPolicyError("gpu_count must be non-negative")
        if self.container_executable is not None:
            _validate_container_executable(self.container_executable)
        if (
            self.heartbeat_interval_seconds is not None
            and self.heartbeat_interval_seconds <= 0
        ):
            raise CommandPolicyError("heartbeat interval must be positive")
        for argument in arguments:
            _validate_template(argument)
        for key, value in self.environment.items():
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                raise CommandPolicyError("invalid environment key")
            if _looks_sensitive(key):
                raise CommandPolicyError(
                    "credential-shaped environment keys are not registry values"
                )
            _validate_template(value)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "allowed_fidelities", allowed_fidelities)
        object.__setattr__(self, "expected_artifacts", expected_artifacts)
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


@dataclass(frozen=True)
class CommandContext:
    repository_root: Path
    worktree: Path
    artifact_dir: Path
    run_id: str
    experiment_id: str
    attempt: int
    fidelity: str
    seed: int
    submission_prediction_path: Optional[Path] = None


@dataclass(frozen=True)
class ResolvedCommand:
    command_id: str
    argv: Tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    network_enabled: bool
    expected_artifacts: Tuple[ExpectedArtifact, ...]
    heartbeat_interval_seconds: Optional[float]
    gpu_count: int
    container_executable: Optional[str]

    def public_configuration(self) -> Mapping[str, object]:
        """Return the secret-free configuration written to the artifact log."""

        return {
            "command_id": self.command_id,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "environment_keys": sorted(self.environment),
            "network_enabled": self.network_enabled,
            "expected_artifacts": [
                {
                    "role": item.role,
                    "relative_path": item.relative_path,
                    "kind": item.kind,
                    "content_type": item.content_type,
                    "required": item.required,
                }
                for item in self.expected_artifacts
            ],
            "gpu_count": self.gpu_count,
            "container_executable": self.container_executable,
        }


class CommandRegistry:
    """Resolve allowlisted command IDs to exact argv arrays without a shell."""

    def __init__(self, profiles: Iterable[CommandProfile]) -> None:
        self._profiles: Dict[str, CommandProfile] = {}
        for profile in profiles:
            if profile.command_id in self._profiles:
                raise CommandPolicyError(
                    "duplicate command_id: {0}".format(profile.command_id)
                )
            self._profiles[profile.command_id] = profile

    @property
    def command_ids(self) -> frozenset[str]:
        return frozenset(self._profiles)

    def require_standard_commands(self) -> None:
        missing = sorted(REQUIRED_COMMAND_IDS.difference(self._profiles))
        if missing:
            raise CommandPolicyError(
                "registry is missing required command IDs: {0}".format(
                    ", ".join(missing)
                )
            )

    def resolve(
        self,
        command_id: str,
        context: CommandContext,
        *,
        network_enabled: bool,
    ) -> ResolvedCommand:
        profile = self._profiles.get(str(command_id))
        if profile is None:
            raise CommandPolicyError("UNAPPROVED_COMMAND: unknown command_id")
        self._validate_context(context)
        if profile.allowed_fidelities and context.fidelity not in set(
            profile.allowed_fidelities
        ):
            raise CommandPolicyError(
                "UNAPPROVED_COMMAND: fidelity is not allowed for command"
            )
        if network_enabled and not profile.allow_network:
            raise CommandPolicyError("UNAPPROVED_NETWORK")
        if profile.command_id == "submission_check":
            prediction = context.submission_prediction_path
            if (
                prediction is None
                or not Path(prediction).is_absolute()
                or Path(prediction).is_symlink()
                or not Path(prediction).is_file()
                or Path(prediction).resolve(strict=True) != Path(prediction)
            ):
                raise CommandPolicyError(
                    "submission_check requires a verified prior prediction"
                )
        elif context.submission_prediction_path is not None:
            raise CommandPolicyError(
                "prior prediction input is restricted to submission_check"
            )

        values = self._template_values(context)
        argv = (profile.executable,) + tuple(
            _render(argument, values) for argument in profile.arguments
        )
        environment = MappingProxyType({
            key: _render(value, values)
            for key, value in profile.environment.items()
        })
        cwd = (
            context.worktree
            if profile.working_directory == "worktree"
            else context.artifact_dir
        )
        if not cwd.is_dir():
            raise CommandPolicyError("resolved working directory does not exist")
        return ResolvedCommand(
            command_id=profile.command_id,
            argv=argv,
            cwd=cwd,
            environment=environment,
            network_enabled=bool(network_enabled),
            expected_artifacts=profile.expected_artifacts,
            heartbeat_interval_seconds=profile.heartbeat_interval_seconds,
            gpu_count=profile.gpu_count,
            container_executable=profile.container_executable,
        )

    @staticmethod
    def _validate_context(context: CommandContext) -> None:
        for field_name in ("run_id", "experiment_id", "fidelity"):
            value = str(getattr(context, field_name))
            if not _SAFE_IDENT.fullmatch(value):
                raise CommandPolicyError("invalid {0}".format(field_name))
        if context.attempt < 1:
            raise CommandPolicyError("attempt must be at least one")
        if not isinstance(context.seed, int):
            raise CommandPolicyError("seed must be an integer")
        for path in (
            context.repository_root,
            context.worktree,
            context.artifact_dir,
        ):
            if not Path(path).is_absolute():
                raise CommandPolicyError("command paths must be absolute")

    @staticmethod
    def _template_values(context: CommandContext) -> Mapping[str, str]:
        return {
            "repository_root": str(context.repository_root),
            "worktree": str(context.worktree),
            "artifact_dir": str(context.artifact_dir),
            "prediction_path": str(context.artifact_dir / "predictions.csv"),
            "checkpoint_path": str(context.artifact_dir / "checkpoint.bin"),
            "submission_prediction_path": (
                ""
                if context.submission_prediction_path is None
                else str(context.submission_prediction_path)
            ),
            "run_id": context.run_id,
            "experiment_id": context.experiment_id,
            "attempt": str(context.attempt),
            "fidelity": context.fidelity,
            "seed": str(context.seed),
        }


def default_command_registry(
    pipeline_inputs: PipelineCommandInputs,
    python_executable: Optional[str] = None,
    container_python_executable: Optional[str] = None,
) -> CommandRegistry:
    """Return the complete reviewed command surface for TacoRank candidates.

    Candidate modules are intentionally only named here; implementation belongs
    under ``solution/``.  The caller must supply canonical data views and exact
    entrypoint names rather than relying on unusable or unsafe guessed defaults.
    Integrators may replace profiles with equally reviewed commands while
    preserving the symbolic IDs.
    """

    python = str(Path(python_executable or sys.executable).resolve())
    prediction = ExpectedArtifact(
        role="prediction",
        relative_path="predictions.csv",
        kind="predictions",
        content_type="text/csv",
    )

    def pipeline_environment(
        command_id: str, entrypoint_key: str, entrypoint_value: str
    ) -> Mapping[str, str]:
        return {
            "TACORANK_CONTRACT_ROOT": str(pipeline_inputs.contract_root),
            "TACORANK_INPUT_ROOT": str(pipeline_inputs.input_roots[command_id]),
            "TACORANK_ARTIFACT_ROOT": "{artifact_dir}",
            entrypoint_key: entrypoint_value,
        }

    profiles = [
        CommandProfile(
            "baseline_full",
            python,
            (
                "-m",
                "tacorank.execution.solution_cli",
                "pipeline",
                "--baseline",
                "--output",
                "{prediction_path}",
            ),
            environment=pipeline_environment(
                "baseline_full",
                "TACORANK_BASELINE_ENTRYPOINT",
                pipeline_inputs.baseline_entrypoint,
            ),
            allowed_fidelities=("full",),
            expected_artifacts=(prediction,),
            container_executable=container_python_executable,
        ),
        CommandProfile(
            "candidate_smoke",
            python,
            (
                "-m",
                "tacorank.execution.solution_cli",
                "pipeline",
                "--fidelity",
                "smoke",
                "--seed",
                "{seed}",
                "--output",
                "{prediction_path}",
            ),
            environment=pipeline_environment(
                "candidate_smoke",
                "TACORANK_CANDIDATE_ENTRYPOINT",
                pipeline_inputs.candidate_entrypoint,
            ),
            allowed_fidelities=("smoke",),
            expected_artifacts=(prediction,),
            container_executable=container_python_executable,
        ),
        CommandProfile(
            "candidate_proxy",
            python,
            (
                "-m",
                "tacorank.execution.solution_cli",
                "pipeline",
                "--fidelity",
                "proxy",
                "--seed",
                "{seed}",
                "--output",
                "{prediction_path}",
            ),
            environment=pipeline_environment(
                "candidate_proxy",
                "TACORANK_CANDIDATE_ENTRYPOINT",
                pipeline_inputs.candidate_entrypoint,
            ),
            allowed_fidelities=("proxy",),
            expected_artifacts=(prediction,),
            container_executable=container_python_executable,
        ),
        CommandProfile(
            "candidate_full",
            python,
            (
                "-m",
                "tacorank.execution.solution_cli",
                "pipeline",
                "--fidelity",
                "full",
                "--seed",
                "{seed}",
                "--output",
                "{prediction_path}",
            ),
            environment=pipeline_environment(
                "candidate_full",
                "TACORANK_CANDIDATE_ENTRYPOINT",
                pipeline_inputs.candidate_entrypoint,
            ),
            allowed_fidelities=("full",),
            expected_artifacts=(prediction,),
            container_executable=container_python_executable,
        ),
        CommandProfile(
            "candidate_final_infer",
            python,
            (
                "-m",
                "tacorank.execution.solution_cli",
                "pipeline",
                "--fidelity",
                "final",
                "--seed",
                "{seed}",
                "--output",
                "{prediction_path}",
            ),
            environment=pipeline_environment(
                "candidate_final_infer",
                "TACORANK_CANDIDATE_ENTRYPOINT",
                pipeline_inputs.candidate_entrypoint,
            ),
            allowed_fidelities=("full",),
            expected_artifacts=(prediction,),
            container_executable=container_python_executable,
        ),
        CommandProfile(
            "submission_check",
            python,
            (
                "-m",
                "tacorank.execution.solution_cli",
                "submission-check",
                "{submission_prediction_path}",
            ),
            environment={
                "TACORANK_CONTRACT_ROOT": str(pipeline_inputs.contract_root),
                "TACORANK_ARTIFACT_ROOT": "{artifact_dir}",
                "TACORANK_VERIFIED_PREDICTION_PATH": (
                    "{submission_prediction_path}"
                ),
                "TACORANK_SUBMISSION_CHECK_ENTRYPOINT": (
                    pipeline_inputs.submission_check_entrypoint
                ),
            },
            allowed_fidelities=("full",),
            container_executable=container_python_executable,
        ),
        CommandProfile(
            "clean_reproduce",
            python,
            (
                "-m",
                "tacorank.execution.solution_cli",
                "pipeline",
                "--fidelity",
                "full",
                "--clean-reproduce",
                "--seed",
                "{seed}",
                "--output",
                "{prediction_path}",
            ),
            environment=pipeline_environment(
                "clean_reproduce",
                "TACORANK_CANDIDATE_ENTRYPOINT",
                pipeline_inputs.candidate_entrypoint,
            ),
            allowed_fidelities=("full",),
            expected_artifacts=(prediction,),
            container_executable=container_python_executable,
        ),
    ]
    registry = CommandRegistry(profiles)
    registry.require_standard_commands()
    return registry


def _validate_template(value: str) -> None:
    if "\x00" in value:
        raise CommandPolicyError("command template contains NUL")
    placeholders = set(_PLACEHOLDER.findall(value))
    unknown = placeholders.difference(_ALLOWED_PLACEHOLDERS)
    if unknown:
        raise CommandPolicyError(
            "unknown command placeholder: {0}".format(sorted(unknown)[0])
        )
    remainder = _PLACEHOLDER.sub("", value)
    if "{" in remainder or "}" in remainder:
        raise CommandPolicyError("malformed command placeholder")


def _render(template: str, values: Mapping[str, str]) -> str:
    rendered = _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)
    if "\x00" in rendered:
        raise CommandPolicyError("resolved command value contains NUL")
    return rendered


def _validate_relative_path(value: str, label: str) -> None:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise CommandPolicyError("{0} path is not normalized".format(label))


def _validate_container_executable(value: str) -> None:
    if "\\" in value or "\x00" in value:
        raise CommandPolicyError("container executable contains an invalid character")
    path = Path(value)
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise CommandPolicyError("container executable must be a normalized absolute path")


def _looks_sensitive(key: str) -> bool:
    upper = key.upper()
    return any(
        marker in upper
        for marker in ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "PRIVATE_KEY")
    )


def _canonical_pipeline_root(value: Path, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute() or "\x00" in str(candidate):
        raise CommandPolicyError("{0} must be an absolute canonical path".format(label))
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise CommandPolicyError("{0} does not exist".format(label)) from error
    if candidate != resolved or not resolved.is_dir():
        raise CommandPolicyError(
            "{0} must be an absolute canonical directory".format(label)
        )
    return resolved


def executable_identity(executable: str) -> Mapping[str, object]:
    """Capture a minimal immutable executable identity for run artifacts."""

    path = Path(executable).resolve(strict=True)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mode": stat.st_mode,
        "mtime_ns": stat.st_mtime_ns,
        "platform": os.name,
    }
