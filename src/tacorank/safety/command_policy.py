"""Symbolic-command, network, dependency, and secret policy checks."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from .path_policy import PolicyViolation, ViolationCode, normalize_policy_path


REQUIRED_COMMAND_IDS = (
    "baseline_full",
    "candidate_smoke",
    "candidate_proxy",
    "candidate_full",
    "candidate_final_infer",
    "submission_check",
    "clean_reproduce",
)

DEPENDENCY_FILES = frozenset(
    {
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "poetry.lock",
        "uv.lock",
        "Pipfile",
        "Pipfile.lock",
        "setup.py",
        "setup.cfg",
        "environment.yml",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
    }
)

_COMMAND_MODULES = {"subprocess", "commands", "pty"}
_NETWORK_MODULES = {
    "aiohttp",
    "ftplib",
    "http.client",
    "httpx",
    "requests",
    "socket",
    "telnetlib",
    "urllib",
    "urllib3",
}
_DANGEROUS_CALLS = {
    "eval",
    "exec",
    "os.popen",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.system",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
}
_COMMAND_CALL_NAMES = {"resolve_command", "run_command", "execute_command"}
_SHELL_EXECUTABLES = {"bash", "dash", "fish", "sh", "zsh"}
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|CREDENTIAL)",
    re.IGNORECASE,
)

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*", re.IGNORECASE),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret(?:[_-]?key)?|password)\b"
        r"\s*[:=]\s*['\"][^'\"\s]{16,}['\"]"
    ),
)


class ResolvedCommandLike(Protocol):
    command_id: str
    argv: Sequence[str]
    cwd: Path
    environment: Mapping[str, str]
    network_enabled: bool


@dataclass(frozen=True)
class CommandCapability:
    """Reviewed shape of one symbolic command after registry resolution."""

    command_id: str
    argv_prefix: Tuple[str, ...]
    network_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.command_id or not self.argv_prefix:
            raise ValueError("command capability requires an id and argv prefix")
        if any(not isinstance(argument, str) or not argument for argument in self.argv_prefix):
            raise ValueError("argv_prefix entries must be non-empty strings")
        if Path(self.argv_prefix[0]).name in _SHELL_EXECUTABLES:
            raise ValueError("shell executables cannot be registered as capabilities")


class CommandPolicy:
    """Fail-closed validation for reviewed, already-resolved commands."""

    def __init__(self, capabilities: Iterable[CommandCapability]) -> None:
        by_id = {}
        for capability in capabilities:
            if capability.command_id in by_id:
                raise ValueError("duplicate command id: {}".format(capability.command_id))
            by_id[capability.command_id] = capability
        if not by_id:
            raise ValueError("at least one command capability is required")
        self._capabilities = by_id

    @property
    def allowed_command_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def validate_id(self, command_id: str) -> Tuple[PolicyViolation, ...]:
        if command_id in self._capabilities:
            return ()
        return (
            PolicyViolation(
                ViolationCode.UNAPPROVED_COMMAND,
                "command_policy",
                "command id is not present in the reviewed registry",
            ),
        )

    def validate_resolved(
        self, command: ResolvedCommandLike
    ) -> Tuple[PolicyViolation, ...]:
        findings = list(self.validate_id(command.command_id))
        capability = self._capabilities.get(command.command_id)
        argv = command.argv
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
            findings.append(
                PolicyViolation(
                    ViolationCode.UNAPPROVED_COMMAND,
                    "command_policy",
                    "resolved command must be an argv sequence, not a shell string",
                )
            )
            return tuple(findings)
        if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
            findings.append(
                PolicyViolation(
                    ViolationCode.UNAPPROVED_COMMAND,
                    "command_policy",
                    "resolved command contains an empty, non-string, or NUL argument",
                )
            )
        if bool(getattr(command, "shell", False)):
            findings.append(
                PolicyViolation(
                    ViolationCode.UNAPPROVED_COMMAND,
                    "command_policy",
                    "shell execution is forbidden",
                )
            )
        if capability is not None and tuple(argv[: len(capability.argv_prefix)]) != capability.argv_prefix:
            findings.append(
                PolicyViolation(
                    ViolationCode.UNAPPROVED_COMMAND,
                    "command_policy",
                    "resolved executable or fixed arguments differ from the reviewed capability",
                )
            )
        if bool(command.network_enabled) and (
            capability is None or not capability.network_allowed
        ):
            findings.append(
                PolicyViolation(
                    ViolationCode.UNAPPROVED_NETWORK,
                    "network_policy",
                    "network was enabled for a command that is sealed network-off",
                )
            )
        environment = getattr(command, "environment", None)
        if not isinstance(environment, Mapping):
            findings.append(
                PolicyViolation(
                    ViolationCode.UNAPPROVED_COMMAND,
                    "command_policy",
                    "resolved environment must be an explicit mapping",
                )
            )
        else:
            for key, value in environment.items():
                if (
                    not isinstance(key, str)
                    or not isinstance(value, str)
                    or _SENSITIVE_ENV_NAME.search(key)
                ):
                    findings.append(
                        PolicyViolation(
                            ViolationCode.SECRET_DETECTED,
                            "secret_scan",
                            "resolved command environment contains a credential-shaped entry",
                        )
                    )
                    break
                if inspect_secrets(value, "<resolved-environment>"):
                    findings.append(
                        PolicyViolation(
                            ViolationCode.SECRET_DETECTED,
                            "secret_scan",
                            "resolved command environment contains a credential-shaped value",
                        )
                    )
                    break
        working_directory = getattr(
            command, "cwd", getattr(command, "working_directory", None)
        )
        valid_working_directory = False
        if isinstance(working_directory, Path):
            valid_working_directory = working_directory.is_absolute()
        elif isinstance(working_directory, str) and working_directory:
            if Path(working_directory).is_absolute():
                valid_working_directory = "\x00" not in working_directory
            else:
                try:
                    normalize_policy_path(working_directory.rstrip("/"))
                    valid_working_directory = True
                except ValueError:
                    pass
        if not valid_working_directory:
            findings.append(
                PolicyViolation(
                    ViolationCode.UNAPPROVED_COMMAND,
                    "command_policy",
                    "working directory must be an explicit canonical path",
                )
            )
        return tuple(findings)


def inspect_source_capabilities(
    source: str,
    path: str,
    *,
    allowed_command_ids: Sequence[str],
    allowed_imports: Sequence[str] = (),
) -> Tuple[PolicyViolation, ...]:
    """Statically reject command/network capabilities introduced by source."""

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        # The syntax/import gate owns the diagnostic and stable code.
        return ()

    allowed_ids = set(allowed_command_ids)
    allowed_modules = set(allowed_imports)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if name in _NETWORK_MODULES or root in _NETWORK_MODULES:
                    if name not in allowed_modules and root not in allowed_modules:
                        findings.append(
                            _source_finding(
                                ViolationCode.UNAPPROVED_NETWORK,
                                "network_policy",
                                "candidate source imports an unapproved network module",
                                path,
                                node,
                            )
                        )
                if root in _COMMAND_MODULES and root not in allowed_modules:
                    findings.append(
                        _source_finding(
                            ViolationCode.UNAPPROVED_COMMAND,
                            "command_policy",
                            "candidate source imports an unapproved process module",
                            path,
                            node,
                        )
                    )
        if isinstance(node, ast.Call):
            call_name = _qualified_name(node.func)
            if call_name in _DANGEROUS_CALLS:
                findings.append(
                    _source_finding(
                        ViolationCode.UNAPPROVED_COMMAND,
                        "command_policy",
                        "candidate source invokes an unapproved dynamic command capability",
                        path,
                        node,
                    )
                )
            if any(
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            ):
                findings.append(
                    _source_finding(
                        ViolationCode.UNAPPROVED_COMMAND,
                        "command_policy",
                        "candidate source requests shell execution",
                        path,
                        node,
                    )
                )
            if call_name.split(".")[-1] in _COMMAND_CALL_NAMES and node.args:
                value = _literal_string(node.args[0])
                if value is None or value not in allowed_ids:
                    findings.append(
                        _source_finding(
                            ViolationCode.UNAPPROVED_COMMAND,
                            "command_policy",
                            "candidate source references a non-literal or unapproved command id",
                            path,
                            node,
                        )
                    )
            for keyword in node.keywords:
                if keyword.arg == "command_id":
                    value = _literal_string(keyword.value)
                    if value is None or value not in allowed_ids:
                        findings.append(
                            _source_finding(
                                ViolationCode.UNAPPROVED_COMMAND,
                                "command_policy",
                                "candidate source references a non-literal or unapproved command id",
                                path,
                                node,
                            )
                        )
            if call_name in {"os.getenv", "os.environ.get"} and node.args:
                environment_name = _literal_string(node.args[0])
                if environment_name and _SENSITIVE_ENV_NAME.search(environment_name):
                    findings.append(
                        _source_finding(
                            ViolationCode.SECRET_DETECTED,
                            "secret_scan",
                            "candidate source attempts to read credential-bearing environment state",
                            path,
                            node,
                        )
                    )
        if isinstance(node, ast.Subscript) and _qualified_name(node.value) == "os.environ":
            environment_name = _literal_string(node.slice)
            if environment_name and _SENSITIVE_ENV_NAME.search(environment_name):
                findings.append(
                    _source_finding(
                        ViolationCode.SECRET_DETECTED,
                        "secret_scan",
                        "candidate source attempts to read credential-bearing environment state",
                        path,
                        node,
                    )
                )
    return _deduplicate(findings)


def inspect_dependency_changes(
    changed_paths: Iterable[str], allowed_paths: Sequence[str] = ()
) -> Tuple[PolicyViolation, ...]:
    allowed = set(allowed_paths)
    findings = []
    for raw_path in changed_paths:
        try:
            path = normalize_policy_path(raw_path)
        except ValueError:
            continue
        name = PurePosixPath(path).name
        is_dependency = name in DEPENDENCY_FILES or (
            name.startswith("requirements-") and name.endswith(".txt")
        )
        if is_dependency and path not in allowed:
            findings.append(
                PolicyViolation(
                    ViolationCode.DEPENDENCY_CHANGE,
                    "dependency_policy",
                    "candidate patch changes an unreviewed dependency manifest or lockfile",
                    path,
                )
            )
    return _deduplicate(findings)


def inspect_secrets(source: str, path: str) -> Tuple[PolicyViolation, ...]:
    """Report credential-shaped values without copying secret bytes."""

    findings = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if any(pattern.search(line) for pattern in _SECRET_PATTERNS):
            findings.append(
                PolicyViolation(
                    ViolationCode.SECRET_DETECTED,
                    "secret_scan",
                    "credential-shaped value detected at line {}".format(line_number),
                    path,
                )
            )
    return _deduplicate(findings)


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return "{}.{}".format(prefix, node.attr) if prefix else node.attr
    return ""


def _literal_string(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _source_finding(
    code: ViolationCode,
    check: str,
    message: str,
    path: str,
    node: ast.AST,
) -> PolicyViolation:
    return PolicyViolation(
        code,
        check,
        "{} at line {}".format(message, getattr(node, "lineno", "?")),
        path,
    )


def _deduplicate(findings: Iterable[PolicyViolation]) -> Tuple[PolicyViolation, ...]:
    unique = {}
    for finding in findings:
        key = (finding.code.value, finding.check, finding.path, finding.message)
        unique[key] = finding
    order = sorted(unique, key=lambda item: tuple(part or "" for part in item))
    return tuple(unique[key] for key in order)
