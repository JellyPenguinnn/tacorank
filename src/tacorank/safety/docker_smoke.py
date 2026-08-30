"""Hardened import smoke for an unaccepted candidate entrypoint."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from typing import Any, Mapping, Tuple

from ..docker_host import normalize_local_docker_host
from .patch_gate import SMOKE_ISOLATION_CAPABILITY


_IMAGE = re.compile(r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
_IMPORT_SCRIPT = """import csv
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

module_name, separator, symbol = sys.argv[1].partition(":")
if not separator:
    raise SystemExit("entrypoint must use module:symbol syntax")
implementation = getattr(importlib.import_module(module_name), symbol)
if not callable(implementation):
    raise SystemExit("entrypoint is not callable")
parameters = tuple(inspect.signature(implementation).parameters)
if parameters != ("invocation",):
    raise SystemExit("entrypoint signature must be run(invocation)")

with tempfile.TemporaryDirectory(dir=str(Path("/tmp").resolve())) as temporary:
    root = Path(temporary)
    input_root = root / "input"
    input_root.mkdir()
    score = input_root / "score.csv"
    score.write_text(
        "row_id,date,user_id,video_id,author_id,tab,duration_ms\\n"
        "0,20220403,1,10,100,0,1000.0\\n"
        "1,20220403,1,11,101,0,2000.0\\n",
        encoding="utf-8",
    )
    train = input_root / "train.csv"
    train.write_text(
        "date,user_id,video_id,author_id,tab,duration_ms,long_view\\n"
        "20220401,1,10,100,0,1000.0,1\\n"
        "20220402,1,11,101,0,2000.0,0\\n",
        encoding="utf-8",
    )
    parent = input_root / "fm_baseline_predictions.csv"
    parent.write_text(
        "row_id,user_id,video_id,score\\n0,1,10,0.25\\n1,1,11,-0.5\\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(parent.read_bytes()).hexdigest()
    (input_root / "fm_baseline_predictions.sha256").write_text(
        digest + "\\n", encoding="ascii"
    )
    history_columns = (
        "row_id", "date", "time_ms", "user_id", "video_id",
        "history_exposure", "global_positive_rate", "tag_exposure",
        "tag_positive", "tag_coverage", "tag_positive_age_days",
        "author_exposure", "author_positive", "author_positive_age_days",
        "duration_exposure", "duration_positive", "tab_exposure",
        "tab_positive", "hour_sin", "hour_cos", "weekday_sin",
        "weekday_cos", "log_duration_scaled", "item_age_scaled",
        "duration_bucket",
    )
    history_train = input_root / "history_train.csv"
    history_train.write_text(
        ",".join((*history_columns, "long_view")) + "\\n"
        "0,20220401,1648771200000,1,10,0,0.5,0,0,0,-1,0,0,-1,0,0,0,0,0,1,0.5,-0.5,0.5,0.1,le_7s,1\\n"
        "1,20220402,1648857600000,1,11,1,0.5,2,1,1,1,1,0,-1,1,0,1,0,0.5,0.5,-0.5,-0.5,0.7,0.2,le_18s,0\\n",
        encoding="utf-8",
    )
    history_score = input_root / "history_score.csv"
    history_score.write_text(
        ",".join(history_columns) + "\\n"
        "0,20220403,1648944000000,1,10,2,0.5,3,2,1,1,2,1,1,2,1,2,1,0,1,0.5,-0.5,0.5,0.1,le_7s\\n"
        "1,20220403,1648944001000,1,11,2,0.5,3,1,1,1,2,0,-1,2,0,2,0,0,1,0.5,-0.5,0.7,0.2,le_18s\\n",
        encoding="utf-8",
    )
    history_manifest = {
        "schema_version": "1.0",
        "train_file": history_train.name,
        "train_sha256": hashlib.sha256(history_train.read_bytes()).hexdigest(),
        "score_file": history_score.name,
        "score_sha256": hashlib.sha256(history_score.read_bytes()).hexdigest(),
        "train_columns": [*history_columns, "long_view"],
        "score_columns": list(history_columns),
        "feature_columns": list(history_columns[5:]),
        "history_update_policy": "emit_before_update_train_frozen_for_score",
        "static_feature_policy": "candidate_conditioned_context_only",
    }
    (input_root / "history-feature-manifest.json").write_text(
        json.dumps(history_manifest, sort_keys=True), encoding="utf-8"
    )
    outputs = []
    for attempt in (1, 2):
        output_root = root / ("output-%d" % attempt)
        output_root.mkdir()
        output = output_root / "predictions.csv"
        implementation(
            SimpleNamespace(
                input_root=input_root,
                output_path=output,
                fidelity="smoke",
                seed=17,
                mode="candidate",
                contract_root=root,
                clean_reproduce=False,
            )
        )
        rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
        if len(rows) != 2 or [int(row["row_id"]) for row in rows] != [0, 1]:
            raise SystemExit("synthetic output rows are invalid")
        if not all(math.isfinite(float(row["score"])) for row in rows):
            raise SystemExit("synthetic output scores are non-finite")
        if not (output_root / "training-diagnostics.json").is_file():
            raise SystemExit("training diagnostics are missing")
        outputs.append(output.read_bytes())
    if outputs[0] != outputs[1]:
        raise SystemExit("candidate is nondeterministic for a fixed seed")

scaffold = importlib.import_module("solution.research_scaffold")
self_test = getattr(scaffold, "self_test", None)
if not callable(self_test):
    raise SystemExit("research scaffold self-test is missing")
self_test()
"""


class DockerEntrypointSmokeCheck:
    """Import the exact candidate entrypoint with no network or writable checkout."""

    isolation_capability = SMOKE_ISOLATION_CAPABILITY

    def __init__(
        self,
        *,
        docker_executable: Path,
        docker_host: str,
        image: str,
        container_python_executable: str,
        entrypoint: str,
        timeout_seconds: int = 60,
        memory_limit_mb: int = 2048,
        pids_limit: int = 64,
        cpu_limit: float = 1.0,
        tmpfs_limit_mb: int = 128,
        max_output_bytes: int = 64 * 1024,
    ) -> None:
        docker = Path(docker_executable)
        if (
            not docker.is_absolute()
            or docker.is_symlink()
            or not docker.is_file()
            or docker.resolve(strict=True) != docker
            or not os.access(docker, os.X_OK)
        ):
            raise ValueError("Gate A Docker executable must be canonical and executable")
        if not _IMAGE.fullmatch(image):
            raise ValueError("Gate A Docker image must be pinned by sha256 digest")
        if not container_python_executable.startswith("/"):
            raise ValueError("container Python executable must be absolute")
        if not _ENTRYPOINT.fullmatch(entrypoint):
            raise ValueError("candidate entrypoint must use canonical module:symbol syntax")
        for value, label in (
            (timeout_seconds, "timeout_seconds"),
            (memory_limit_mb, "memory_limit_mb"),
            (pids_limit, "pids_limit"),
            (tmpfs_limit_mb, "tmpfs_limit_mb"),
            (max_output_bytes, "max_output_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(cpu_limit, bool)
            or not isinstance(cpu_limit, (int, float))
            or cpu_limit <= 0
        ):
            raise ValueError("cpu_limit must be positive")
        self.docker_executable = docker
        self.docker_host = normalize_local_docker_host(docker_host)
        self.image = image
        self.container_python_executable = container_python_executable
        self.entrypoint = entrypoint
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.pids_limit = pids_limit
        self.cpu_limit = float(cpu_limit)
        self.tmpfs_limit_mb = tmpfs_limit_mb
        self.max_output_bytes = max_output_bytes

    def run(self, repository_root: Path, candidate: Any) -> Tuple[bool, str]:
        root = Path(repository_root).resolve(strict=True)
        if any(character in str(root) for character in (",", "\n", "\r", "\x00")):
            return False, "candidate worktree path cannot be mounted safely"
        identity = "%s\0%s\0%s" % (
            getattr(candidate, "run_id", ""),
            getattr(candidate, "experiment_id", ""),
            getattr(candidate, "patch_commit_sha", ""),
        )
        get_uid = getattr(os, "getuid", lambda: 65534)
        get_gid = getattr(os, "getgid", lambda: 65534)
        memory = f"{self.memory_limit_mb}m"
        cpu = format(self.cpu_limit, ".3f").rstrip("0").rstrip(".")
        mount = (
            f"type=bind,src={root},dst=/workspace,readonly,bind-propagation=rprivate"
        )
        with tempfile.TemporaryDirectory(prefix="tacorank-gatea-docker-") as temporary:
            nonce = hashlib.sha256(
                (identity + "\0" + temporary).encode("utf-8")
            ).hexdigest()[:20]
            name = "tacorank-gatea-" + nonce
            config_root = Path(temporary) / "docker-config"
            config_root.mkdir(mode=0o700)
            cidfile = Path(temporary) / "container.cid"
            command = (
                str(self.docker_executable),
                "run",
                "--cidfile",
                str(cidfile),
                "--name",
                name,
                "--pull",
                "never",
                "--network",
                "none",
                "--log-driver",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--pids-limit",
                str(self.pids_limit),
                "--memory",
                memory,
                "--memory-swap",
                memory,
                "--cpus",
                cpu,
                "--user",
                f"{get_uid()}:{get_gid()}",
                "--tmpfs",
                f"/tmp:rw,nosuid,nodev,noexec,size={self.tmpfs_limit_mb}m",
                "--mount",
                mount,
                "--workdir",
                "/workspace",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--entrypoint",
                self.container_python_executable,
                self.image,
                "-B",
                "-c",
                _IMPORT_SCRIPT,
                self.entrypoint,
            )
            environment = {
                "DOCKER_CONFIG": str(config_root),
                "DOCKER_HOST": self.docker_host,
            }
            try:
                return_code, timed_out, output_limited, raw_output = _bounded_run(
                    command,
                    cwd=root,
                    environment=environment,
                    timeout=self.timeout_seconds,
                    max_output_bytes=self.max_output_bytes,
                )
                output = raw_output.decode("utf-8", errors="replace")
                if timed_out:
                    return False, "isolated entrypoint import exceeded its wall-time limit"
                if output_limited:
                    return False, "isolated entrypoint import exceeded its output limit"
                if return_code != 0:
                    return False, _safe_failure_summary(output)
                return True, "isolated candidate synthetic execution succeeded"
            except OSError:
                return False, "isolated entrypoint import could not launch Docker"
            finally:
                self._remove_created_container(
                    root=root,
                    environment=environment,
                    cidfile=cidfile,
                )

    def _remove_created_container(
        self,
        *,
        root: Path,
        environment: Mapping[str, str],
        cidfile: Path,
    ) -> None:
        if not cidfile.exists():
            return
        try:
            if cidfile.is_symlink() or not cidfile.is_file() or cidfile.stat().st_size > 128:
                raise RuntimeError("Gate A Docker cidfile is invalid")
            container_id = cidfile.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError("Gate A Docker cidfile could not be read") from exc
        if not _CONTAINER_ID.fullmatch(container_id):
            raise RuntimeError("Gate A Docker cidfile contains an invalid identity")
        try:
            removed = subprocess.run(
                (str(self.docker_executable), "rm", "--force", container_id),
                cwd=root,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=min(self.timeout_seconds, 30),
                check=False,
                shell=False,
            )
            inspected = subprocess.run(
                (
                    str(self.docker_executable),
                    "inspect",
                    "--type",
                    "container",
                    container_id,
                ),
                cwd=root,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=min(self.timeout_seconds, 30),
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Gate A could not remove its Docker smoke container") from exc
        if removed.returncode != 0 or inspected.returncode == 0:
            raise RuntimeError("Gate A could not prove Docker smoke container removal")


def _safe_failure_summary(output: str) -> str:
    compact = " ".join(output.split())
    if not compact:
        return "isolated candidate synthetic execution failed"
    # Candidate output is untrusted and may contain structured payloads. Keep a
    # short printable diagnostic without relaying JSON-like content to prompts.
    compact = compact.replace("{", "(").replace("}", ")")
    return "isolated candidate synthetic execution failed: " + compact[-1000:]


def _bounded_run(
    command: Tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: int,
    max_output_bytes: int,
) -> tuple[int, bool, bool, bytes]:
    """Run Docker with a hard in-memory output bound."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
    )
    output = bytearray()
    output_limited = False

    def drain() -> None:
        nonlocal output_limited
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                return
            remaining = max_output_bytes - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                output_limited = True
                try:
                    process.kill()
                except OSError:
                    pass

    reader = threading.Thread(target=drain, name="tacorank-gatea-output", daemon=True)
    reader.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        return_code = process.wait(timeout=10)
    finally:
        reader.join(timeout=10)
        if process.stdout is not None:
            process.stdout.close()
    return return_code, timed_out, output_limited, bytes(output)
