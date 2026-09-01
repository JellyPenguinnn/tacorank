"""Subprocess lifecycle with bounded logs and process-group cleanup."""

from __future__ import annotations

import codecs
import errno
import json
import os
import re
import signal
import stat
import subprocess
import tarfile
import threading
import time
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Deque, Mapping, Optional, TextIO, Tuple

from tacorank.execution.sandbox import (
    LaunchSpec,
    ResourceLimits,
    RuntimeCleanupSpec,
    RuntimeMetricsSpec,
    RuntimeOutputExtractionSpec,
)


_RUNTIME_NOT_READY_EXIT_CODE = 75


class ProcessLaunchError(RuntimeError):
    """Raised when the executor cannot create the reviewed child process."""


class OutputQuotaExceeded(ProcessLaunchError):
    """Raised when allowlisted runtime output exceeds its hard byte quota."""


class DiskSpaceExhausted(ProcessLaunchError):
    """Raised when output extraction cannot write because storage is full."""


_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
_HARD_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _windows_taskkill(process_id: int, *, force: bool) -> bool:
    """Terminate a Windows process tree without invoking a shell.

    ``os.killpg`` has no Windows equivalent.  The launcher creates a
    dedicated process group there, and ``taskkill /T`` is the native way to
    apply termination to the leader and its descendants.  A failed command
    is deliberately reported to the caller so it can fall back to the
    ``Popen`` leader operation.
    """

    # Windows has no portable group-level graceful signal.  Without /F,
    # console children can ignore the request and leave descendants alive;
    # use the bounded forced tree operation for both termination paths.
    command = ["taskkill.exe", "/PID", str(process_id), "/T", "/F"]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def redact_runtime_output(text: str) -> str:
    """Remove common credential forms before they reach persistent logs."""

    redacted = text
    redacted = _SECRET_PATTERNS[0].sub(r"\1\2[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[1].sub("Bearer [REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[2].sub("[REDACTED]", redacted)
    return redacted


class ManagedProcess:
    """Own one child, its output reader, and its entire process group."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        log_handle: TextIO,
        limits: ResourceLimits,
        runtime_cleanup: Optional[RuntimeCleanupSpec] = None,
        runtime_metrics: Optional[RuntimeMetricsSpec] = None,
        output_filter: Callable[[str], str] = redact_runtime_output,
    ) -> None:
        self._process = process
        self._limits = limits
        self._log_handle = log_handle
        self._runtime_cleanup = runtime_cleanup
        self._runtime_metrics = runtime_metrics
        self._output_filter = output_filter
        self._lock = threading.Lock()
        self._recent: Deque[str] = deque(maxlen=128)
        self._recent_length = 0
        self._pending_preview = ""
        self._last_output_monotonic = time.monotonic()
        self._written_bytes = 0
        self._log_truncated = False
        self._reader_error: Optional[BaseException] = None
        self._runtime_state: Mapping[str, Any] = {}
        self._runtime_outputs_extracted = False
        self._reader = threading.Thread(
            target=self._read_output,
            name="tacorank-output-{0}".format(process.pid),
            daemon=True,
        )
        self._reader.start()

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def process_group_id(self) -> int:
        return self._process.pid

    @property
    def return_code(self) -> Optional[int]:
        return self._process.poll()

    @property
    def reader_error(self) -> Optional[BaseException]:
        return self._reader_error

    @property
    def runtime_metrics(self) -> Optional[RuntimeMetricsSpec]:
        return self._runtime_metrics

    @property
    def runtime_state(self) -> Mapping[str, Any]:
        return self._runtime_state

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def last_output_age_ms(self, now: Optional[float] = None) -> int:
        current = time.monotonic() if now is None else now
        with self._lock:
            age = current - self._last_output_monotonic
        return max(0, int(age * 1000))

    def recent_output_tail(self, limit: int = 4096) -> str:
        with self._lock:
            joined = "".join(self._recent) + self._pending_preview
        return joined[-limit:]

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def terminate_group(self, grace_seconds: Optional[float] = None) -> int:
        """Terminate then kill every process in the isolated process group."""

        grace = (
            self._limits.termination_grace_seconds
            if grace_seconds is None
            else max(0.0, grace_seconds)
        )
        self._signal_group(signal.SIGTERM)
        deadline = time.monotonic() + grace
        while self._group_exists() and time.monotonic() < deadline:
            self._process.poll()
            time.sleep(0.02)
        cleanup_error: Optional[ProcessLaunchError] = None
        if self._runtime_cleanup is not None:
            try:
                self._terminate_external_runtime()
            except ProcessLaunchError as error:
                cleanup_error = error
        if self._group_exists():
            self._signal_group(_HARD_KILL_SIGNAL, force=True)
        try:
            return_code = self._process.wait(timeout=max(1.0, grace + 0.5))
        except subprocess.TimeoutExpired as error:
            raise ProcessLaunchError("failed to reap child process") from error
        try:
            self._require_external_runtime_absent()
        except ProcessLaunchError as error:
            cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            raise cleanup_error
        return return_code

    def finish(self) -> int:
        """Reap the leader and remove any background descendants."""

        return_code = self._process.wait()
        if self._group_exists():
            self._signal_group(signal.SIGTERM)
            deadline = time.monotonic() + self._limits.termination_grace_seconds
            while self._group_exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            if self._group_exists():
                self._signal_group(_HARD_KILL_SIGNAL, force=True)
        self._close_reader()
        state_error: Optional[ProcessLaunchError] = None
        try:
            self._capture_external_runtime_state()
        except ProcessLaunchError as error:
            state_error = error
        extraction_error: Optional[ProcessLaunchError] = None
        try:
            cleanup = self._runtime_cleanup
            supervised = cleanup is not None and cleanup.completion_argv is not None
            if supervised and not self._runtime_outputs_extracted:
                raise ProcessLaunchError(
                    "container exited before live output extraction completed"
                )
            if not self._runtime_outputs_extracted:
                self._extract_external_runtime_outputs()
        except ProcessLaunchError as error:
            extraction_error = error
        cleanup_error: Optional[ProcessLaunchError] = None
        try:
            if self._runtime_cleanup is not None:
                self._terminate_external_runtime()
            self._require_external_runtime_absent()
        except ProcessLaunchError as error:
            cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error
        if extraction_error is not None:
            raise extraction_error
        if state_error is not None:
            raise state_error
        return return_code

    def close_after_termination(self) -> None:
        self._close_reader()

    def runtime_outputs_ready(self) -> bool:
        """Return whether the trusted supervisor has finished the candidate."""

        cleanup = self._runtime_cleanup
        if cleanup is None or cleanup.completion_argv is None:
            return False
        completed = self._run_cleanup_command(cleanup.completion_argv)
        if completed.returncode == 0:
            return True
        if completed.returncode == _RUNTIME_NOT_READY_EXIT_CODE:
            return False
        raise ProcessLaunchError("container output completion probe failed")

    def extract_ready_runtime_outputs(self) -> None:
        """Copy bounded tmpfs outputs while live, then release the supervisor."""

        cleanup = self._runtime_cleanup
        if (
            cleanup is None
            or cleanup.output_extraction is None
            or cleanup.completion_argv is None
            or cleanup.release_argv is None
        ):
            raise ProcessLaunchError("live runtime output handshake is unavailable")
        if self._runtime_outputs_extracted:
            raise ProcessLaunchError("runtime outputs were already extracted")
        if not self.runtime_outputs_ready():
            raise ProcessLaunchError("candidate has not completed output production")
        self._extract_external_runtime_outputs()
        self._runtime_outputs_extracted = True
        released = self._run_cleanup_command(cleanup.release_argv)
        if released.returncode != 0:
            raise ProcessLaunchError("container output release failed")

    def _signal_group(
        self, requested_signal: signal.Signals, *, force: bool = False
    ) -> None:
        if os.name == "posix":
            try:
                os.killpg(self.process_group_id, requested_signal)
            except ProcessLookupError:
                return
        elif self._process.poll() is None:
            if not _windows_taskkill(self.process_group_id, force=force):
                if force:
                    self._process.kill()
                else:
                    self._process.terminate()

    def _group_exists(self) -> bool:
        if os.name != "posix":
            return self._process.poll() is None
        try:
            os.killpg(self.process_group_id, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _close_reader(self) -> None:
        pipe = self._process.stdout
        self._reader.join(timeout=2.0)
        if self._reader.is_alive() and pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass
            self._reader.join(timeout=2.0)

    def _terminate_external_runtime(self) -> None:
        cleanup = self._runtime_cleanup
        if cleanup is None:
            return
        completed = self._run_cleanup_command(cleanup.terminate_argv)
        if completed.returncode != 0 and not self._external_runtime_absent():
            raise ProcessLaunchError("container runtime cleanup failed")

    def _extract_external_runtime_outputs(self) -> None:
        cleanup = self._runtime_cleanup
        if cleanup is None or cleanup.output_extraction is None:
            return
        _extract_bounded_tar(cleanup.output_extraction, cleanup.environment)

    def _require_external_runtime_absent(self) -> None:
        if self._runtime_cleanup is None:
            return
        deadline = time.monotonic() + self._runtime_cleanup.timeout_seconds
        while time.monotonic() < deadline:
            if self._external_runtime_absent():
                return
            time.sleep(0.05)
        raise ProcessLaunchError("container runtime object survived cleanup")

    def _external_runtime_absent(self) -> bool:
        cleanup = self._runtime_cleanup
        if cleanup is None:
            return True
        inspected = self._run_cleanup_command(
            cleanup.inspect_argv, capture_output=True
        )
        if inspected.returncode != 0:
            health = self._run_cleanup_command(cleanup.healthcheck_argv)
            if health.returncode != 0:
                raise ProcessLaunchError(
                    "cannot prove container cleanup because the runtime is unavailable"
                )
            raise ProcessLaunchError(
                "cannot prove container cleanup because the absence probe failed"
            )
        return not inspected.stdout.strip()

    def _capture_external_runtime_state(self) -> None:
        cleanup = self._runtime_cleanup
        if cleanup is None or cleanup.state_argv is None:
            return
        completed = self._run_cleanup_command(
            cleanup.state_argv, capture_output=True
        )
        if completed.returncode != 0:
            raise ProcessLaunchError("container runtime state could not be inspected")
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProcessLaunchError("container runtime state is malformed") from error
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("OOMKilled"), bool)
            or not isinstance(payload.get("ExitCode"), int)
        ):
            raise ProcessLaunchError("container runtime state is malformed")
        self._runtime_state = payload

    def _run_cleanup_command(
        self, argv: Tuple[str, ...], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        cleanup = self._runtime_cleanup
        if cleanup is None:
            raise ProcessLaunchError("runtime cleanup specification is missing")
        try:
            return subprocess.run(
                list(argv),
                cwd=str(cleanup.cwd),
                env=dict(cleanup.environment),
                stdin=subprocess.DEVNULL,
                stdout=(subprocess.PIPE if capture_output else subprocess.DEVNULL),
                stderr=subprocess.DEVNULL,
                shell=False,
                close_fds=True,
                timeout=cleanup.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ProcessLaunchError("container runtime cleanup failed") from error

    def _read_output(self) -> None:
        pipe = self._process.stdout
        if pipe is None:
            self._log_handle.close()
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        carry = ""
        try:
            with self._log_handle as log:
                while True:
                    try:
                        block = os.read(pipe.fileno(), 4096)
                    except OSError as error:
                        if error.errno == errno.EBADF:
                            break
                        raise
                    if not block:
                        break
                    with self._lock:
                        self._last_output_monotonic = time.monotonic()
                    decoded = carry + decoder.decode(block)
                    newline = decoded.rfind("\n")
                    if newline >= 0:
                        emit, carry = decoded[: newline + 1], decoded[newline + 1 :]
                        self._emit(log, self._output_filter(emit))
                    elif len(decoded) > 8192:
                        cutoff = decoded.rfind(" ", 0, len(decoded) - 1024)
                        if cutoff < 0:
                            cutoff = len(decoded) - 1024
                        emit, carry = decoded[:cutoff], decoded[cutoff:]
                        self._emit(log, self._output_filter(emit))
                    else:
                        carry = decoded
                    with self._lock:
                        self._pending_preview = self._output_filter(carry)
                final = carry + decoder.decode(b"", final=True)
                if final:
                    self._emit(log, self._output_filter(final))
                with self._lock:
                    self._pending_preview = ""
                if self._log_truncated:
                    marker = "\n[TacoRank log truncated at configured byte limit]\n"
                    log.write(marker)
                    self._remember(marker)
        except BaseException as error:  # reader failures are surfaced by runner
            self._reader_error = error

    def _emit(self, log: object, text: str) -> None:
        if not text:
            return
        encoded_size = len(text.encode("utf-8"))
        remaining = self._limits.max_log_bytes - self._written_bytes
        if remaining > 0:
            if encoded_size <= remaining:
                persisted = text
            else:
                persisted = text.encode("utf-8")[:remaining].decode(
                    "utf-8", errors="ignore"
                )
                self._log_truncated = True
            log.write(persisted)  # type: ignore[attr-defined]
            log.flush()  # type: ignore[attr-defined]
            self._written_bytes += len(persisted.encode("utf-8"))
        else:
            self._log_truncated = True
        self._remember(text)

    def _remember(self, text: str) -> None:
        with self._lock:
            self._recent.append(text)
            self._recent_length += len(text)
            while self._recent_length > 8192 and self._recent:
                removed = self._recent.popleft()
                self._recent_length -= len(removed)


class ProcessLauncher:
    """Injectable launcher used by the execution runner."""

    def launch(
        self, specification: LaunchSpec, log_path: Path, limits: ResourceLimits
    ) -> ManagedProcess:
        destination = Path(log_path)
        if not destination.is_absolute():
            raise ProcessLaunchError("runtime log path must be absolute")
        log_handle, parent_descriptor, identity = _open_exclusive_log(destination)
        try:
            process = subprocess.Popen(
                list(specification.argv),
                cwd=str(specification.cwd),
                env=dict(specification.environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                close_fds=True,
                start_new_session=specification.start_new_session,
                preexec_fn=specification.preexec_fn,
                bufsize=0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            log_handle.close()
            _unlink_owned_log(destination, parent_descriptor, identity)
            if parent_descriptor is not None:
                os.close(parent_descriptor)
            raise ProcessLaunchError(
                "unable to launch reviewed command: {0}".format(type(error).__name__)
            ) from error
        try:
            return ManagedProcess(
                process,
                log_handle,
                limits,
                runtime_cleanup=specification.runtime_cleanup,
                runtime_metrics=specification.runtime_metrics,
            )
        except BaseException:
            log_handle.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            cleanup_error: Optional[ProcessLaunchError] = None
            if specification.runtime_cleanup is not None:
                try:
                    _force_cleanup_after_launch_failure(
                        specification.runtime_cleanup
                    )
                except ProcessLaunchError as error:
                    cleanup_error = error
            _unlink_owned_log(destination, parent_descriptor, identity)
            if cleanup_error is not None:
                raise cleanup_error
            raise
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)


def _force_cleanup_after_launch_failure(cleanup: RuntimeCleanupSpec) -> None:
    """Remove a runtime object if ManagedProcess construction itself fails."""

    def run(
        argv: Tuple[str, ...], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                list(argv),
                cwd=str(cleanup.cwd),
                env=dict(cleanup.environment),
                stdin=subprocess.DEVNULL,
                stdout=(subprocess.PIPE if capture_output else subprocess.DEVNULL),
                stderr=subprocess.DEVNULL,
                shell=False,
                close_fds=True,
                timeout=cleanup.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ProcessLaunchError(
                "container runtime cleanup failed after launch"
            ) from error

    run(cleanup.terminate_argv)
    inspected = run(cleanup.inspect_argv, capture_output=True)
    if inspected.returncode != 0:
        if run(cleanup.healthcheck_argv).returncode != 0:
            raise ProcessLaunchError(
                "cannot prove post-launch cleanup because runtime is unavailable"
            )
        raise ProcessLaunchError("post-launch runtime absence probe failed")
    if inspected.stdout.strip():
        raise ProcessLaunchError("runtime object survived post-launch cleanup")


def _extract_bounded_tar(
    specification: RuntimeOutputExtractionSpec,
    environment: Mapping[str, str],
) -> None:
    """Extract only reviewed regular files from a bounded Docker tar stream."""

    destination = Path(specification.destination)
    if not destination.is_absolute():
        raise ProcessLaunchError("container output destination must be absolute")
    _reject_symlink_components(destination)
    try:
        resolved_destination = destination.resolve(strict=True)
    except OSError as error:
        raise ProcessLaunchError(
            "container output destination is unavailable"
        ) from error
    if resolved_destination != destination or not destination.is_dir():
        raise ProcessLaunchError("container output destination must be canonical")
    if specification.max_bytes < 1 or specification.timeout_seconds <= 0:
        raise ProcessLaunchError("container output extraction bounds are invalid")

    allowed = {
        _normalize_output_relative_path(value)
        for value in specification.allowed_relative_paths
    }
    if len(allowed) != len(specification.allowed_relative_paths):
        raise ProcessLaunchError("container output allowlist contains duplicates")
    allowed_directories = {
        parent.as_posix()
        for relative in allowed
        for parent in PurePosixPath(relative).parents
        if parent != PurePosixPath(".")
    }
    created_files: list[Path] = []
    extraction_errors: list[BaseException] = []
    try:
        process = subprocess.Popen(
            list(specification.argv),
            cwd=str(destination),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ProcessLaunchError(
            "container output extraction failed: "
            f"{type(error).__name__}: {error}"
        ) from error

    def consume() -> None:
        total_bytes = 0
        seen: set[str] = set()
        try:
            if process.stdout is None:
                raise ProcessLaunchError("container output stream is unavailable")
            with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
                for member in archive:
                    relative = _normalize_tar_member(member.name)
                    if relative is None:
                        if member.isdir():
                            continue
                        raise ProcessLaunchError(
                            "container output archive contains an invalid root member"
                        )
                    if member.isdir():
                        if relative not in allowed_directories:
                            raise ProcessLaunchError(
                                "container output archive contains an unexpected directory"
                            )
                        continue
                    if not member.isfile() or relative not in allowed:
                        raise ProcessLaunchError(
                            "container output archive contains an unexpected member"
                        )
                    if relative in seen:
                        raise ProcessLaunchError(
                            "container output archive contains a duplicate member"
                        )
                    if member.size < 0 or total_bytes + member.size > specification.max_bytes:
                        raise OutputQuotaExceeded(
                            "container output archive exceeds the hard byte limit"
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise ProcessLaunchError(
                            "container output archive member is unreadable"
                        )
                    target = destination.joinpath(*PurePosixPath(relative).parts)
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    _reject_symlink_components(target.parent)
                    if target.parent.resolve(strict=True) != target.parent:
                        raise ProcessLaunchError(
                            "container output parent must be canonical"
                        )
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    flags |= getattr(os, "O_CLOEXEC", 0)
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                    descriptor = os.open(str(target), flags, 0o600)
                    created_files.append(target)
                    written = 0
                    try:
                        output = os.fdopen(descriptor, "wb")
                    except BaseException:
                        os.close(descriptor)
                        raise
                    try:
                        with output:
                            while written < member.size:
                                block = source.read(min(1024 * 1024, member.size - written))
                                if not block:
                                    raise ProcessLaunchError(
                                        "container output archive member is truncated"
                                    )
                                output.write(block)
                                written += len(block)
                            output.flush()
                            os.fsync(output.fileno())
                    except BaseException:
                        raise
                    total_bytes += written
                    seen.add(relative)
        except BaseException as error:
            extraction_errors.append(error)

    worker = threading.Thread(
        target=consume,
        name="tacorank-container-output",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=specification.timeout_seconds)
    if worker.is_alive():
        process.kill()
        worker.join(timeout=2.0)
        extraction_errors.append(
            ProcessLaunchError("container output extraction timed out")
        )
    try:
        return_code = process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return_code = -1
    if return_code != 0:
        detail = ""
        if process.stderr is not None:
            try:
                detail = process.stderr.read().decode("utf-8", errors="replace").strip()
            except OSError:
                detail = ""
        if detail:
            detail = " " + detail[-512:]
        extraction_errors.append(
            ProcessLaunchError(
                "container output copy command failed "
                f"(exit {return_code}).{detail}"
            )
        )
    if extraction_errors:
        for created in reversed(created_files):
            try:
                created.unlink()
            except FileNotFoundError:
                pass
        first = extraction_errors[0]
        if isinstance(first, OutputQuotaExceeded):
            raise first
        if isinstance(first, OSError) and first.errno == errno.ENOSPC:
            raise DiskSpaceExhausted(
                "container output extraction ran out of disk space"
            ) from first
        raise ProcessLaunchError(
            "container output extraction failed: "
            f"{type(first).__name__}: {first}"
        ) from first


def _normalize_output_relative_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ProcessLaunchError("container output allowlist path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ProcessLaunchError("container output allowlist path is invalid")
    return path.as_posix()


def _normalize_tar_member(value: str) -> Optional[str]:
    if not value or "\\" in value or "\x00" in value:
        raise ProcessLaunchError("container output archive path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ProcessLaunchError("container output archive path is invalid")
    parts = [part for part in path.parts if part not in {"", "."}]
    if parts and parts[0] == "artifacts":
        parts = parts[1:]
    if not parts:
        return None
    return PurePosixPath(*parts).as_posix()


def _open_exclusive_log(
    destination: Path,
) -> Tuple[TextIO, Optional[int], os.stat_result]:
    """Create a regular log once and retain its descriptor across launch."""

    _reject_symlink_components(destination.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor: Optional[int] = None
    try:
        if os.name == "posix":
            parent_descriptor = _open_directory_without_symlinks(destination.parent)
            descriptor = os.open(
                destination.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        else:
            descriptor = os.open(str(destination), flags, 0o600)
    except (FileExistsError, OSError) as error:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise ProcessLaunchError("runtime log path cannot be created exclusively") from error

    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise ProcessLaunchError("runtime log destination is not a regular file")
        # Windows does not expose fchmod; the exclusive create above still
        # binds this handle to the reviewed path before applying its closest
        # available permission mode.
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        else:
            os.chmod(str(destination), 0o600)
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
    except BaseException:
        os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise
    return handle, parent_descriptor, identity


def _unlink_owned_log(
    destination: Path,
    parent_descriptor: Optional[int],
    identity: os.stat_result,
) -> None:
    """Remove only the exact inode this launcher created."""

    try:
        if parent_descriptor is not None:
            current = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                os.unlink(destination.name, dir_fd=parent_descriptor)
        else:
            current = destination.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                destination.unlink()
    except FileNotFoundError:
        return


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ProcessLaunchError("runtime log parent contains a symbolic link")
        if current == current.parent:
            return
        current = current.parent


def _open_directory_without_symlinks(path: Path) -> int:
    """Walk an absolute directory with ``openat`` and pin every component."""

    if not path.is_absolute():
        raise ProcessLaunchError("runtime log parent must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
