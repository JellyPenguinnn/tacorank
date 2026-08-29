"""Trusted POSIX rlimit wrapper used instead of unsafe ``preexec_fn`` hooks."""

from __future__ import annotations

import argparse
import os
import resource
import sys
from typing import Sequence


def _bounded_setrlimit(kind: int, requested: int) -> None:
    _, hard = resource.getrlimit(kind)
    value = requested
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(kind, (value, value))


def main(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--open-files", type=int, required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    namespace = parser.parse_args(arguments)
    command = list(namespace.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command or not os.path.isabs(command[0]):
        parser.error("an absolute executable is required after --")

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # macOS rejects lowering these limits for an already-mapped interpreter or
    # a user whose current process count is above the requested per-run cap.
    # The parent still hard-enforces aggregate RSS and process-group cleanup.
    if sys.platform != "darwin":
        _bounded_setrlimit(resource.RLIMIT_AS, namespace.memory_bytes)
        if hasattr(resource, "RLIMIT_NPROC"):
            _bounded_setrlimit(resource.RLIMIT_NPROC, namespace.processes)
    _bounded_setrlimit(resource.RLIMIT_CPU, namespace.cpu_seconds)
    _bounded_setrlimit(resource.RLIMIT_NOFILE, namespace.open_files)
    os.execve(command[0], command, dict(os.environ))
    return 126


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
