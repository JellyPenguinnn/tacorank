"""Trusted in-container handshake for bounded tmpfs output extraction."""

from __future__ import annotations

import argparse
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


_CONTROL_DIRECTORY = re.compile(r"^/tmp/tacorank-[0-9a-f]{24}-control$")
_COMPLETE = "candidate-complete"
_RELEASE = "output-captured"
NOT_READY_EXIT_CODE = 75


class SupervisorError(RuntimeError):
    """The controller/candidate output handshake violated its contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", allow_abbrev=False)
    run.add_argument("--control-directory", required=True)
    run.add_argument("--uid", type=int, required=True)
    run.add_argument("--gid", type=int, required=True)
    run.add_argument("candidate_arguments", nargs=argparse.REMAINDER)
    self_test = commands.add_parser("self-test", allow_abbrev=False)
    self_test.add_argument("--uid", type=int, required=True)
    self_test.add_argument("--gid", type=int, required=True)
    for name in ("probe", "release"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--control-directory", required=True)
    return parser


def _control_directory(value: str) -> Path:
    if not _CONTROL_DIRECTORY.fullmatch(value):
        raise SupervisorError("control directory does not match the reviewed path")
    return Path(value)


def _owned_control_directory(path: Path) -> os.stat_result:
    try:
        identity = path.lstat()
    except OSError as error:
        raise SupervisorError("control directory is unavailable") from error
    if (
        not stat.S_ISDIR(identity.st_mode)
        or stat.S_IMODE(identity.st_mode) != 0o700
        or identity.st_uid != os.geteuid()
    ):
        raise SupervisorError("control directory identity is invalid")
    return identity


def _write_exclusive(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _candidate_preexec(uid: int, gid: int):
    if uid < 0 or gid < 0:
        raise SupervisorError("candidate uid and gid must be non-negative")
    current_uid = os.geteuid()
    current_gid = os.getegid()
    if current_uid != 0:
        if (uid, gid) != (current_uid, current_gid):
            raise SupervisorError("non-root supervisor cannot change candidate identity")
        return None

    def demote() -> None:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
        os.umask(0o077)

    return demote


def _run(path: Path, uid: int, gid: int, arguments: Sequence[str]) -> int:
    candidate_arguments = list(arguments)
    if candidate_arguments[:1] == ["--"]:
        candidate_arguments = candidate_arguments[1:]
    if not candidate_arguments or any(
        "\x00" in value for value in candidate_arguments
    ):
        raise SupervisorError("candidate arguments are missing or invalid")
    try:
        path.mkdir(mode=0o700)
    except OSError as error:
        raise SupervisorError("control directory could not be created exclusively") from error
    _owned_control_directory(path)

    try:
        child = subprocess.Popen(
            [sys.executable, *candidate_arguments],
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            shell=False,
            close_fds=True,
            preexec_fn=_candidate_preexec(uid, gid),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SupervisorError("candidate process could not be started") from error

    def forward(requested: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(requested)

    previous = {
        requested: signal.signal(requested, forward)
        for requested in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return_code = child.wait()
    finally:
        for requested, handler in previous.items():
            signal.signal(requested, handler)

    _write_exclusive(path / _COMPLETE, (str(return_code) + "\n").encode("ascii"))
    while True:
        release = path / _RELEASE
        try:
            identity = release.lstat()
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if not stat.S_ISREG(identity.st_mode) or identity.st_uid != os.geteuid():
            raise SupervisorError("release marker identity is invalid")
        break
    return return_code if return_code >= 0 else 128 - return_code


def _probe(path: Path) -> int:
    try:
        _owned_control_directory(path)
        identity = (path / _COMPLETE).lstat()
    except (OSError, SupervisorError):
        return NOT_READY_EXIT_CODE
    ready = stat.S_ISREG(identity.st_mode) and identity.st_uid == os.geteuid()
    return 0 if ready else NOT_READY_EXIT_CODE


def _release(path: Path) -> int:
    _owned_control_directory(path)
    complete = path / _COMPLETE
    try:
        identity = complete.lstat()
    except OSError as error:
        raise SupervisorError("candidate completion is unavailable") from error
    if not stat.S_ISREG(identity.st_mode) or identity.st_uid != os.geteuid():
        raise SupervisorError("candidate completion identity is invalid")
    _write_exclusive(path / _RELEASE, b"captured\n")
    return 0


def _self_test(uid: int, gid: int) -> int:
    script = (
        "import json,os,sys;"
        "assert os.geteuid()==int(sys.argv[1]);"
        "assert os.getegid()==int(sys.argv[2]);"
        "status=dict(line.split(':',1) for line in open('/proc/self/status') if ':' in line);"
        "assert int(status['CapEff'].strip(),16)==0;"
        "assert status['NoNewPrivs'].strip()=='1';"
        "import tacorank.execution.solution_cli;"
        "import benchmarks.kuairand_pure.pipeline;"
        "s=os.statvfs('/artifacts');"
        "p='/artifacts/preflight';"
        "open(p,'xb').write(b'ok');os.unlink(p);"
        "print(json.dumps({'capacity':s.f_frsize*s.f_blocks}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(uid), str(gid)],
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            shell=False,
            close_fds=True,
            check=False,
            preexec_fn=_candidate_preexec(uid, gid),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SupervisorError("candidate identity self-test could not start") from error
    if completed.returncode != 0:
        raise SupervisorError("candidate identity self-test failed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "self-test":
            return _self_test(arguments.uid, arguments.gid)
        control = _control_directory(arguments.control_directory)
        if arguments.command == "run":
            return _run(
                control,
                arguments.uid,
                arguments.gid,
                arguments.candidate_arguments,
            )
        if arguments.command == "probe":
            return _probe(control)
        if arguments.command == "release":
            return _release(control)
        raise SupervisorError("unknown supervisor command")
    except SupervisorError as error:
        print("tacorank container supervisor: {0}".format(error), file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
