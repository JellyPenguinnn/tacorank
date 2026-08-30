from __future__ import annotations

import base64
import io
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest

from tacorank.execution.process import (
    OutputQuotaExceeded,
    ProcessLaunchError,
    ProcessLauncher,
    _extract_bounded_tar,
)
from tacorank.execution.sandbox import (
    IsolationGuarantees,
    LaunchSpec,
    ResourceLimits,
    RuntimeCleanupSpec,
    RuntimeOutputExtractionSpec,
)


def _archive(name: str, content: bytes, *, symbolic: bool = False) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        root = tarfile.TarInfo("artifacts")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        member = tarfile.TarInfo(name)
        if symbolic:
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            archive.addfile(member)
        else:
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return payload.getvalue()


def _spec(destination: Path, payload: bytes) -> RuntimeOutputExtractionSpec:
    encoded = base64.b64encode(payload).decode("ascii")
    return RuntimeOutputExtractionSpec(
        argv=(
            str(Path(sys.executable).resolve()),
            "-c",
            "import base64,sys;sys.stdout.buffer.write(base64.b64decode(sys.argv[1]))",
            encoded,
        ),
        destination=destination,
        allowed_relative_paths=("predictions.csv",),
        max_bytes=1024,
        timeout_seconds=5,
    )


def test_bounded_container_output_extraction_accepts_only_allowlisted_file(
    tmp_path: Path,
) -> None:
    destination = (tmp_path / "outputs").resolve()
    destination.mkdir()

    _extract_bounded_tar(
        _spec(destination, _archive("artifacts/predictions.csv", b"scores\n")),
        {},
    )

    assert (destination / "predictions.csv").read_bytes() == b"scores\n"


def test_bounded_container_output_extraction_reports_quota_exhaustion(
    tmp_path: Path,
) -> None:
    destination = (tmp_path / "outputs").resolve()
    destination.mkdir()

    with pytest.raises(OutputQuotaExceeded, match="hard byte limit"):
        _extract_bounded_tar(
            _spec(destination, _archive("artifacts/predictions.csv", b"x" * 2048)),
            {},
        )

    assert not tuple(destination.iterdir())


def test_live_runtime_output_is_extracted_before_supervisor_release(
    tmp_path: Path,
) -> None:
    python = str(Path(sys.executable).resolve())
    destination = (tmp_path / "outputs").resolve()
    destination.mkdir()
    complete = tmp_path / "candidate-complete"
    release = tmp_path / "output-captured"
    environment = {"PATH": os.environ.get("PATH", os.defpath)}
    outer = (
        "from pathlib import Path; import sys,time; "
        "complete=Path(sys.argv[1]); release=Path(sys.argv[2]); "
        "complete.write_text('0'); "
        "exec(\"while not release.exists():\\n time.sleep(0.01)\")"
    )
    extraction = _spec(
        destination,
        _archive("artifacts/predictions.csv", b"row_id,score\n0,0.5\n"),
    )
    cleanup = RuntimeCleanupSpec(
        terminate_argv=(python, "-c", "raise SystemExit(0)"),
        inspect_argv=(python, "-c", "print('')"),
        healthcheck_argv=(python, "-c", "raise SystemExit(0)"),
        cwd=tmp_path,
        environment=environment,
        output_extraction=extraction,
        completion_argv=(
            python,
            "-c",
            "from pathlib import Path; import sys; raise SystemExit(0 if Path(sys.argv[1]).is_file() else 75)",
            str(complete),
        ),
        release_argv=(
            python,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch(exist_ok=False)",
            str(release),
        ),
    )
    launch = LaunchSpec(
        argv=(python, "-c", outer, str(complete), str(release)),
        cwd=tmp_path,
        environment=environment,
        preexec_fn=None,
        start_new_session=True,
        guarantees=IsolationGuarantees(False, False, False, False, False, False, True),
        runtime_cleanup=cleanup,
    )
    managed = ProcessLauncher().launch(
        launch,
        tmp_path / "execution.log",
        ResourceLimits(5, 1024, 0),
    )
    deadline = time.monotonic() + 2
    while not managed.runtime_outputs_ready() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert managed.runtime_outputs_ready() is True
    assert managed.is_alive() is True
    assert not (destination / "predictions.csv").exists()
    managed.extract_ready_runtime_outputs()
    assert managed.finish() == 0
    assert (destination / "predictions.csv").read_bytes() == b"row_id,score\n0,0.5\n"


@pytest.mark.parametrize(
    "name,symbolic",
    [
        ("artifacts/unexpected.csv", False),
        ("artifacts/predictions.csv", True),
        ("artifacts/../escape.csv", False),
    ],
)
def test_bounded_container_output_extraction_rejects_unsafe_members(
    tmp_path: Path, name: str, symbolic: bool
) -> None:
    destination = (tmp_path / "outputs").resolve()
    destination.mkdir()

    with pytest.raises(ProcessLaunchError, match="extraction failed"):
        _extract_bounded_tar(
            _spec(destination, _archive(name, b"unsafe", symbolic=symbolic)),
            {},
        )

    assert not tuple(destination.iterdir())
