from __future__ import annotations

import base64
import io
import sys
import tarfile
from pathlib import Path

import pytest

from tacorank.execution.process import ProcessLaunchError, _extract_bounded_tar
from tacorank.execution.sandbox import RuntimeOutputExtractionSpec


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
