from __future__ import annotations

import io
import os
import secrets
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

from tacorank.execution import container_supervisor


def test_self_test_probes_candidate_runtime_dependencies(monkeypatch) -> None:
    observed = {}

    def run(argv, **kwargs):
        observed["argv"] = argv
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(container_supervisor.subprocess, "run", run)
    monkeypatch.setattr(container_supervisor, "_candidate_preexec", lambda uid, gid: None)

    assert container_supervisor._self_test(501, 20) == 0
    script = observed["argv"][2]
    assert "import certifi,numpy,pandas,pydantic,yaml" in script
    assert "import tacorank.execution.solution_cli" in script
    assert "import benchmarks.kuairand_pure.pipeline" in script


def test_self_test_command_does_not_require_control_directory(
    monkeypatch,
) -> None:
    def self_test(uid: int, gid: int) -> int:
        assert (uid, gid) == (501, 20)
        return 0

    monkeypatch.setattr(container_supervisor, "_self_test", self_test)

    assert container_supervisor.main(("self-test", "--uid", "501", "--gid", "20")) == 0


def test_export_command_does_not_require_root_control_access(monkeypatch) -> None:
    def export(allowed_outputs) -> int:
        assert allowed_outputs == ["predictions.csv"]
        return 0

    monkeypatch.setattr(container_supervisor, "_export", export)

    assert (
        container_supervisor.main(
            ("export", "--allowed-output", "predictions.csv")
        )
        == 0
    )


def test_exporter_streams_only_reviewed_regular_outputs(tmp_path: Path) -> None:
    artifact_root = (tmp_path / "artifacts").resolve()
    artifact_root.mkdir()
    (artifact_root / "tmp").mkdir()
    (artifact_root / "tmp" / "scratch").write_text("ignored", encoding="utf-8")
    prediction = artifact_root / "predictions.csv"
    prediction.write_text("row_id,score\n0,0.5\n", encoding="utf-8")
    payload = io.BytesIO()

    container_supervisor._write_archive(
        payload,
        artifact_root,
        ("predictions.csv",),
    )

    payload.seek(0)
    with tarfile.open(fileobj=payload, mode="r:") as archive:
        assert archive.getnames() == ["artifacts", "artifacts/predictions.csv"]
        extracted = archive.extractfile("artifacts/predictions.csv")
        assert extracted is not None
        assert extracted.read() == b"row_id,score\n0,0.5\n"


def test_supervisor_holds_candidate_container_until_controller_release(
    tmp_path: Path,
) -> None:
    control = Path("/tmp/tacorank-{0}-control".format(secrets.token_hex(12)))
    marker = tmp_path / "candidate-finished"
    command = (
        sys.executable,
        "-m",
        "tacorank.execution.container_supervisor",
        "run",
        "--control-directory",
        str(control),
        "--uid",
        str(os.geteuid()),
        "--gid",
        str(os.getegid()),
        "--",
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('done')",
        str(marker),
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        close_fds=True,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            container_supervisor.main(
                ("probe", "--control-directory", str(control))
            )
            != 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert marker.read_text(encoding="utf-8") == "done"
        assert process.poll() is None
        assert (
            container_supervisor.main(
                ("release", "--control-directory", str(control))
            )
            == 0
        )
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for name in ("output-captured", "candidate-complete"):
            (control / name).unlink(missing_ok=True)
        control.rmdir()
