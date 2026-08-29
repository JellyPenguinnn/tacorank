from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from tacorank.execution.container_supervisor import main


def test_supervisor_holds_candidate_container_until_controller_release(
    tmp_path: Path,
) -> None:
    control = Path("/tmp/tacorank-{0}-control".format(secrets.token_hex(12)))
    marker = tmp_path / "candidate-finished"
    command = (
        str(Path(sys.executable).resolve()),
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
            main(("probe", "--control-directory", str(control))) != 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert marker.read_text(encoding="utf-8") == "done"
        assert process.poll() is None
        assert main(("release", "--control-directory", str(control))) == 0
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for name in ("output-captured", "candidate-complete"):
            (control / name).unlink(missing_ok=True)
        control.rmdir()
