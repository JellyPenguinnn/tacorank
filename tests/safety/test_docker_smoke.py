from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tacorank.safety.docker_smoke import DockerEntrypointSmokeCheck


_FAKE_DOCKER = r'''import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / "calls.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\n")
behavior = (root / "behavior").read_text(encoding="utf-8").strip()
if sys.argv[1] == "run":
    cidfile = Path(sys.argv[sys.argv.index("--cidfile") + 1])
    cidfile.write_text("d" * 64 + "\n", encoding="ascii")
    if behavior == "fail":
        print("candidate import failed")
        raise SystemExit(7)
elif sys.argv[1] == "rm" and behavior == "cleanup_fail":
    raise SystemExit(2)
elif sys.argv[1] == "inspect":
    raise SystemExit(0 if behavior == "cleanup_fail" else 1)
raise SystemExit(0)
'''


def _smoke(tmp_path: Path, monkeypatch):
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n" + _FAKE_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    (tmp_path / "behavior").write_text("ok", encoding="utf-8")
    monkeypatch.setattr(
        "tacorank.safety.docker_smoke.normalize_local_docker_host",
        lambda value: value,
    )
    checker = DockerEntrypointSmokeCheck(
        docker_executable=docker.resolve(),
        docker_host="unix:///tmp/fake-docker.sock",
        image="tacorank@sha256:" + "a" * 64,
        container_python_executable="/usr/local/bin/python3",
        entrypoint="solution.candidate:run",
    )
    return checker


def test_docker_smoke_uses_read_only_no_network_import_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    checker = _smoke(tmp_path, monkeypatch)
    repository = tmp_path / "worktree"
    repository.mkdir()
    passed, summary = checker.run(
        repository,
        SimpleNamespace(
            run_id="run_1",
            experiment_id="exp_1",
            patch_commit_sha="a" * 40,
        ),
    )

    assert passed
    assert "synthetic execution succeeded" in summary
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    run = calls[0]
    assert run[0] == "run"
    assert "--cidfile" in run
    assert run[run.index("--network") + 1] == "none"
    assert "--read-only" in run
    assert run[run.index("--cap-drop") + 1] == "ALL"
    assert "readonly" in run[run.index("--mount") + 1]
    assert run[run.index("--entrypoint") + 1] == "/usr/local/bin/python3"
    assert run[-1] == "solution.candidate:run"
    assert calls[-2][:2] == ["rm", "--force"]
    assert calls[-1][:3] == ["inspect", "--type", "container"]


def test_docker_smoke_reports_candidate_import_failure(
    tmp_path: Path, monkeypatch
) -> None:
    checker = _smoke(tmp_path, monkeypatch)
    (tmp_path / "behavior").write_text("fail", encoding="utf-8")
    repository = tmp_path / "worktree"
    repository.mkdir()
    passed, summary = checker.run(
        repository,
        SimpleNamespace(
            run_id="run_1",
            experiment_id="exp_1",
            patch_commit_sha="b" * 40,
        ),
    )

    assert not passed
    assert "candidate import failed" in summary


def test_docker_smoke_fails_closed_when_container_cleanup_is_unproven(
    tmp_path: Path, monkeypatch
) -> None:
    checker = _smoke(tmp_path, monkeypatch)
    (tmp_path / "behavior").write_text("cleanup_fail", encoding="utf-8")
    repository = tmp_path / "worktree"
    repository.mkdir()

    with pytest.raises(RuntimeError, match="prove Docker smoke container removal"):
        checker.run(
            repository,
            SimpleNamespace(
                run_id="run_1",
                experiment_id="exp_1",
                patch_commit_sha="c" * 40,
            ),
        )
