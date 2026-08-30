"""Cross-platform Docker endpoint discovery tests.

Production setup must retain the local-only boundary on every host.  macOS and
Linux Docker contexts expose a Unix socket; Docker Desktop on Windows exposes a
local named pipe instead.  These tests exercise both forms without contacting
the Docker daemon.
"""

import os
import socket
from pathlib import Path

import pytest

from tacorank.deployment import DeploymentError, _discover_docker_host
from tacorank.docker_host import normalize_local_docker_host
from tacorank.execution.sandbox import SandboxPolicyError, _validated_docker_host


def _context_output(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(
        "tacorank.deployment._run_output",
        lambda *args, **kwargs: value,
    )


@pytest.mark.skipif(os.name != "nt", reason="Docker Desktop named pipes are Windows-only")
def test_discover_accepts_docker_desktop_local_named_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del tmp_path
    endpoint = "npipe:////./pipe/docker_engine"
    _context_output(monkeypatch, endpoint)

    assert _discover_docker_host(Path("docker"), Path.cwd()) == endpoint


@pytest.mark.skipif(os.name != "nt", reason="Docker Desktop named pipes are Windows-only")
@pytest.mark.parametrize(
    "endpoint",
    (
        "npipe:////remote-host/pipe/docker_engine",
        "npipe:////./not-pipe/docker_engine",
        "npipe:////./pipe/../docker_engine",
        "tcp://127.0.0.1:2375",
        "ssh://docker.example/var/run/docker.sock",
    ),
)
def test_discover_rejects_nonlocal_or_malformed_windows_endpoint(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _context_output(monkeypatch, endpoint)

    with pytest.raises(DeploymentError):
        _discover_docker_host(Path("docker"), Path.cwd())


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable on this Python build"
)
def test_discover_accepts_canonical_unix_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = tmp_path / "docker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(socket_path))
        endpoint = "unix://" + str(socket_path.resolve(strict=True))
        _context_output(monkeypatch, endpoint)

        assert _discover_docker_host(Path("docker"), Path.cwd()) == endpoint
    finally:
        listener.close()
        if socket_path.exists():
            socket_path.unlink()


@pytest.mark.parametrize(
    "endpoint",
    (
        "tcp://127.0.0.1:2375",
        "ssh://docker.example/var/run/docker.sock",
        "unix://relative/docker.sock",
    ),
)
def test_discover_rejects_remote_or_malformed_endpoint(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _context_output(monkeypatch, endpoint)

    with pytest.raises(DeploymentError):
        _discover_docker_host(Path("docker"), Path.cwd())


@pytest.mark.skipif(os.name != "nt", reason="Docker Desktop named pipes are Windows-only")
def test_sandbox_and_trae_validate_the_same_local_named_pipe() -> None:
    endpoint = "npipe:////./pipe/docker_engine"

    assert _validated_docker_host(endpoint) == endpoint


def test_named_pipe_normalization_is_local_only_on_windows() -> None:
    endpoint = "npipe:////./pipe/docker_engine"
    assert normalize_local_docker_host(endpoint, system_name="nt") == endpoint

    for remote in (
        "npipe:////remote-host/pipe/docker_engine",
        "npipe:////./pipe/../docker_engine",
        "npipe:////./not-pipe/docker_engine",
    ):
        with pytest.raises(ValueError):
            normalize_local_docker_host(remote, system_name="nt")


@pytest.mark.skipif(os.name != "nt", reason="Docker Desktop named pipes are Windows-only")
@pytest.mark.parametrize(
    "endpoint",
    (
        "npipe:////remote-host/pipe/docker_engine",
        "npipe:////./not-pipe/docker_engine",
        "tcp://127.0.0.1:2375",
    ),
)
def test_sandbox_rejects_nonlocal_named_pipe(endpoint: str) -> None:
    with pytest.raises(SandboxPolicyError):
        _validated_docker_host(endpoint)
