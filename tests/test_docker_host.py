import socket
from pathlib import Path

import pytest

from tacorank.docker_host import normalize_local_docker_host


def test_windows_docker_desktop_named_pipe_is_accepted() -> None:
    endpoint = "npipe:////./pipe/docker_engine"
    assert normalize_local_docker_host(endpoint, system_name="nt") == endpoint


def test_windows_remote_or_tcp_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_local_docker_host("tcp://127.0.0.1:2375", system_name="nt")
    with pytest.raises(ValueError, match="local Windows named pipe"):
        normalize_local_docker_host("npipe:////server/pipe/docker_engine", system_name="nt")


def test_unix_docker_socket_is_canonicalized_on_posix(tmp_path: Path) -> None:
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("Unix sockets are unavailable on this platform")
    path = tmp_path / "docker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        assert normalize_local_docker_host(
            "unix://" + str(path), system_name="posix"
        ) == "unix://" + str(path)
    finally:
        listener.close()


def test_windows_docker_pipe_is_rejected_on_posix() -> None:
    with pytest.raises(ValueError, match="local Unix socket"):
        normalize_local_docker_host(
            "npipe:////./pipe/docker_engine", system_name="posix"
        )
