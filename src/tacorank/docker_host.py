"""Validation and normalization for local Docker daemon endpoints."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Optional


_WINDOWS_NPIPE = re.compile(r"^npipe:////\./pipe/([A-Za-z0-9._-]+)$")


def normalize_local_docker_host(
    value: str, *, system_name: Optional[str] = None
) -> str:
    """Return a canonical local Docker endpoint or raise ``ValueError``.

    Docker Desktop uses a named pipe on native Windows and a Unix socket on
    macOS/Linux. TCP and remote endpoints are rejected by design.
    """

    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("Docker host must be a local endpoint")
    selected_system = os.name if system_name is None else system_name
    if value.startswith("npipe://"):
        if selected_system != "nt":
            raise ValueError("Docker host must be a local Unix socket")
        match = _WINDOWS_NPIPE.fullmatch(value)
        if match is None:
            raise ValueError("Docker host must be a local Windows named pipe")
        return "npipe:////./pipe/" + match.group(1)

    if not value.startswith("unix://"):
        raise ValueError("Docker host must be a local Unix socket")
    socket_path = Path(value[len("unix://") :])
    if not socket_path.is_absolute() or socket_path.is_symlink():
        raise ValueError("Docker host must be a canonical local Unix socket")
    try:
        resolved = socket_path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise ValueError("Docker host socket is unavailable") from error
    if resolved != socket_path or not stat.S_ISSOCK(metadata.st_mode):
        raise ValueError("Docker host must be a canonical local Unix socket")
    return "unix://" + str(resolved)
