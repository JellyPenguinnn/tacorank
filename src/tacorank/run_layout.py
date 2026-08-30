"""Canonical filesystem layout for one TacoRank run.

The event ledger and immutable artifacts are durable evidence.  State, graph,
lesson, and report files are materialized views that can be rebuilt from that
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from .schemas import ID_RE


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise ValueError("%s is not a valid TacoRank identifier" % field)
    return value


def run_relative_directory(run_id: str) -> PurePosixPath:
    return PurePosixPath("runs") / _identifier(run_id, "run_id")


def run_artifact_root(run_id: str) -> str:
    """Return the repository-relative artifact root for a new run."""

    return (run_relative_directory(run_id) / "artifacts").as_posix()


def experiment_artifact_prefix(
    run_id: str,
    experiment_id: str,
    *,
    attempt: Optional[int] = None,
) -> str:
    """Return a stable repository-relative experiment artifact prefix."""

    path = run_relative_directory(run_id) / "artifacts"
    path /= _identifier(experiment_id, "experiment_id")
    if attempt is not None:
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        path /= "attempt_%03d" % attempt
    return path.as_posix()


@dataclass(frozen=True)
class RunLayout:
    """Resolved paths for one run without creating or mutating them."""

    repository_root: Path
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_root", Path(self.repository_root).resolve()
        )
        _identifier(self.run_id, "run_id")

    @property
    def run_directory(self) -> Path:
        return self.repository_root / run_relative_directory(self.run_id)

    @property
    def ledger(self) -> Path:
        return self.run_directory / "events.jsonl"

    @property
    def state(self) -> Path:
        return self.run_directory / "state.json"

    @property
    def status(self) -> Path:
        return self.run_directory / "STATUS.md"

    @property
    def contexts(self) -> Path:
        return self.run_directory / "contexts"

    @property
    def lessons(self) -> Path:
        return self.run_directory / "lessons"

    @property
    def experiment_graph(self) -> Path:
        return self.run_directory / "experiment-graph"

    @property
    def artifacts(self) -> Path:
        return self.run_directory / "artifacts"

    @property
    def reports(self) -> Path:
        return self.run_directory / "reports"
