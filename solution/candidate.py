"""Stable TacoRank candidate entrypoint."""

from __future__ import annotations

from typing import Any

from .experiment_config import CONFIG
from .research_scaffold import run_experiment


def run(invocation: Any) -> None:
    run_experiment(invocation, CONFIG)
