from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from tacorank.cli import build_parser, main
from tacorank.config import RunConfig
from tacorank.reporting import rebuild_views


def test_views_are_derived_and_cli_validates(harness, baseline_evaluation, capsys):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    run_directory = (
        harness.config.repository_root / "runs" / harness.config.run_id
    )
    rebuild_views(run_directory, harness.events())

    summary = (run_directory / "reports/SUMMARY.md").read_text()
    assert "exp_001" in summary
    assert "Unmeasured tokens:" in summary
    assert "Total reported tokens:" in summary
    lesson_index = (run_directory / "lessons/INDEX.md").read_text()
    assert "lesson_001" in lesson_index
    lesson = (run_directory / "lessons/lesson_001.md").read_text()
    assert "confirmed clean result" in lesson
    assert '"best_experiment_id": "exp_001"' in (
        run_directory / "STATUS.md"
    ).read_text()
    state = json.loads((run_directory / "state.json").read_text())
    assert state["derived_from"]["event_id"] == harness.events()[-1].event_id
    assert state["execution_mode"] == "sequential"
    assert state["active_jobs"] == []
    graph = json.loads((run_directory / "experiment-graph/graph.json").read_text())
    assert [node["experiment_id"] for node in graph["nodes"]] == [
        "baseline",
        "exp_001",
    ]
    experiment = graph["nodes"][1]
    assert experiment["diagnostics"]["validation_arm_gap"] == 0.01
    experiment_report = (
        run_directory
        / "experiment-graph/directions/feature-cross/experiments/exp_001.md"
    ).read_text()
    assert "## Diagnostics" in experiment_report
    assert "user_history.cold" in experiment_report
    assert "## Diagnostic findings" in summary
    assert experiment["diagnostic_metrics"]["user_rankable_fraction"] == 1.0
    assert experiment["adapter_failures"] == []
    assert experiment["recovery_decisions"] == []
    assert (
        run_directory
        / "experiment-graph/directions/feature-cross/experiments/exp_001.md"
    ).is_file()
    assert (run_directory / "reports/RESOURCES.md").is_file()
    assert (run_directory / "artifacts/exp_001/attempt_001/patch.diff").is_file()

    assert (
        main(
            [
                "validate-ledger",
                "--run-id",
                harness.config.run_id,
                "--repository-root",
                str(harness.config.repository_root),
            ]
        )
        == 0
    )
    assert "valid:" in capsys.readouterr().out


def test_run_command_requires_live_configuration():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--config", "run-config.json"])


def test_fake_runtime_configuration_is_rejected(config):
    payload = config.model_dump(mode="python")
    payload["adapter_mode"] = "fake"
    payload["research_provider"] = "fake"
    with pytest.raises(ValidationError, match="adapter_mode|research_provider"):
        RunConfig.model_validate(payload)

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "run",
                "--config",
                "run-config.json",
                "--live-config",
                "live-adapters.json",
                "--allow-test-adapters",
            ]
        )
