from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from tacorank.cli import build_parser, main
from tacorank.config import RunConfig
from tacorank.reporting import experiment_timing, rebuild_views, runtime_status


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
    assert "| Loop time | Trae coding | Execution | Recovery |" in summary
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
    assert state["current"]["experiment_id"] == "exp_001"
    assert state["current"]["phase"] == "planning"
    assert state["current"]["last_event_id"] == harness.events()[-1].event_id
    timing = state["experiment_timings"]["exp_001"]
    assert timing["loop_time_seconds"] is not None
    assert timing["loop_time_seconds"] >= 0
    assert timing["trae_coding_time_seconds"] >= 0
    assert timing["execution_time_seconds"] >= 0
    assert timing["recovery_time_seconds"] == 0
    graph = json.loads((run_directory / "experiment-graph/graph.json").read_text())
    assert [node["experiment_id"] for node in graph["nodes"]] == [
        "baseline",
        "exp_001",
    ]
    experiment = graph["nodes"][1]
    assert experiment["timing"] == timing
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
                "status",
                "--run-id",
                harness.config.run_id,
                "--repository-root",
                str(harness.config.repository_root),
            ]
        )
        == 0
    )
    live_status = json.loads(capsys.readouterr().out)
    assert live_status["current_experiment_id"] == "exp_001"
    assert live_status["stage_started_at"] is not None
    assert live_status["stage_elapsed_seconds"] >= 0
    assert live_status["last_event_at"] is not None
    assert live_status["last_event_age_seconds"] >= 0
    assert live_status["status_observed_at"] is not None

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


def test_runtime_status_exposes_live_stage_anchors_and_timeout(
    harness, baseline_evaluation
):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    events = list(harness.events())
    coder_context_index = next(
        index
        for index, event in enumerate(events)
        if event.payload.type == "context.created"
        and event.payload.context.role == "coder"
    )
    active_events = events[: coder_context_index + 1]

    current = runtime_status(active_events)
    assert current["experiment_id"] == "exp_001"
    assert current["phase"] == "coder_context"
    assert current["attempt"] == 1
    assert current["fidelity"] is None
    assert current["stage_started_at"] == current["last_event_at"]
    assert current["stage_elapsed_seconds_at_ledger_head"] == 0
    assert current["configured_timeout_seconds"] == 1800
    assert current["estimated_deadline"] is not None
    assert current["last_event_type"] == "context.created"

    completed = experiment_timing(events, "exp_001")
    assert completed["proposed_at"] is not None
    assert completed["terminal_at"] is not None
    assert completed["terminal_event_id"] is not None
    assert completed["loop_time_seconds"] >= 0


def test_run_command_requires_live_configuration():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--config", "run-config.json"])


def test_new_live_deployments_default_to_discovery_mode():
    args = build_parser().parse_args(["setup-live"])

    assert args.run_mode == "discovery"


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
