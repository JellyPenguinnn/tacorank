from __future__ import annotations

import asyncio
import json

from tacorank.cli import main
from tacorank.reporting import rebuild_views


def test_views_are_derived_and_cli_validates(harness, baseline_evaluation, capsys):
    harness.bootstrap(baseline_evaluation)
    asyncio.run(harness.run_one_experiment())
    run_directory = (
        harness.config.repository_root / "runs" / harness.config.run_id
    )
    rebuild_views(run_directory, harness.events())

    summary = (run_directory / "SUMMARY.md").read_text()
    assert "exp_001" in summary
    assert "Unmeasured tokens:" in summary
    assert "Total reported tokens:" in summary
    assert "No active lessons" in (run_directory / "LESSONS.md").read_text()
    assert '"best_experiment_id": "exp_001"' in (
        run_directory / "STATUS.md"
    ).read_text()

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


def test_run_command_executes_the_fake_vertical_slice(config, capsys):
    config.baseline_metrics = {"gauc": 0.6, "ndcg@5": 0.6, "primary": 0.6}
    config_path = config.repository_root / "run-config.json"
    config_path.write_text(
        json.dumps(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    assert main(["run", "--config", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["best_experiment_id"] == "exp_001"
    assert (config.repository_root / "runs/run_test/STATUS.md").is_file()
