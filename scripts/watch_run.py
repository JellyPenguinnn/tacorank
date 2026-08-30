#!/usr/bin/env python3
"""Live score table for a TacoRank run.

Reads the run's append-only ledger and reprints a table of every experiment,
its evaluated scores, and its distance from the verified baseline. Intended to
be left running in a second terminal while a run proceeds.

    python scripts/watch_run.py                 # newest run, refresh every 10s
    python scripts/watch_run.py --run-id run_x  # a specific run
    python scripts/watch_run.py --once          # print once and exit

The ledger is append-only and read without locking, so a partially written
trailing line is expected and skipped rather than treated as corruption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reference points from kuairand-starter-kit/baseline_scores.json. The oracle
# uses true labels as scores, so it is the denominator for progress, never a
# reachable target: ~30% of scoring users are all-negative and score nDCG 0
# under every possible model.
RANDOM_PRIMARY = 0.4815
ORACLE_PRIMARY = 0.8484

TERMINAL = {"accept", "reject", "prune", "invalid"}


def find_run(root: Path, run_id: Optional[str]) -> Path:
    runs = root / "runs"
    if run_id:
        path = runs / run_id
        if not (path / "events.jsonl").exists():
            raise SystemExit("no ledger at %s" % (path / "events.jsonl"))
        return path
    candidates = [p for p in runs.iterdir() if (p / "events.jsonl").exists()]
    if not candidates:
        raise SystemExit("no runs with a ledger under %s" % runs)
    return max(candidates, key=lambda p: (p / "events.jsonl").stat().st_mtime)


def read_events(path: Path) -> List[dict]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Trailing partial write from the concurrent appender.
                break
    return events


def collect(events: List[dict]) -> Dict[str, Any]:
    baseline: Optional[float] = None
    experiments: Dict[str, Dict[str, Any]] = {}
    stopped: Optional[str] = None

    for event in events:
        kind = event["event_type"]
        payload = event["payload"]

        if kind == "baseline.verified":
            metrics = payload.get("result", payload).get("metric_set", {})
            baseline = metrics.get("primary_score", baseline)

        elif kind == "experiment.proposed":
            spec = payload["spec"]
            experiments.setdefault(spec["experiment_id"], {}).update(
                parent=spec.get("parent_experiment_id"),
                family=spec.get("family"),
                methods=",".join(spec.get("method_card_ids") or []),
                status="proposed",
            )

        elif kind == "evaluation.completed":
            result = payload["result"]
            row = experiments.setdefault(result["experiment_id"], {})
            metrics = result["metric_set"]
            # Prefer a public-validation full result over an internal proxy.
            rank = (result["fidelity"] == "full", result["population"] == "public_validation")
            if rank >= row.get("_rank", (False, False)):
                row.update(
                    _rank=rank,
                    fidelity=result["fidelity"],
                    population=result["population"],
                    primary=metrics["primary_score"],
                    gauc=metrics["metrics"].get("GAUC"),
                    ndcg=metrics["metrics"].get("nDCG@5"),
                    verdict=result["trust"]["verdict"],
                    flags=result["trust"].get("flags") or [],
                )

        elif kind == "experiment.decided":
            decision = payload["decision"]
            row = experiments.setdefault(decision["experiment_id"], {})
            # A promotion is a waypoint; keep the last terminal decision.
            if decision["decision"] in TERMINAL or "status" not in row:
                row["status"] = decision["decision"]
                row["reason"] = decision.get("reason_code", "")

        elif kind == "adapter.failed":
            result = payload.get("result", {})
            row = experiments.setdefault(result.get("experiment_id") or "?", {})
            row["failed"] = result.get("error_class", "failure")

        elif kind == "run.stopped":
            stopped = payload.get("reason_code")

    return {"baseline": baseline, "experiments": experiments, "stopped": stopped}


def load_state(run_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def progress(primary: Optional[float]) -> str:
    if primary is None:
        return "     -"
    span = ORACLE_PRIMARY - RANDOM_PRIMARY
    return "%5.1f%%" % (100.0 * (primary - RANDOM_PRIMARY) / span)


def render(run_dir: Path) -> str:
    data = collect(read_events(run_dir / "events.jsonl"))
    state = load_state(run_dir)
    glob = state.get("global", {})
    baseline = data["baseline"] or glob.get("baseline_primary_score")

    lines: List[str] = []
    lines.append("TacoRank  %s" % run_dir.name)
    lines.append(
        "phase=%s  proposed=%s  full_evals=%s  remaining=%s"
        % (
            glob.get("phase", "?"),
            glob.get("experiments_proposed", "?"),
            glob.get("full_evaluations_completed", "?"),
            glob.get("remaining_iterations", "?"),
        )
    )
    if baseline is not None:
        lines.append(
            "baseline %.6f (%s of random->oracle)   best %s %s"
            % (
                baseline,
                progress(baseline).strip(),
                glob.get("best_experiment_id", "?"),
                ("%.6f" % glob["best_primary_score"])
                if glob.get("best_primary_score") is not None
                else "",
            )
        )
    if data["stopped"]:
        lines.append("STOPPED: %s" % data["stopped"])
    lines.append("")

    header = "%-9s %-16s %-10s %-6s %-11s %-11s %-9s %-9s %s" % (
        "exp", "family", "status", "fid", "primary", "vs base", "GAUC", "nDCG@5", "verdict"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name in sorted(data["experiments"]):
        row = data["experiments"][name]
        primary = row.get("primary")
        delta = None if (primary is None or baseline is None) else primary - baseline
        if delta is None:
            shown_delta = "-"
        else:
            # Mark anything ahead of the verified baseline, which is the only
            # comparison that decides the final submission.
            shown_delta = "%+.6f%s" % (delta, "*" if delta > 0 else "")
        lines.append(
            "%-9s %-16s %-10s %-6s %-11s %-11s %-9s %-9s %s"
            % (
                name,
                (row.get("family") or "")[:16],
                (row.get("status") or "")[:10],
                (row.get("fidelity") or "")[:6],
                "%.6f" % primary if primary is not None else "-",
                shown_delta,
                "%.5f" % row["gauc"] if row.get("gauc") is not None else "-",
                "%.5f" % row["ndcg"] if row.get("ndcg") is not None else "-",
                row.get("failed") or row.get("verdict") or "",
            )
        )
    if not data["experiments"]:
        lines.append("(no experiments yet)")

    lines.append("")
    lines.append("* = above the verified baseline. Oracle 0.8484 uses true labels")
    lines.append("  as scores; it is the progress denominator, not a target.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="defaults to the most recently updated run")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = find_run(args.repository_root, args.run_id)
    while True:
        try:
            frame = render(run_dir)
        except Exception as error:  # keep watching through a transient read
            frame = "%s\n\nread error: %s" % (run_dir.name, error)
        if args.once:
            print(frame)
            return 0
        os.system("cls" if os.name == "nt" else "clear")
        print(frame)
        print("\nrefreshing every %gs - Ctrl+C to stop" % args.interval)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
