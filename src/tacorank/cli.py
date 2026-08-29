"""Command-line interface for production runs and ledger operations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

from .artifacts import ArtifactStore
from .agents import ResearchPlanner
from .config import ContractError, RunConfig, verify_contract
from .context.builder import ContextBuilder
from .memory.event_store import EventStore, LedgerError
from .memory.projections import project
from .memory.replay import replay
from .orchestrator.fakes import (
    FakeCodingWorker,
    FakeEvaluator,
    FakeExecutionRunner,
    FakeHealthObserver,
    FakeOutputGate,
    FakePatchGate,
    FakeRecoveryManager,
    FakeResearchPlanner,
)
from .orchestrator.router import Harness
from .orchestrator.state_machine import validator
from .providers import DeepSeekResearchProvider, ProviderError
from .reporting import rebuild_views
from .schemas import (
    EvaluationResult,
    Fidelity,
    Integrity,
    MetricSet,
    Population,
    Stability,
    TrustAssessment,
    TrustVerdict,
)


def _ledger(root: Path, run_id: str) -> Path:
    return root / "runs" / run_id / "events.jsonl"


def _store(root: Path, run_id: str) -> EventStore:
    artifacts = ArtifactStore(root)
    return EventStore(
        _ledger(root, run_id), artifact_store=artifacts, transition_validator=validator
    )


def _status_dict(state) -> dict:
    return {
        "run_id": state.run_id,
        "status": state.status.value,
        "phase": state.phase,
        "last_event_id": state.last_event_id,
        "best_experiment_id": state.best_experiment_id,
        "best_primary_score": state.best_primary_score,
        "experiments_proposed": state.experiments_proposed,
        "remaining_experiments": state.remaining_experiments,
    }


def _planner_for(config: RunConfig):
    if config.research_provider == "fake":
        if config.adapter_mode != "fake":
            raise ContractError("live adapter mode cannot use the fake research provider")
        return FakeResearchPlanner(config.baseline_commit_sha)
    api_key = os.environ.get(config.deepseek_api_key_env, "").strip()
    if not api_key:
        raise ProviderError(
            "DeepSeek research provider requires environment variable %s"
            % config.deepseek_api_key_env
        )
    provider = DeepSeekResearchProvider(
        api_key=api_key,
        model=config.deepseek_model,
        base_url=config.deepseek_base_url,
        timeout_seconds=config.deepseek_timeout_seconds,
        max_output_tokens=config.deepseek_max_output_tokens,
        thinking_enabled=config.deepseek_thinking_enabled,
        reasoning_effort=config.deepseek_reasoning_effort,
    )
    return ResearchPlanner(
        provider,
        input_token_limit=config.context_token_limit,
        output_token_limit=config.deepseek_max_output_tokens,
    )


def _runtime(
    config: RunConfig,
    *,
    live_config_path: Optional[Path],
    allow_test_adapters: bool,
) -> tuple[Harness, EvaluationResult]:
    verified = verify_contract(config)
    artifacts = ArtifactStore(config.repository_root, config.artifact_roots)
    store = EventStore(
        _ledger(config.repository_root, config.run_id),
        artifact_store=artifacts,
        transition_validator=validator,
    )
    if config.adapter_mode == "fake":
        if not allow_test_adapters:
            raise ContractError(
                "fake adapters are test-only; pass --allow-test-adapters explicitly"
            )
        adapters = {
            "coding_worker": FakeCodingWorker(artifacts),
            "patch_gate": FakePatchGate(artifacts),
            "runner": FakeExecutionRunner(artifacts),
            "health_observer": FakeHealthObserver(),
            "recovery_manager": FakeRecoveryManager(),
            "output_gate": FakeOutputGate(),
            "evaluator": FakeEvaluator(
                config.metric_names, config.primary_metric_name
            ),
        }
        baseline = _fake_baseline(config, verified.contract_sha256)
    else:
        if live_config_path is None:
            raise ContractError("live adapter mode requires --live-config")
        if config.live_adapter_config_sha256 is None:
            raise ContractError(
                "live adapter mode requires a frozen live_adapter_config_sha256"
            )
        actual_live_hash = hashlib.sha256(live_config_path.read_bytes()).hexdigest()
        if actual_live_hash != config.live_adapter_config_sha256:
            raise ContractError("live adapter configuration hash does not match")
        from .orchestrator.live import LiveAdapterConfig, build_live_adapters

        live = LiveAdapterConfig.load(live_config_path)
        built = build_live_adapters(
            config=config,
            verified=verified,
            live=live,
            event_store=store,
            artifact_store=artifacts,
        )
        adapters = {
            "coding_worker": built.coding_worker,
            "patch_gate": built.patch_gate,
            "runner": built.runner,
            "health_observer": built.health_observer,
            "recovery_manager": built.recovery_manager,
            "output_gate": built.output_gate,
            "evaluator": built.evaluator,
        }
        baseline = built.baseline
    planner = _planner_for(config)
    if config.adapter_mode == "live":
        planner.provider.preflight()
    harness = Harness(
        config=config,
        verified_contract=verified,
        event_store=store,
        context_builder=ContextBuilder(config, verified, artifacts),
        planner=planner,
        **adapters,
    )
    return harness, baseline


def _fake_baseline(config: RunConfig, contract_sha256: str) -> EvaluationResult:
    if config.baseline_metrics is None:
        raise ContractError("fake adapter mode requires frozen baseline_metrics in config")
    metric_set = MetricSet(
        metrics=config.baseline_metrics,
        primary_metric_name=config.primary_metric_name,
        primary_score=config.baseline_metrics[config.primary_metric_name],
    )
    return EvaluationResult(
        run_id=config.run_id,
        experiment_id="baseline",
        attempt=1,
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        seed=config.seed_schedule[0],
        public_query_index=1,
        evaluator_sha256=config.evaluator_sha256,
        contract_sha256=contract_sha256,
        metric_set=metric_set,
        baseline_delta=0,
        parent_delta=0,
        previous_best_delta=0,
        prediction_change=1,
        trust=TrustAssessment(
            verdict=TrustVerdict.ACCEPTED,
            stability=Stability.CONFIRMED,
            integrity=Integrity.CLEAN,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tacorank")
    commands = parser.add_subparsers(dest="command", required=True)

    setup_live = commands.add_parser(
        "setup-live",
        help="prepare a hash-bound production deployment from a clean clone",
    )
    setup_live.add_argument("--repository-root", type=Path, default=Path.cwd())
    setup_live.add_argument(
        "--deployment-dir", type=Path, default=Path(".tacorank/deployment")
    )
    setup_live.add_argument("--runtime-dir", type=Path)
    setup_live.add_argument(
        "--data-dir", type=Path, default=Path("KuaiRand-Pure/data")
    )
    setup_live.add_argument("--python312", type=Path)
    setup_live.add_argument("--docker", type=Path)
    setup_live.add_argument("--run-id", default="run_001")
    setup_live.add_argument("--download-data", action="store_true")

    run = commands.add_parser("run", help="start a frozen run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--live-config",
        type=Path,
        help="operator-reviewed production adapter configuration",
    )
    run.add_argument(
        "--allow-test-adapters",
        action="store_true",
        help="explicitly permit deterministic fake adapters for tests only",
    )

    preflight = commands.add_parser(
        "preflight", help="verify every production prerequisite without creating a run"
    )
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--live-config", type=Path, required=True)

    for name in ("resume", "status", "validate-ledger", "rebuild-views", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "setup-live":
            from .deployment import setup_live_deployment

            root = args.repository_root.resolve(strict=True)
            python312 = args.python312 or Path(shutil.which("python3.12") or "python3.12")
            docker = args.docker or Path(shutil.which("docker") or "docker")
            runtime = args.runtime_dir or (
                Path(".tacorank-runtime") / root.name
            )
            result = setup_live_deployment(
                repository_root=root,
                deployment_directory=args.deployment_dir,
                runtime_directory=runtime,
                data_directory=args.data_dir,
                python312=python312,
                docker_executable=docker,
                run_id=args.run_id,
                download_data=args.download_data,
            )
            print(json.dumps(result, sort_keys=True))
            return 0

        if args.command in {"run", "preflight"}:
            config = RunConfig.load(args.config)
            if args.command == "preflight" and config.adapter_mode != "live":
                raise ContractError("preflight accepts production live configuration only")
            harness, baseline = _runtime(
                config,
                live_config_path=args.live_config,
                allow_test_adapters=(
                    args.allow_test_adapters if args.command == "run" else False
                ),
            )
            if args.command == "preflight":
                print(
                    json.dumps(
                        {
                            "run_id": config.run_id,
                            "status": "passed",
                            "adapter_mode": "live",
                            "research_provider": config.research_provider,
                            "baseline_primary": baseline.metric_set.primary_score,
                            "ledger_created": False,
                        },
                        sort_keys=True,
                    )
                )
                del harness
                return 0
            harness.bootstrap(baseline)
            asyncio.run(harness.run_one_experiment())
            events = harness.events()
            rebuild_views(config.repository_root / "runs" / config.run_id, events)
            print(json.dumps(_status_dict(project(events)), sort_keys=True))
            return 0

        root = args.repository_root.resolve()
        store = _store(root, args.run_id)
        events = store.read_events(repair_tail=args.command == "resume")
        if not events and args.command != "status":
            raise LedgerError("no ledger exists for run_id %s" % args.run_id)
        state = project(events)
        if args.command == "validate-ledger":
            replay(events, artifact_store=store.artifact_store)
            print("valid: %d events, head=%s" % (len(events), state.last_event_hash))
        elif args.command == "status":
            print(json.dumps(_status_dict(state), sort_keys=True))
        elif args.command == "rebuild-views":
            rebuild_views(root / "runs" / args.run_id, events)
            print("rebuilt derived views for %s" % args.run_id)
        elif args.command == "resume":
            print(json.dumps({**_status_dict(state), "resume_from": state.phase}, sort_keys=True))
        elif args.command == "finalize":
            if state.status.value != "stopped":
                raise LedgerError("finalize requires run.stopped and clean reproduction evidence")
            raise LedgerError(
                "final selection requires the real runner/evaluator reproduction adapter; "
                "P0 intentionally refuses to fabricate it"
            )
        return 0
    except (
        ContractError,
        LedgerError,
        ProviderError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
