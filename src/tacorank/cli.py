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
from .orchestrator.router import Harness
from .orchestrator.state_machine import validator
from .providers import DeepSeekResearchProvider, ProviderError
from .reporting import rebuild_views
from .research.eda import PlannerEdaToolbox
from .run_layout import RunLayout
from .schemas import EvaluationResult


def _ledger(root: Path, run_id: str) -> Path:
    return RunLayout(root, run_id).ledger


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
        "full_evaluations_completed": state.full_evaluations_completed,
        "convergence_pressure": state.consecutive_non_improving_full_evaluations,
        "stop_reason_code": state.stop_reason_code,
        "final_experiment_id": state.final_experiment_id,
    }


def _planner_for(config: RunConfig):
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
) -> tuple[Harness, EvaluationResult]:
    verified = verify_contract(config)
    artifacts = ArtifactStore(config.repository_root, config.artifact_roots)
    store = EventStore(
        _ledger(config.repository_root, config.run_id),
        artifact_store=artifacts,
        transition_validator=validator,
    )
    if live_config_path is None:
        raise ContractError("production runs require --live-config")
    if config.live_adapter_config_sha256 is None:
        raise ContractError(
            "production runs require a frozen live_adapter_config_sha256"
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
        "final_submission_provider": built.final_submission_provider,
    }
    baseline = built.baseline
    planner = _planner_for(config)
    planner.provider.preflight()
    eda_toolbox = PlannerEdaToolbox(live.input_roots["candidate_full"])
    eda_toolbox.inspect()
    harness = Harness(
        config=config,
        verified_contract=verified,
        event_store=store,
        context_builder=ContextBuilder(
            config,
            verified,
            artifacts,
            eda_toolbox=eda_toolbox,
        ),
        planner=planner,
        **adapters,
    )
    return harness, baseline


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
    setup_live.add_argument(
        "--research-campaign",
        type=Path,
        help="frozen ordered depth-campaign JSON inside the repository",
    )

    setup_trae = commands.add_parser(
        "setup-trae",
        help="prepare the production Trae coding path without benchmark data",
    )
    setup_trae.add_argument("--repository-root", type=Path, default=Path.cwd())
    setup_trae.add_argument(
        "--deployment-dir", type=Path, default=Path(".tacorank/trae")
    )
    setup_trae.add_argument("--runtime-dir", type=Path)
    setup_trae.add_argument("--python312", type=Path)
    setup_trae.add_argument("--docker", type=Path)

    trae_preflight = commands.add_parser(
        "trae-preflight",
        help="verify Docker, pinned Trae, credential, and DeepSeek model access",
    )
    trae_preflight.add_argument("--config", type=Path, required=True)
    trae_preflight.add_argument(
        "--local-only",
        action="store_true",
        help="verify the pinned Trae and Docker runtime without reading a credential",
    )

    trae_example = commands.add_parser(
        "trae-run-example",
        help="run one real Trae patch and Gate A without ML training",
    )
    trae_example.add_argument("--config", type=Path, required=True)
    trae_example.add_argument(
        "--input",
        type=Path,
        default=Path("examples/trae/experiment-spec.json"),
    )
    trae_example.add_argument("--run-id", default="trae_trial_001")
    trae_example.add_argument("--experiment-id", default="exp_0001")

    run = commands.add_parser(
        "run", help="run the frozen autonomous loop through final submission checking"
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--live-config",
        type=Path,
        required=True,
        help="required operator-reviewed production adapter configuration",
    )

    preflight = commands.add_parser(
        "preflight", help="verify every production prerequisite without creating a run"
    )
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--live-config", type=Path, required=True)

    for name in ("resume", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--live-config", type=Path, required=True)
    for name in ("status", "validate-ledger", "rebuild-views"):
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
                research_campaign_path=args.research_campaign,
            )
            print(json.dumps(result, sort_keys=True))
            return 0

        if args.command == "setup-trae":
            from .deployment import setup_trae_deployment

            root = args.repository_root.resolve(strict=True)
            python312 = args.python312 or Path(shutil.which("python3.12") or "python3.12")
            docker = args.docker or Path(shutil.which("docker") or "docker")
            runtime = args.runtime_dir or (
                Path(".tacorank-runtime") / (root.name + "-trae")
            )
            result = setup_trae_deployment(
                repository_root=root,
                deployment_directory=args.deployment_dir,
                runtime_directory=runtime,
                python312=python312,
                docker_executable=docker,
            )
            print(json.dumps(result, sort_keys=True))
            return 0

        if args.command in {"trae-preflight", "trae-run-example"}:
            from .coding.standalone import (
                TraeStandaloneConfig,
                preflight_trae,
                run_example_sync,
            )

            trae_config = TraeStandaloneConfig.load(args.config)
            if args.command == "trae-preflight":
                result = preflight_trae(
                    trae_config,
                    local_only=args.local_only,
                )
            else:
                result = run_example_sync(
                    trae_config,
                    args.input,
                    run_id=args.run_id,
                    experiment_id=args.experiment_id,
                )
            print(json.dumps(result, sort_keys=True))
            return 0

        if args.command in {"run", "preflight", "resume", "finalize"}:
            config = RunConfig.load(args.config)
            harness, baseline = _runtime(
                config,
                live_config_path=args.live_config,
            )
            if args.command == "preflight":
                eda_toolbox = harness.context_builder.eda_toolbox
                if eda_toolbox is None:
                    raise ContractError("planner EDA toolbox is unavailable")
                data_profile = eda_toolbox.inspect()
                print(
                    json.dumps(
                        {
                            "run_id": config.run_id,
                            "status": "passed",
                            "runtime": "live",
                            "research_provider": config.research_provider,
                            "planner_data_profile_sha256": data_profile.profile_sha256,
                            "baseline_primary": baseline.metric_set.primary_score,
                            "ledger_created": False,
                        },
                        sort_keys=True,
                    )
                )
                del harness
                return 0
            if args.command == "run":
                harness.bootstrap(baseline)
            else:
                events = harness.events()
                if not events:
                    raise LedgerError("no ledger exists for run_id %s" % config.run_id)
                started = events[0].payload
                if (
                    started.type != "run.started"
                    or started.config_sha256 != harness.verified_contract.config_sha256
                    or started.contract_sha256
                    != harness.verified_contract.contract_sha256
                ):
                    raise ContractError(
                        "resume configuration does not match the frozen run identity"
                    )
            if args.command == "finalize":
                asyncio.run(harness.finalize())
            else:
                asyncio.run(harness.run_to_completion())
            events = harness.events()
            rebuild_views(
                RunLayout(config.repository_root, config.run_id).run_directory,
                events,
            )
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
            rebuild_views(RunLayout(root, args.run_id).run_directory, events)
            print("rebuilt derived views for %s" % args.run_id)
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
