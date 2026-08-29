from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tacorank.artifacts import ArtifactStore
from tacorank.config import RunConfig, verify_contract
from tacorank.context.builder import ContextBuilder
from tacorank.memory.event_store import EventStore
from tacorank.orchestrator.fakes import (
    FakeCodingWorker,
    FakeEvaluator,
    FakeExecutionRunner,
    FakeHealthObserver,
    FakeOutputGate,
    FakePatchGate,
    FakeRecoveryManager,
    FakeResearchPlanner,
)
from tacorank.orchestrator.router import Harness
from tacorank.orchestrator.state_machine import validator
from tacorank.schemas import (
    EvaluationResult,
    Fidelity,
    Integrity,
    MetricSet,
    Population,
    Stability,
    TrustAssessment,
    TrustVerdict,
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    (tmp_path / "contract").mkdir()
    (tmp_path / "contract/COMPETITION.md").write_text(
        "# Competition contract\n\n"
        "Contract status: FROZEN\n\n"
        "Metrics: gauc, ndcg@5, primary. Primary is the mean.\n"
        "Allowed command IDs: run_smoke, run_proxy, run_full\n"
        "Artifact roots: artifacts, runs\n"
        "Development uses public validation; hidden final is exposed once after stop.\n",
        encoding="utf-8",
    )
    (tmp_path / "PROTECTED_PATHS.md").write_text(
        "# Protected\n\ncontract/\ntests/evaluation/\n", encoding="utf-8"
    )
    (tmp_path / "research/methods").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "runs").mkdir()
    return tmp_path


@pytest.fixture
def config(repository: Path) -> RunConfig:
    return RunConfig(
        run_id="run_test",
        repository_root=repository,
        command_ids=["run_smoke", "run_proxy", "run_full"],
        metric_names=["gauc", "ndcg@5", "primary"],
        primary_metric_name="primary",
        data_manifest_sha256=sha(b"data"),
        evaluator_sha256=sha(b"evaluator"),
        baseline_commit_sha="b" * 40,
        research_provider="fake",
        max_experiments=3,
        seed_schedule=[11, 22, 33, 44],
        context_token_limit=2000,
    )


@pytest.fixture
def baseline_evaluation(config: RunConfig) -> EvaluationResult:
    metrics = {"gauc": 0.6, "ndcg@5": 0.6, "primary": 0.6}
    return EvaluationResult(
        run_id=config.run_id,
        experiment_id="baseline",
        attempt=1,
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        seed=config.seed_schedule[0],
        public_query_index=1,
        evaluator_sha256=config.evaluator_sha256,
        contract_sha256=sha((config.repository_root / config.contract_path).read_bytes()),
        metric_set=MetricSet(
            metrics=metrics, primary_metric_name="primary", primary_score=0.6
        ),
        baseline_delta=0.0,
        parent_delta=0.0,
        previous_best_delta=0.0,
        prediction_change=1.0,
        trust=TrustAssessment(
            verdict=TrustVerdict.ACCEPTED,
            stability=Stability.CONFIRMED,
            integrity=Integrity.CLEAN,
        ),
    )


@pytest.fixture
def harness(config: RunConfig) -> Harness:
    verified = verify_contract(config)
    artifacts = ArtifactStore(config.repository_root, config.artifact_roots)
    store = EventStore(
        config.repository_root / "runs" / config.run_id / "events.jsonl",
        artifact_store=artifacts,
        transition_validator=validator,
    )
    return Harness(
        config=config,
        verified_contract=verified,
        event_store=store,
        context_builder=ContextBuilder(config, verified, artifacts),
        planner=FakeResearchPlanner(config.baseline_commit_sha),
        coding_worker=FakeCodingWorker(artifacts),
        patch_gate=FakePatchGate(artifacts),
        runner=FakeExecutionRunner(artifacts),
        health_observer=FakeHealthObserver(),
        recovery_manager=FakeRecoveryManager(),
        output_gate=FakeOutputGate(),
        evaluator=FakeEvaluator(config.metric_names, config.primary_metric_name),
    )
