from types import SimpleNamespace

from tacorank.memory.historical import _compatible_run, _run_observations
from tacorank.schemas import (
    EventType,
    Fidelity,
    Integrity,
    Population,
    Stability,
    TrustVerdict,
)


def _evaluation(
    experiment_id,
    *,
    score,
    seed_mean=None,
    stderr=None,
    verdict=TrustVerdict.ACCEPTED,
):
    return SimpleNamespace(
        experiment_id=experiment_id,
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        metric_set=SimpleNamespace(primary_score=score),
        parent_delta=score - 0.6,
        contract_sha256="c" * 64,
        evaluator_sha256="e" * 64,
        trust=SimpleNamespace(
            seed_mean=seed_mean,
            seed_stderr=stderr,
            seed_count=3 if seed_mean is not None else 1,
            stability=Stability.CONFIRMED
            if seed_mean is not None
            else Stability.SINGLE_SEED,
            verdict=verdict,
            integrity=Integrity.CLEAN,
            flags=[],
        ),
    )


def test_historical_observations_use_seed_mean_and_risk_adjusted_reward():
    baseline = SimpleNamespace(
        experiment_id="baseline",
        metric_set=SimpleNamespace(primary_score=0.6),
        evaluation=_evaluation("baseline", score=0.6),
    )
    result = _evaluation(
        "exp_0001", score=0.603, seed_mean=0.602, stderr=0.0005
    )
    events = [
        SimpleNamespace(run_id="run_prior", event_type=EventType.BASELINE_VERIFIED, payload=baseline),
        SimpleNamespace(
            run_id="run_prior",
            event_type=EventType.EXPERIMENT_PROPOSED,
            payload=SimpleNamespace(
                spec=SimpleNamespace(
                    experiment_id="exp_0001",
                    parent_experiment_id="baseline",
                    family="objective",
                    method_card_ids=["objective_pairwise_bpr"],
                )
            ),
        ),
        SimpleNamespace(
            run_id="run_prior",
            event_type=EventType.OUTPUT_CHECKED,
            payload=SimpleNamespace(
                result=SimpleNamespace(experiment_id="exp_0001", accepted=True)
            ),
        ),
        SimpleNamespace(
            run_id="run_prior",
            event_type=EventType.EVALUATION_COMPLETED,
            seq=8,
            payload=SimpleNamespace(result=result),
        ),
    ]

    observations = _run_observations(events)

    assert len(observations) == 1
    assert observations[0].stable_primary_score == 0.602
    assert abs(observations[0].reward - 0.002) < 1e-12
    assert abs(observations[0].risk_adjusted_reward - 0.001) < 1e-12


def test_historical_compatibility_requires_contract_evaluator_and_baseline():
    baseline = SimpleNamespace(
        metric_set=SimpleNamespace(primary_score=0.6),
        evaluation=SimpleNamespace(
            contract_sha256="c" * 64,
            evaluator_sha256="e" * 64,
            population=Population.PUBLIC_VALIDATION,
        ),
    )
    events = [
        SimpleNamespace(event_type=EventType.RUN_STOPPED),
        SimpleNamespace(
            event_type=EventType.CONTRACT_VERIFIED,
            payload=SimpleNamespace(
                contract_sha256="c" * 64,
                evaluator_sha256="e" * 64,
            ),
        ),
        SimpleNamespace(
            event_type=EventType.BASELINE_VERIFIED,
            payload=baseline,
        ),
    ]

    assert _compatible_run(
        events,
        contract_sha256="c" * 64,
        evaluator_sha256="e" * 64,
        baseline_score=0.6,
    )
    assert not _compatible_run(
        events,
        contract_sha256="d" * 64,
        evaluator_sha256="e" * 64,
        baseline_score=0.6,
    )
