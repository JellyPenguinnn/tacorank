import subprocess
import sys

from tacorank.evaluation.no_op import analyze_prediction_change
from tacorank.evaluation.trust import TrustEvidence, assess_trust
from tacorank.evaluation.types import (
    Fidelity,
    Population,
    Verdict,
)
from tacorank.evaluation.uncertainty import paired_user_delta_interval


def test_candidate_runtime_import_does_not_require_numpy() -> None:
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'numpy' or name.startswith('numpy.'):
        raise AssertionError('candidate runtime attempted to import numpy')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import benchmarks.kuairand_pure.pipeline
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        close_fds=True,
        timeout=10.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_paired_user_bootstrap_is_zero_for_identical_predictions() -> None:
    users = [str(user) for user in range(20) for _ in range(2)]
    labels = [value for _ in range(20) for value in (1, 0)]
    scores = [value for _ in range(20) for value in (0.8, 0.2)]

    interval = paired_user_delta_interval(users, labels, scores, scores, seed=7)

    assert interval.mean == 0.0
    assert interval.lower == 0.0
    assert interval.upper == 0.0


def test_paired_user_bootstrap_detects_consistent_ranking_gain() -> None:
    users = [str(user) for user in range(20) for _ in range(2)]
    labels = [value for _ in range(20) for value in (1, 0)]
    candidate = [value for _ in range(20) for value in (0.8, 0.2)]
    reference = [value for _ in range(20) for value in (0.2, 0.8)]

    interval = paired_user_delta_interval(
        users, labels, candidate, reference, seed=7
    )

    assert interval.mean > 0
    assert interval.lower > 0


def _cross_population_evidence(*, decisive: bool) -> TrustEvidence:
    change = analyze_prediction_change([0.9, 0.1], [0.1, 0.9])
    width = 0.00001 if decisive else 0.001
    return TrustEvidence(
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        parent_primary=0.6,
        parent_delta=0.0001,
        metric_deltas={"gauc": 0.0001, "ndcg@5": 0.0001},
        prediction_change=change,
        seed_scores=(0.6001, 0.6001, 0.6001),
        seed_parent_deltas=(0.0001, 0.0001, 0.0001),
        paired_parent_delta_stderr=0.001,
        paired_parent_delta_ci_lower=-0.001,
        paired_parent_delta_ci_upper=0.001,
        internal_proxy_delta=-0.0001,
        internal_proxy_ci_lower=-0.0001 - width,
        internal_proxy_ci_upper=-0.0001 + width,
        val_a_delta=0.0001,
        val_a_delta_ci_lower=0.0001 - width,
        val_a_delta_ci_upper=0.0001 + width,
    )


def test_proxy_full_sign_change_inside_intervals_is_within_noise() -> None:
    trust = assess_trust(_cross_population_evidence(decisive=False))

    assert trust.verdict == Verdict.INCONCLUSIVE
    assert "PROXY_FULL_WITHIN_UNCERTAINTY" in trust.flags
    assert "PROXY_FULL_SIGN_CONFLICT" not in trust.flags


def test_proxy_full_disjoint_opposite_intervals_are_suspicious() -> None:
    trust = assess_trust(_cross_population_evidence(decisive=True))

    assert trust.verdict == Verdict.SUSPICIOUS
    assert "PROXY_FULL_SIGN_CONFLICT" in trust.flags
