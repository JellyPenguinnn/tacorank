import unittest
from dataclasses import replace

from tacorank.evaluation.decisions import DecisionContext, NoOpRecoveryRequired, decide
from tacorank.evaluation.types import (
    Decision,
    EvaluationResult,
    Fidelity,
    Integrity,
    MetricDelta,
    MetricSet,
    Population,
    PredictionChange,
    Stability,
    TrustAssessment,
    Verdict,
)


def result(verdict, stability, best_delta=0.01, fidelity=Fidelity.FULL):
    metric_set = MetricSet({"GAUC": 0.67, "nDCG@5": 0.55}, "primary", 0.61)
    trust = TrustAssessment(
        verdict,
        stability,
        Integrity.CLEAN,
        eta_applied=0.0016,
        seed_mean=0.61,
        seed_stderr=0.0002,
        seed_count=3,
    )
    return EvaluationResult(
        "run_x", "exp_0001", 1,
        Population.INTERNAL_PROXY if fidelity == Fidelity.PROXY else Population.PUBLIC_VALIDATION,
        fidelity, 0, 1, "a" * 64, "b" * 64, "c" * 64,
        metric_set,
        MetricDelta(0.01, {"GAUC": 0.01, "nDCG@5": 0.01}),
        MetricDelta(0.01, {"GAUC": 0.01, "nDCG@5": 0.01}),
        MetricDelta(best_delta, {"GAUC": best_delta, "nDCG@5": best_delta}),
        PredictionChange(0.5, 1.0, 0.0, 1.0),
        trust,
    )


CTX = DecisionContext("evt_000010", ("evt_000009",), confirmations_remaining=2)


class DecisionTests(unittest.TestCase):
    def test_proxy_can_promote_but_never_be_parent_or_best(self):
        decision = decide(result(Verdict.ACCEPTED, Stability.NOT_APPLICABLE, fidelity=Fidelity.PROXY), CTX)
        self.assertEqual(decision.decision, Decision.PROMOTE)
        self.assertFalse(decision.parent_eligible)
        self.assertFalse(decision.best_eligible)

    def test_single_seed_requests_confirmation(self):
        decision = decide(result(Verdict.ACCEPTED, Stability.SINGLE_SEED), CTX)
        self.assertEqual(decision.reason_code, "CONFIRMATION_REQUIRED")
        self.assertEqual(decision.next_fidelity, Fidelity.FULL)

    def test_confirmed_candidate_can_be_parent_without_becoming_best(self):
        decision = decide(result(Verdict.ACCEPTED, Stability.CONFIRMED, best_delta=0.0005), CTX)
        self.assertTrue(decision.parent_eligible)
        self.assertFalse(decision.best_eligible)
        self.assertEqual(decision.decision, Decision.REJECT)

    def test_no_op_requires_recovery_before_decision(self):
        with self.assertRaises(NoOpRecoveryRequired):
            decide(result(Verdict.NO_OP, Stability.NOT_APPLICABLE), CTX)

    def test_unbiased_audit_never_updates_experiment_state(self):
        audit = replace(
            result(Verdict.ACCEPTED, Stability.CONFIRMED),
            population=Population.UNBIASED_AUDIT,
            public_query_index=None,
        )
        with self.assertRaisesRegex(ValueError, "unbiased-audit"):
            decide(audit, CTX)


if __name__ == "__main__":
    unittest.main()
