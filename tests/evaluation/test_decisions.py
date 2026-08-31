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


def result(
    verdict,
    stability,
    best_delta=0.01,
    fidelity=Fidelity.FULL,
    current_primary=0.61,
    seed_mean=0.61,
    integrity=Integrity.CLEAN,
    flags=(),
):
    metric_set = MetricSet(
        {"GAUC": current_primary, "nDCG@5": current_primary},
        "primary",
        current_primary,
    )
    trust = TrustAssessment(
        verdict,
        stability,
        integrity,
        flags=flags,
        eta_applied=0.0016,
        seed_mean=seed_mean,
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

    def test_proxy_within_noise_gets_one_full_fidelity_check(self):
        decision = decide(
            result(
                Verdict.INCONCLUSIVE,
                Stability.NOT_APPLICABLE,
                fidelity=Fidelity.PROXY,
                flags=("WITHIN_NOISE",),
            ),
            CTX,
        )

        self.assertEqual(decision.decision, Decision.PROMOTE)
        self.assertEqual(decision.reason_code, "PROXY_WITHIN_NOISE")
        self.assertEqual(decision.next_fidelity, Fidelity.FULL)
        self.assertFalse(decision.parent_eligible)
        self.assertFalse(decision.best_eligible)

    def test_other_inconclusive_proxy_does_not_bypass_integrity_gate(self):
        decision = decide(
            result(
                Verdict.INCONCLUSIVE,
                Stability.NOT_APPLICABLE,
                fidelity=Fidelity.PROXY,
                integrity=Integrity.INCONCLUSIVE,
                flags=("PREDICTION_ALIGNMENT_SUSPECT",),
            ),
            CTX,
        )

        self.assertEqual(decision.decision, Decision.INVALID)
        self.assertEqual(decision.reason_code, "INTEGRITY_UNVERIFIED")
        self.assertIsNone(decision.next_fidelity)

    def test_clear_proxy_regression_is_still_pruned(self):
        decision = decide(
            result(
                Verdict.NEGATIVE,
                Stability.NOT_APPLICABLE,
                fidelity=Fidelity.PROXY,
            ),
            CTX,
        )

        self.assertEqual(decision.decision, Decision.PRUNE)
        self.assertEqual(decision.reason_code, "PROXY_FAILED")
        self.assertIsNone(decision.next_fidelity)

    def test_proxy_screening_never_depends_on_sequence_slot(self):
        decision = decide(
            result(
                Verdict.INCONCLUSIVE,
                Stability.NOT_APPLICABLE,
                fidelity=Fidelity.PROXY,
                flags=("WITHIN_NOISE",),
            ),
            replace(CTX, promote_inconclusive_proxy=False),
        )

        self.assertEqual(decision.decision, Decision.PROMOTE)
        self.assertEqual(decision.reason_code, "PROXY_WITHIN_NOISE")

    def test_single_seed_requests_confirmation(self):
        decision = decide(result(Verdict.ACCEPTED, Stability.SINGLE_SEED), CTX)
        self.assertEqual(decision.reason_code, "CONFIRMATION_REQUIRED")
        self.assertEqual(decision.next_fidelity, Fidelity.FULL)

    def test_confirmed_candidate_can_be_parent_without_becoming_best(self):
        decision = decide(result(Verdict.ACCEPTED, Stability.CONFIRMED, best_delta=0.0005), CTX)
        self.assertTrue(decision.parent_eligible)
        self.assertFalse(decision.best_eligible)
        self.assertEqual(decision.decision, Decision.ACCEPT)
        self.assertEqual(decision.reason_code, "TRUSTED_PARENT_ONLY")

    def test_confirmed_near_best_result_can_be_exploratory_parent(self):
        decision = decide(
            result(
                Verdict.INCONCLUSIVE,
                Stability.CONFIRMED,
                best_delta=-0.0002,
                current_primary=0.6014,
                seed_mean=0.6013,
                flags=("WITHIN_NOISE",),
            ),
            CTX,
        )

        self.assertEqual(decision.decision, Decision.ACCEPT)
        self.assertEqual(
            decision.reason_code,
            "EXPLORATORY_PARENT_WITHIN_TOLERANCE",
        )
        self.assertTrue(decision.parent_eligible)
        self.assertFalse(decision.best_eligible)

    def test_confirmed_result_beyond_best_tolerance_is_rejected(self):
        decision = decide(
            result(
                Verdict.INCONCLUSIVE,
                Stability.CONFIRMED,
                best_delta=-0.002,
                current_primary=0.6014,
                seed_mean=0.6013,
                flags=("WITHIN_NOISE",),
            ),
            CTX,
        )

        self.assertEqual(decision.decision, Decision.REJECT)
        self.assertEqual(decision.reason_code, "WITHIN_NOISE")
        self.assertFalse(decision.parent_eligible)

    def test_confirmed_decision_compares_aggregate_seed_mean_to_best(self):
        confirmed = result(
            Verdict.ACCEPTED,
            Stability.CONFIRMED,
            best_delta=-0.0001,
            current_primary=0.5999,
            seed_mean=0.6020,
        )
        decision = decide(confirmed, CTX)
        self.assertEqual(decision.decision, Decision.ACCEPT)
        self.assertTrue(decision.best_eligible)

    def test_no_op_requires_recovery_before_decision(self):
        with self.assertRaises(NoOpRecoveryRequired):
            decide(result(Verdict.NO_OP, Stability.NOT_APPLICABLE), CTX)

    def test_confirmed_within_noise_result_is_retained_not_rejected(self):
        decision = decide(
            result(
                Verdict.INCONCLUSIVE,
                Stability.CONFIRMED,
                flags=("WITHIN_NOISE",),
            ),
            CTX,
        )

        self.assertEqual(decision.decision, Decision.RETAIN)
        self.assertEqual(decision.reason_code, "WITHIN_NOISE")

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
