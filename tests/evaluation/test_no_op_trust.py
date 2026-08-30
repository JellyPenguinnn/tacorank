import unittest

from tacorank.evaluation.no_op import NoOpConfig, analyze_prediction_change, is_no_op
from tacorank.evaluation.trust import TrustConfig, TrustEvidence, assess_trust
from tacorank.evaluation.types import (
    Fidelity,
    Integrity,
    Population,
    Stability,
    Verdict,
)


def evidence(change, parent_delta=0.0, **overrides):
    values = dict(
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        parent_primary=0.60,
        parent_delta=parent_delta,
        metric_deltas={"GAUC": parent_delta, "nDCG@5": parent_delta},
        prediction_change=change,
        seed_scores=(0.61, 0.6105, 0.6098),
    )
    values.update(overrides)
    return TrustEvidence(**values)


class NoOpTrustTests(unittest.TestCase):
    def test_identical_predictions_are_no_op(self):
        change = analyze_prediction_change([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        self.assertTrue(is_no_op(change, 0.0, NoOpConfig()))
        self.assertEqual(assess_trust(evidence(change)).verdict, Verdict.NO_OP)

    def test_rank_changes_prevent_false_no_op(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        self.assertFalse(is_no_op(change, 0.0, NoOpConfig()))

    def test_integrity_precedes_improvement(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        trust = assess_trust(
            evidence(change, parent_delta=0.02, contract_hash_matches=False)
        )
        self.assertEqual(trust.verdict, Verdict.SUSPICIOUS)
        self.assertIn("CONTRACT_HASH_MISMATCH", trust.flags)

    def test_confirmed_positive_negative_and_within_noise(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        positive = assess_trust(evidence(change, parent_delta=0.01))
        self.assertEqual(positive.verdict, Verdict.ACCEPTED)
        within = assess_trust(
            evidence(
                change,
                parent_delta=0.0005,
                seed_scores=(0.5998, 0.6004, 0.6001),
            )
        )
        self.assertEqual(within.verdict, Verdict.INCONCLUSIVE)
        negative = assess_trust(
            evidence(
                change,
                parent_delta=-0.01,
                seed_scores=(0.5900, 0.5905, 0.5898),
            )
        )
        self.assertEqual(negative.verdict, Verdict.NEGATIVE)

    def test_confirmed_directional_gain_below_ladder_is_trusted_for_search(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        trust = assess_trust(
            evidence(
                change,
                parent_delta=0.00083,
                parent_primary=0.601468756352959,
                seed_scores=(
                    0.6022655471814293,
                    0.6022722994862122,
                    0.6022983340637944,
                ),
            )
        )

        self.assertEqual(trust.verdict, Verdict.ACCEPTED)
        self.assertEqual(trust.stability, Stability.CONFIRMED)
        self.assertIn("CONFIRMED_POSITIVE_BELOW_LADDER", trust.flags)
        self.assertLess(trust.seed_mean - 0.601468756352959, trust.eta_applied)

    def test_proxy_uses_symmetric_noise_band_before_pruning(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        proxy = {
            "population": Population.INTERNAL_PROXY,
            "fidelity": Fidelity.PROXY,
            "seed_scores": (0.60,),
        }

        positive = assess_trust(evidence(change, parent_delta=0.002, **proxy))
        near_positive = assess_trust(
            evidence(change, parent_delta=0.0001, **proxy)
        )
        near_negative = assess_trust(
            evidence(change, parent_delta=-0.0001, **proxy)
        )
        negative = assess_trust(evidence(change, parent_delta=-0.002, **proxy))

        self.assertEqual(positive.verdict, Verdict.ACCEPTED)
        self.assertEqual(near_positive.verdict, Verdict.INCONCLUSIVE)
        self.assertIn("WITHIN_NOISE", near_positive.flags)
        self.assertEqual(near_negative.verdict, Verdict.INCONCLUSIVE)
        self.assertIn("WITHIN_NOISE", near_negative.flags)
        self.assertEqual(negative.verdict, Verdict.NEGATIVE)

    def test_proxy_full_direction_conflict_is_advisory_during_confirmation(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        trust = assess_trust(
            evidence(
                change,
                parent_primary=0.601468756352959,
                parent_delta=0.00007795016284650735,
                seed_scores=(0.6015467065158056,),
                internal_proxy_delta=-0.0006046776713612978,
            )
        )

        self.assertEqual(trust.verdict, Verdict.ACCEPTED)
        self.assertEqual(trust.stability, Stability.SINGLE_SEED)
        self.assertEqual(trust.integrity, Integrity.CLEAN)
        self.assertIn("PROXY_FULL_DIRECTION_CONFLICT", trust.flags)
        self.assertNotIn("PROXY_FULL_SIGN_CONFLICT", trust.flags)

    def test_cross_population_and_metric_conflicts_are_visible(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        suspicious = assess_trust(
            evidence(change, parent_delta=0.01, unbiased_audit_delta=-0.001)
        )
        self.assertEqual(suspicious.verdict, Verdict.SUSPICIOUS)
        lopsided = assess_trust(
            evidence(
                change,
                parent_delta=0.01,
                metric_deltas={"GAUC": 0.02, "nDCG@5": -0.001},
            )
        )
        self.assertIn("METRIC_DIRECTION_CONFLICT", lopsided.flags)
        prohibited = assess_trust(
            evidence(
                change,
                parent_delta=0.01,
                metric_deltas={"GAUC": 0.02, "nDCG@5": -0.001},
            ),
            TrustConfig(require_non_decreasing_metrics=True),
        )
        self.assertEqual(prohibited.verdict, Verdict.INCONCLUSIVE)

    def test_validation_arm_conflict_is_advisory(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        trust = assess_trust(
            evidence(
                change,
                parent_delta=0.01,
                val_a_delta=0.02,
                val_b_delta=-0.001,
            )
        )

        self.assertEqual(trust.verdict, Verdict.ACCEPTED)
        self.assertIn("VALIDATION_ARM_SIGN_CONFLICT", trust.flags)

    def test_validation_arm_gap_and_temporal_drift_are_visible(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        trust = assess_trust(
            evidence(
                change,
                parent_delta=0.01,
                val_a_delta=0.02,
                val_b_delta=0.01,
                drift_primary_slope=-0.003,
            )
        )

        self.assertIn("VALIDATION_ARM_GAP", trust.flags)
        self.assertIn("DRIFT_DETECTED", trust.flags)

    def test_redundancy_does_not_claim_seed_confirmation(self):
        change = analyze_prediction_change([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])
        trust = assess_trust(
            evidence(
                change,
                parent_delta=0.01,
                seed_scores=(0.61,),
                delta_correlation=0.9,
                delta_correlation_experiment_id="exp_0001",
            )
        )
        self.assertEqual(trust.verdict, Verdict.REDUNDANT)
        self.assertEqual(trust.stability, Stability.NOT_APPLICABLE)


if __name__ == "__main__":
    unittest.main()
