import unittest

from tacorank.evaluation.metrics import (
    auc,
    evaluate_independent,
    ndcg_at_k,
    validate_metric_set,
)


class MetricTests(unittest.TestCase):
    def test_auc_ties_match_expected_probability(self):
        self.assertAlmostEqual(auc([0, 1, 0, 1], [0.1, 0.2, 0.2, 0.9]), 0.875)

    def test_zero_positive_users_count_as_zero_ndcg(self):
        result = evaluate_independent(
            ["a", "a", "b", "b"],
            [0, 0, 1, 0],
            [0.4, 0.3, 0.9, 0.1],
        )
        self.assertEqual(result["GAUC"], 1.0)
        self.assertEqual(result["nDCG@5"], 0.5)
        self.assertEqual(result["primary"], 0.75)

    def test_metric_registry_rejects_primary_mismatch_and_extra_metric(self):
        with self.assertRaisesRegex(ValueError, "aggregation mismatch"):
            validate_metric_set(
                {"GAUC": 0.7, "nDCG@5": 0.5, "primary": 0.7},
                ("GAUC", "nDCG@5"),
                "primary",
                {"GAUC": 0.5, "nDCG@5": 0.5},
            )
        with self.assertRaisesRegex(ValueError, "undeclared"):
            validate_metric_set(
                {"GAUC": 0.7, "nDCG@5": 0.5, "rogue": 1.0, "primary": 0.6},
                ("GAUC", "nDCG@5"),
                "primary",
                {"GAUC": 0.5, "nDCG@5": 0.5},
            )

    def test_ndcg_rejects_nonbinary_labels(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            ndcg_at_k([2, 0], 5)


if __name__ == "__main__":
    unittest.main()
