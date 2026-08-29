import unittest

from tacorank.evaluation.metrics import evaluate_independent
from tacorank.evaluation.slices import (
    delta_vector,
    duration_threshold_bucket,
    gain_concentration,
    impression_bucket,
    positive_rank_diagnostics,
    reconstruct_from_user_slices,
    slice_users,
    user_metrics,
)


class SliceTests(unittest.TestCase):
    def setUp(self):
        self.users = ["a", "a", "b", "b", "b", "c", "c"]
        self.labels = [1, 0, 1, 1, 0, 0, 0]
        self.scores = [0.9, 0.1, 0.3, 0.8, 0.2, 0.7, 0.6]

    def test_user_slices_reconstruct_official_metrics(self):
        official = evaluate_independent(self.users, self.labels, self.scores)
        slices = slice_users(user_metrics(self.users, self.labels, self.scores), impression_bucket)
        rebuilt = reconstruct_from_user_slices(slices)
        self.assertAlmostEqual(rebuilt["GAUC"], official["GAUC"])
        self.assertAlmostEqual(rebuilt["nDCG@5"], official["nDCG@5"])
        self.assertAlmostEqual(rebuilt["primary"], official["primary"])

    def test_delta_vector_sums_to_primary_delta(self):
        parent = [0.1, 0.9, 0.2, 0.3, 0.8, 0.7, 0.6]
        _, vector = delta_vector(self.users, self.labels, self.scores, parent)
        candidate_score = evaluate_independent(self.users, self.labels, self.scores)["primary"]
        parent_score = evaluate_independent(self.users, self.labels, parent)["primary"]
        self.assertAlmostEqual(sum(vector), candidate_score - parent_score)
        self.assertGreaterEqual(gain_concentration(vector), 0.0)

    def test_row_level_rank_diagnostics_are_bounded_and_nonadditive(self):
        diagnostics = positive_rank_diagnostics(
            self.users,
            self.labels,
            self.scores,
            [duration_threshold_bucket(value) for value in (6000, 6000, 10000, 20000, 10000, 20000, 20000)],
        )
        self.assertTrue(diagnostics)
        self.assertTrue(
            all(0.0 <= value.mean_normalized_rank <= 1.0 for value in diagnostics.values())
        )


if __name__ == "__main__":
    unittest.main()
