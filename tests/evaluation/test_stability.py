import random
import unittest

from tacorank.evaluation.stability import Ladder, aggregate_seeds, seed_independence_passes


class StabilityTests(unittest.TestCase):
    def test_ladder_rejects_noise_and_keeps_reported_best(self):
        ladder = Ladder(100_000, quantize=False)
        self.assertTrue(ladder.submit([0.6000, 0.6008, 0.5996]).accepted)
        rng = random.Random(7)
        for _ in range(50):
            scores = [0.6000 + rng.gauss(0, 0.0008) for _ in range(3)]
            self.assertFalse(ladder.submit(scores).accepted)
        self.assertTrue(ladder.submit([0.6100, 0.6105, 0.6098]).accepted)
        before = ladder.best_reported
        self.assertFalse(ladder.submit([0.6101, 0.6099, 0.6103]).accepted)
        self.assertEqual(ladder.best_reported, before)

    def test_eta_has_noise_floor(self):
        aggregate = aggregate_seeds([0.6, 0.60001, 0.59999])
        self.assertEqual(aggregate.eta, 0.0016)

    def test_seed_independence_rejects_collapsed_variance(self):
        self.assertFalse(seed_independence_passes([0.6, 0.60001, 0.59999], 0.0008))
        self.assertTrue(seed_independence_passes([0.5992, 0.6, 0.6008], 0.0008))


if __name__ == "__main__":
    unittest.main()
