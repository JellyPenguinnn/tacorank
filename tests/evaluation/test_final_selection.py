import unittest

from tacorank.evaluation.final_selection import CandidateEvidence, rank_average, select_final
from tacorank.evaluation.types import Integrity, Stability, Verdict


def candidate(experiment_id, public, val_b, clean=True):
    return CandidateEvidence(
        experiment_id,
        experiment_id.replace("exp", "commit"),
        public,
        val_b,
        Verdict.ACCEPTED,
        Stability.CONFIRMED,
        Integrity.CLEAN,
        True,
        True,
        clean,
    )


class FinalSelectionTests(unittest.TestCase):
    def test_final_selection_uses_audit_split_after_filtering(self):
        selected = select_final(
            [
                candidate("exp_0001", 0.63, 0.61),
                candidate("exp_0002", 0.62, 0.615),
                candidate("exp_0003", 0.64, 0.64, clean=False),
            ]
        )
        self.assertEqual(selected.experiment_id, "exp_0002")

    def test_rank_average_ignores_score_scale(self):
        averaged = rank_average([[0.1, 0.2, 0.3], [10.0, 20.0, 30.0]])
        self.assertEqual(averaged, (0.0, 0.5, 1.0))

    def test_missing_val_b_evidence_is_not_final_eligible(self):
        with self.assertRaisesRegex(ValueError, "no trusted"):
            select_final([candidate("exp_0001", 0.63, None)])


if __name__ == "__main__":
    unittest.main()
