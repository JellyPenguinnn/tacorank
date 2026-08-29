from pathlib import Path
import tempfile
import unittest

from benchmarks.kuairand_pure.submission_adapter import validate_submission


class SubmissionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            (20220422, "u1", "v1"),
            (20220422, "u1", "v1"),
            (20220422, "u2", "v2"),
        ]

    def _write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_duplicate_pairs_are_preserved_by_row_id(self):
        path = self._write(
            "row_id,user_id,video_id,score\n"
            "0,u1,v1,0.1\n"
            "1,u1,v1,0.2\n"
            "2,u2,v2,0.3\n"
        )
        result = validate_submission(path, self.rows)
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.scores, (0.1, 0.2, 0.3))

    def test_alignment_and_nonfinite_scores_are_rejected(self):
        wrong = self._write(
            "row_id,user_id,video_id,score\n"
            "0,u1,v1,0.1\n1,u2,v2,0.2\n2,u2,v2,0.3\n"
        )
        with self.assertRaisesRegex(ValueError, "align"):
            validate_submission(wrong, self.rows)
        nonfinite = self._write(
            "row_id,user_id,video_id,score\n"
            "0,u1,v1,0.1\n1,u1,v1,nan\n2,u2,v2,0.3\n"
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_submission(nonfinite, self.rows)


if __name__ == "__main__":
    unittest.main()
