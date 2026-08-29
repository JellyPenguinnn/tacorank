import unittest

from tacorank.reporting.resources import aggregate_resources


class ReportingTests(unittest.TestCase):
    def test_provider_and_estimated_tokens_stay_separate(self):
        summary = aggregate_resources(
            [
                {
                    "llm_input_tokens": 10,
                    "llm_output_tokens": 2,
                    "token_measurement": "provider",
                    "gpu_time_ms": 3_600_000,
                    "gpu_count": 1,
                    "manual_interventions": 0,
                },
                {
                    "llm_input_tokens": 7,
                    "llm_output_tokens": 3,
                    "token_measurement": "estimated",
                    "gpu_time_ms": 0,
                    "gpu_count": 0,
                    "manual_interventions": 1,
                },
            ]
        )
        self.assertEqual(summary.llm_input_tokens_provider, 10)
        self.assertEqual(summary.llm_input_tokens_estimated, 7)
        self.assertEqual(summary.gpu_hours, 1.0)
        self.assertEqual(summary.manual_interventions, 1)


if __name__ == "__main__":
    unittest.main()
