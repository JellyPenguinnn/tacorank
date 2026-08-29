import unittest

from tacorank.evaluation.types import (
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
from tacorank.reflection.research import (
    ActiveLesson,
    build_research_lesson,
    recommend_frame_staleness,
)


def evaluation(verdict, fidelity=Fidelity.FULL, stability=Stability.CONFIRMED):
    return EvaluationResult(
        "run_x", "exp_0001", 1, Population.PUBLIC_VALIDATION, fidelity, 0, 1,
        "a" * 64, "b" * 64, "c" * 64,
        MetricSet({"GAUC": 0.67, "nDCG@5": 0.55}, "primary", 0.61),
        MetricDelta(0.01, {"GAUC": 0.02, "nDCG@5": -0.001}),
        MetricDelta(0.01, {"GAUC": 0.02, "nDCG@5": -0.001}),
        MetricDelta(0.01, {"GAUC": 0.02, "nDCG@5": -0.001}),
        PredictionChange(0.5, 1.0, 0.0, 1.0),
        TrustAssessment(verdict, stability, Integrity.CLEAN, ("METRIC_DIRECTION_CONFLICT",)),
    )


def lesson(result):
    return build_research_lesson(
        result,
        ["evt_000010"],
        ["a" * 40],
        "objective",
        "pairwise ranking improves alignment",
        "within-user pairs align training and evaluation",
        "groups with both classes",
        "single-class or cross-user groups",
        "exp_0000",
    )


class ReflectionTests(unittest.TestCase):
    def test_no_op_and_proxy_negative_do_not_create_lessons(self):
        self.assertIsNone(lesson(evaluation(Verdict.NO_OP, stability=Stability.NOT_APPLICABLE)))
        self.assertIsNone(lesson(evaluation(Verdict.NEGATIVE, fidelity=Fidelity.PROXY)))

    def test_confirmed_result_preserves_metric_conflict(self):
        created = lesson(evaluation(Verdict.ACCEPTED))
        self.assertIsNotNone(created)
        self.assertIn("opposite directions", created.summary)
        self.assertEqual(created.origin, "research")

    def test_frame_move_marks_only_explicit_content_lessons_stale(self):
        recommendations = recommend_frame_staleness(
            "exp_0008",
            "evt_000020",
            [
                ActiveLesson("lesson_0001", "research_result", ("feature",), "exp_0002"),
                ActiveLesson("lesson_0002", "process_rule", ("feature",), "exp_0002"),
                ActiveLesson("lesson_0003", "research_result", ("objective",), "exp_0002"),
            ],
        )
        self.assertEqual([value.lesson_id for value in recommendations], ["lesson_0001"])


if __name__ == "__main__":
    unittest.main()
