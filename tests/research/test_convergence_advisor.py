from types import SimpleNamespace

from tacorank.research.convergence_advisor import ConvergenceAdvisor


def test_advisor_recommends_stop_when_experiment_budget_is_exhausted():
    context = SimpleNamespace(
        remaining_budget=SimpleNamespace(remaining_experiments=0),
        convergence=SimpleNamespace(patience=3, consecutive_non_improving_full_evaluations=0),
        source_event_ids=["evt_000010"],
    )

    advice = ConvergenceAdvisor().advise(context)

    assert advice.action == "recommend_stop"
    assert advice.reason_code == "EXPERIMENT_BUDGET_EXHAUSTED"
    assert advice.supporting_event_ids == ("evt_000010",)


def test_advisor_is_non_terminal_when_context_can_continue():
    context = SimpleNamespace(
        remaining_budget=SimpleNamespace(remaining_experiments=2),
        convergence=SimpleNamespace(
            patience=3,
            consecutive_non_improving_full_evaluations=1,
            full_evaluations_completed=2,
        ),
        source_event_ids=[],
    )

    assert ConvergenceAdvisor().advise(context).action == "propose"
