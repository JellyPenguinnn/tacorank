"""Pure convergence advice for the Person 1 planner.

The outer state machine and final stop gate belong to Person 2.  This module
only turns the verified planner context into an advisory recommendation; it
does not mutate state, write events, or enforce termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph_view import as_list, get_value


@dataclass(frozen=True)
class ConvergenceAdvice:
    action: str
    reason_code: str
    reason: str
    supporting_event_ids: tuple[str, ...] = ()


class ConvergenceAdvisor:
    """Return a stop recommendation only when verified context warrants it."""

    def advise(self, context: Any) -> ConvergenceAdvice:
        budget = get_value(context, "remaining_budget", None) or get_value(
            context, "remaining_budgets", None
        )
        convergence = get_value(context, "convergence", None)
        source_events = tuple(map(str, as_list(get_value(context, "source_event_ids", None))))

        for names, code, label in (
            (("remaining_experiments", "experiments_remaining", "experiments"), "EXPERIMENT_BUDGET_EXHAUSTED", "experiment"),
            (("remaining_public_queries", "public_queries_remaining", "remaining_public_validation_queries"), "QUERY_BUDGET_EXHAUSTED", "public-query"),
            (("remaining_wall_time_seconds", "wall_time_seconds", "agent_wall_time_seconds"), "WALL_TIME_BUDGET_EXHAUSTED", "wall-time"),
        ):
            for name in names:
                value = get_value(budget, name, None)
                if value is not None and float(value) <= 0:
                    return ConvergenceAdvice(
                        action="recommend_stop",
                        reason_code=code,
                        reason=f"The remaining {label} budget is exhausted.",
                        supporting_event_ids=source_events,
                    )

        patience = get_value(convergence, "patience", None)
        if patience is None:
            patience = get_value(convergence, "convergence_patience", None)
        no_improvement = get_value(convergence, "consecutive_non_improving_full_evaluations", 0)
        full_count = get_value(convergence, "full_evaluations_completed", None)
        coverage_complete = bool(
            get_value(convergence, "priority_coverage_complete", False)
        )
        if (
            patience is not None
            and float(no_improvement or 0) >= float(patience)
            and coverage_complete
        ):
            if full_count is None or int(full_count) >= 1:
                return ConvergenceAdvice(
                    action="recommend_stop",
                    reason_code="CONVERGENCE_PATIENCE_REACHED",
                    reason=(
                        f"No trusted primary improvement exceeded epsilon for "
                        f"{no_improvement} full evaluations."
                    ),
                    supporting_event_ids=source_events,
                )

        return ConvergenceAdvice(
            action="propose",
            reason_code="SEARCH_CONTINUES",
            reason="The verified context does not meet an advisory stop condition.",
        )
