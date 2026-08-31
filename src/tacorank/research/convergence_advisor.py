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

        return ConvergenceAdvice(
            action="propose",
            reason_code="SEARCH_CONTINUES",
            reason=(
                "No deterministic budget is exhausted. Convergence and target "
                "stopping are owned by the controller."
            ),
        )
