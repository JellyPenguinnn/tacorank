"""Abandoning an experiment must always leave a terminal decision.

An experiment promoted at smoke or proxy has a pending `promote` as its latest
decision. If recovery then abandons it without recording a terminal decision,
the search policy reports FIDELITY_PROMOTION_REQUIRED forever and the run stops
on no_legal_proposal with its remaining iterations unused.

run_20260831T005157Z lost thirteen of fifteen iterations that way:

  DECIDED  exp_002 promote PROXY_WITHIN_NOISE
  RECOVERY retry_same_commit TRANSIENT_SAME_COMMIT_RETRY
  RECOVERY abandon           REPEATED_ERROR_FINGERPRINT
  PLANNER  blocked FIDELITY_PROMOTION_REQUIRED
"""

from __future__ import annotations

import inspect
import re

from tacorank.orchestrator import router as router_module
from tacorank.schemas import ExperimentDecisionKind


def test_every_terminal_kind_is_recognised():
    # A decision kind missing here would make the guard re-record a decision
    # that already exists, or fail to notice one that does.
    assert router_module.Harness._TERMINAL_DECISIONS == frozenset(
        {
            ExperimentDecisionKind.ACCEPT,
            ExperimentDecisionKind.REJECT,
            ExperimentDecisionKind.PRUNE,
            ExperimentDecisionKind.INVALID,
        }
    )
    assert ExperimentDecisionKind.PROMOTE not in (
        router_module.Harness._TERMINAL_DECISIONS
    )


def test_fidelity_loop_has_no_bare_abandon_exit():
    """Guard against reintroducing the deadlock on a path not yet covered.

    The first fix closed only the evaluation-failure exits, and the execution
    path deadlocked identically. Assert structurally that no stage-failure exit
    inside the fidelity loop returns without recording a decision.
    """

    source = inspect.getsource(router_module.Harness._run_one_experiment)
    body = source.split("output_causation_event_id")[1].split(
        "if fidelity == Fidelity.SMOKE:"
    )[0]
    bare = [
        line.strip()
        for line in body.splitlines()
        if line.strip() == "return self.state()"
    ]
    # The only permitted bare exit is the runtime-budget stop, which ends the
    # run rather than leaving the planner waiting on this experiment.
    budget_exits = len(
        re.findall(r"_stop_if_runtime_budget_exhausted\(\):\s*\n\s*return self\.state\(\)", body)
    )
    assert len(bare) == budget_exits, (
        "unguarded abandon exit in the fidelity loop: %d bare returns, %d budget stops"
        % (len(bare), budget_exits)
    )
