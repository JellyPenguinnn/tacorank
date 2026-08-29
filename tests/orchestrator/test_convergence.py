from __future__ import annotations

from tacorank.accounting import aggregate_resources
from tacorank.orchestrator.convergence import stop_decision
from tacorank.orchestrator.state import RunState
from tacorank.schemas import ResourceDelta, TokenMeasurement


def test_unmeasured_tokens_still_exhaust_the_frozen_limit(config):
    limited_config = config.model_copy(update={"token_limit": 10})
    state = RunState(
        resource_totals=aggregate_resources(
            [
                ResourceDelta(
                    llm_input_tokens=6,
                    llm_output_tokens=4,
                    token_measurement=TokenMeasurement.NONE,
                )
            ]
        )
    )

    decision = stop_decision(state, [], limited_config)

    assert state.resource_totals.total_reported_tokens == 10
    assert decision.stop
    assert decision.reason_code == "token_budget"
