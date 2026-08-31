"""The generic routes must never emit an ensemble without components.

run_20260831T_v4 ended at experiment 14 of 15 on an invalid-provider-plan
checkpoint: the policy offered the ensemble family through a route that
defaults component_experiment_ids to (), and Person 1 validation rejected it
with ENSEMBLE_COMPONENT_REQUIRED.
"""

from __future__ import annotations

from tacorank.research.search_policy import _method_for_family

from .conftest import make_summary  # noqa: F401  (fixture module import)


def test_generic_route_declines_the_ensemble_family(planner_context):
    context = planner_context

    assert _method_for_family(context, "ensemble") is None


def test_named_ensemble_card_is_still_available_to_the_portfolio_route(
    planner_context,
):
    # The soft-portfolio route passes an explicit preferred card and supplies
    # the components alongside it, so the guard must not reach it.
    from types import SimpleNamespace

    contract = SimpleNamespace(**vars(planner_context.contract_summary))
    contract.research_capabilities = list(
        getattr(contract, "research_capabilities", []) or []
    ) + ["diverse_clean_proxy_member"]
    context = SimpleNamespace(
        **{**vars(planner_context), "contract_summary": contract}
    )

    card = _method_for_family(
        context,
        "ensemble",
        preferred="ensemble_diverse_residual_candidate",
        parent_experiment_id="exp_0001",
    )

    assert card is not None


def test_other_families_are_unaffected(planner_context):
    assert _method_for_family(planner_context, "objective") is not None
