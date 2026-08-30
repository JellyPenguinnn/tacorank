# Adaptive research campaigns

Campaign files freeze the search boundary, not every experiment design.

`objective_temporal_50.json` reserves 25 experiments for `objective`, followed
by 25 for `temporal_history`. For each slot, the research planner receives the
active family's prior outcomes and chooses one currently eligible method card,
a distinct formulation, and its hyperparameters. The exact choice is stored as
`variant_instruction` plus a canonical `variant_parameters` signature in the
experiment specification and shown in generated reports.

This is an exploratory depth campaign: its per-family minimum and patience are
both equal to the 25-slot family budget, so the generic three-result convergence
signal does not end the campaign. Standard non-campaign runs retain the frozen
epsilon/patience stop and six-hour wall-clock ceiling.

The deterministic controller still owns family order, budgets, parent
eligibility, recovery, execution, evaluation, duplicate rejection, and final
selection. Campaign duplicate identity uses the selected method card and
structured parameter signature while excluding the chronological slot and Git
parent, so moving to a new slot or rephrasing an instruction cannot make an
identical scientific design legal.
