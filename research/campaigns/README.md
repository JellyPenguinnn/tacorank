# Legacy opt-in research campaigns

Campaign files remain available for controlled compatibility studies. They are
not the production default and must be supplied explicitly with
`setup-live --research-campaign`. A normal run uses first-class adaptive
research plans and proposes several conditional experiments per scientific
question.

`objective_temporal_50.json` reserves 25 experiments for `objective`, followed
by 25 for `temporal_history`. For each slot, the research planner receives the
active family's prior outcomes and chooses one currently eligible method card,
a distinct formulation, and its hyperparameters. The exact choice is stored as
`variant_instruction` plus a canonical `variant_parameters` signature in the
experiment specification and shown in generated reports.

This historical campaign still declares 25+25 slots as a maximum search
boundary. The global three-result non-improvement rule and six-hour wall-clock
ceiling apply to it as they do to every run, so those slots are not a quota and
the controller may stop much earlier.

The deterministic controller still owns family order, budgets, parent
eligibility, recovery, execution, evaluation, duplicate rejection, and final
selection. Campaign duplicate identity uses the selected method card and
structured parameter signature while excluding the chronological slot and Git
parent, so moving to a new slot or rephrasing an instruction cannot make an
identical scientific design legal.
