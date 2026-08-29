# Person 1 research planning and search contract

This module is the research-intelligence boundary for RankForge/TacoRank.
It consumes a verified `PlannerContext` and returns one `PlannerOutput`.
Person 1 does not mutate the run, write `events.jsonl`, edit code, execute a
candidate, compute metrics, or enforce the final stop gate. Those operations
remain with the outer harness and evaluation owners.

## Control flow

1. Person 2 assembles a deterministic context from `COMPETITION.md`, Git, and
   the append-only event ledger. For the current run it also retrieves the
   applicable rule from `research/CURRENT_RUN_IMPROVEMENT_PLAN.md` and the
   referenced method cards; the playbook itself remains read-only.
2. `ConvergenceAdvisor` checks only advisory budget/patience conditions.
3. `SearchPolicy` creates a transient `GraphView` and chooses one eligible
   parent plus one legal experiment family.
4. `ResearchProvider` receives one bounded structured request and may make one
   format-repair call.
5. `PlanValidator` checks identity, lineage, contract paths, fidelity order,
   evidence citations, duplicate identity, hidden-data references, and cost
   budgets.
6. The planner returns exactly one proposal, an advisory stop, or a blocked
   result. The harness owns event logging and all subsequent gates.

Evaluation-driven family selection must apply the mandatory decision order in
`research/CURRENT_RUN_IMPROVEMENT_PLAN.md` before the generic two-phase search
policy. Invalid, suspicious, no-op, unstable, and proxy-only results never
become positive research rewards or parent nodes.

## Search policy

The core policy is deterministic two-phase AIDE-style beam search:

- breadth first probes untried high-value families in this order: objective,
  temporal history, multitask, duration bias, temporal features, and model;
- depth then retains at most three trusted frontier nodes, preferring higher
  trusted primary score, fewer children, family diversity, and stable ID
  tie-breaking;
- only baseline roots with a verified decision and non-root experiments with a
  trusted full-fidelity result may be parents;
- LinUCB/UCB and Reflexion-guided branching are optional ablations after the
  deterministic loop is complete. They are not required for the core path.

## Interfaces owned by Person 1

- `research.graph_view`: read-only lineage projection;
- `research.search_policy`: deterministic parent/family selection;
- `research.portfolio` and `research.method_cards`: documented method cards;
- `research.duplicate_detection`: stable duplicate identity;
- `research.plan_validation`: pure proposal validator;
- `agents.research_planner`: bounded provider adapter;
- `providers.research_provider`: provider protocol and test double.

The shared Pydantic schema definitions remain in `tacorank.schemas`, owned by
Person 2. The planner imports those definitions at runtime and deliberately
does not redefine them.

## Memory-schema-v1 compatibility

Person 1 accepts the memory schema's planner-facing identifiers and evidence
references: `run_YYYYMMDD_<slug>`, `exp_0000`, `ctx_planner_000000`,
`evt_000000`, and lowercase Git commit hashes. Method cards use a first fenced
JSON block with `schema_version: "1.0"`, followed by the required Markdown
sections. The planner validates the contract SHA-256, parent commit identity,
evidence-event membership, and duplicate identity before returning a proposal.

The append-only event envelope, hash chain, replay/fold logic, idempotency,
and derived run-state remain one shared harness concern. Person 1 supplies
event IDs and immutable context references but never appends `events.jsonl`.
