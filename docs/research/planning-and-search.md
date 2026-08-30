# Person 1 research planning and search contract

This module is the research-intelligence boundary for TacoRank.
It consumes a verified `PlannerContext` and returns one `PlannerOutput`.
Person 1 does not mutate the run, write `events.jsonl`, edit code, execute a
candidate, compute metrics, or enforce the final stop gate. Those operations
remain with the outer harness and evaluation owners.

## Control flow

1. Person 2 assembles a deterministic, code-blind context from the frozen
   research policy and append-only evaluation feedback. For the current run it
   also retrieves the applicable high-level research rule and scientific method
   overviews; repository paths and implementation details stay controller-side.
2. `ConvergenceAdvisor` checks only advisory budget/patience conditions.
3. `SearchPolicy` creates a transient `GraphView`, applies the authoritative
   branch/refinement/ensemble portfolios, and chooses one legal search action.
4. `ResearchProvider` receives one bounded structured request and may make one
   format-repair call.
5. `PlanValidator` checks research identity and policy, evidence citations,
   duplicate identity, hidden-data references, cost budgets, and the absence of
   code-specific instructions.
6. The planner returns exactly one `ResearchProposal`, an advisory stop, or a
   blocked result. The controller then binds the authorized code target and
   execution ladder into an `ExperimentSpec` for the coder. The harness owns
   event logging and all subsequent gates.

Evaluation-driven family selection must apply the mandatory decision order in
`research/CURRENT_RUN_IMPROVEMENT_PLAN.md` before the generic two-phase search
policy. Invalid, suspicious, no-op, unstable, and proxy-only results never
become positive research rewards or parent nodes.

## Search policy

The core policy is deterministic AIDE-style portfolio search:

- breadth first probes untried high-value families in this order: objective,
  temporal history, multitask, duration bias, temporal features, and model;
- depth then retains at most three trusted frontier nodes, preferring higher
  trusted primary score, fewer children, family diversity, and stable ID
  tie-breaking;
- only baseline roots with a verified decision and non-root experiments with a
  trusted full-fidelity result may be normal branch parents; a soft node can be
  used only as the base of its one authorized refinement;
- clean proxy/full results within `max(5 * epsilon, 0.01)` of their parent, or
  with a component-metric trade-off, are soft-pruned rather than forgotten;
- a soft result may receive at most one documented metric-trade-off refinement
  and never becomes validation-best eligible through that permission;
- a clean soft result whose prediction Spearman magnitude is below `0.98` may
  enter one fixed residual-ensemble test from the trusted parent;
- severe regressions, rejected outputs, suspicious/compromised results,
  unstable results, and invalid/retracted nodes are hard-pruned as parents and
  checkpoints;
- after Person 4's single bounded Trae wiring-repair action still produces a
  no-op, recovery returns a neutral `no_op` node and its unchanged-prediction
  evidence to the planner without emitting a prune decision;
- the legal-choice ranker then receives both one bounded same-mechanism
  reimplementation from the last trusted parent and the available independent
  mechanisms; choosing an independent mechanism retires that branch, while a
  second no-op for the same parent/family/method retires the reimplementation
  option;
- a stateless LinUCB ranker is reconstructed from verified ledger history and
  reorders only the legal choices emitted by these deterministic gates. It
  cannot invent a family, method, parent, refinement, or ensemble component.

`eligible_frontier`, `refinement_frontier_ids`, and `ensemble_candidate_ids`
are separate authoritative context collections. Empty collections remain empty;
Person 1 does not resurrect candidates from generic history. Canonical
`parent_eligible` and `best_eligible` continue to control trusted branching and
checkpoint selection.

Semantic duplicate identity is `parent + family + method cards + ensemble
components`. Rephrasing the same method does not normally authorize another
trial from the same parent. The only exception is the policy-selected, single
reimplementation after a verified no-op; the planner cannot repeat it after a
second no-op. A genuine refinement receives a new parent commit and therefore a
distinct identity.

## Interfaces owned by Person 1

- `research.graph_view`: read-only lineage projection;
- `research.search_policy`: deterministic parent/family selection;
- `research.search_eligibility`: derived hard/soft/refinement/ensemble flags;
- `research.linucb`: legal-choice-only contextual-bandit ranker;
- `research.portfolio` and `research.method_cards`: documented method cards;
- `research.duplicate_detection`: stable duplicate identity;
- `research.plan_validation`: pure proposal validator;
- `agents.research_planner`: bounded provider adapter;
- `providers.research_provider`: provider protocol and test double.

The shared Pydantic schema definitions remain in `tacorank.schemas`, owned by
Person 2. The planner imports the code-blind `ResearchProposal` definition at
runtime and deliberately does not redefine it.

## Memory-schema-v1 compatibility

Person 1 accepts the memory schema's planner-facing identifiers and evidence
references: `run_YYYYMMDD_<slug>`, `exp_0000`, `ctx_planner_000000`,
`evt_000000`. Method cards use a first fenced JSON block with
`schema_version: "1.0"`, followed by the required Markdown sections. The
planner validates the contract SHA-256, controller-supplied parent identity,
evidence-event membership, and duplicate identity before returning a proposal.
The external research model never receives commit hashes, repository paths,
source interfaces, commands, or execution stages.

The append-only event envelope, hash chain, replay/fold logic, idempotency,
and derived run-state remain one shared harness concern. Person 1 supplies
event IDs and immutable context references but never appends `events.jsonl`.
