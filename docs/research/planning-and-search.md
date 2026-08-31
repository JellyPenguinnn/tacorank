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
4. The hash-bound local paper bank selects a small method-relevant set of
   advisory references. The planner may cite zero or more of them; any cited
   record must match the supplied immutable snapshot exactly.
5. `ResearchProvider` receives one bounded structured request and may make one
   format-repair call.
6. `PlanValidator` checks research identity and policy, any evidence citations,
   duplicate identity, hidden-data references, cost budgets, and the absence of
   code-specific instructions.
7. The planner returns exactly one `ResearchProposal`, an advisory stop, or a
   blocked result. The controller then binds the authorized code target and
   execution ladder into an `ExperimentSpec` for the coder. The harness owns
   event logging and all subsequent gates.

Evaluation-driven family selection must apply the mandatory decision order in
`research/CURRENT_RUN_IMPROVEMENT_PLAN.md` before the generic two-phase search
policy. Invalid, suspicious, no-op, unstable, and proxy-only results never
become positive research rewards or parent nodes.

The proxy trust assessment keeps a symmetric `0.0016` primary-score noise
band, but the resource gate promotes an inconclusive within-noise proxy only
when its measured parent delta is positive. A zero or negative within-noise
proxy is cleanly pruned with `PROXY_NON_POSITIVE_WITHIN_NOISE`; results below
`-0.0016` remain hard proxy regressions. Proxy results never become parents or
best checkpoints directly.

Full-fidelity branching and validation-best selection use separate gates. A
clean, confirmed result within `0.0016` of the current validation best may be
an explicitly accepted exploratory DFS parent only when its aggregate gain
over its declared parent is strictly positive. It remains
ineligible for validation-best selection until it clears the full Ladder
threshold against the current best. Equal-score copies and regressions no
longer expand the frontier.

Proxy/full direction disagreement is advisory rather than an integrity
failure. The controller records `PROXY_FULL_DIRECTION_CONFLICT` and completes
seed confirmation. Only concrete evaluator, contract, alignment,
forbidden-input, or output evidence causes integrity quarantine. A suspicious
but non-compromised experiment is excluded from reward and all future lineage,
then search backtracks to a verified eligible parent and an independent legal
method. Only compromised integrity, or actual exhaustion of legal choices,
stops the loop at this gate.

## Search policy

The core policy is deterministic, score-guided AIDE-style depth-first search:

- rank at most three trusted or controller-approved exploratory frontier nodes
  by higher confirmed primary score,
  deeper lineage, and stable newest-ID tie-breaking;
- continue from the best-ranked branch while it has a legal untried method,
  and backtrack to the next trusted branch only when that branch is exhausted;
- when the latest clean result changes predictions but has no trusted gain,
  select the highest-scoring non-baseline eligible node rather than blindly
  extending the newest node; prefer an untried method in that node's family,
  permit one materially different same-card child when the family exposes only
  one method, and switch families only after that parent/family route is used;
- do not require every research family to be probed from the baseline before
  deepening a better branch; family order remains a deterministic tie-break for
  legal methods on the selected parent;
- only baseline roots with a verified decision and non-root experiments with a
  clean, confirmed full-fidelity result may be branch parents; an inconclusive
  node additionally requires explicit exploratory parent approval and must
  remain within `0.0016` of validation best;
- clean proxy/full results within one `epsilon` of their parent, or with an
  eligible component-metric trade-off inside that same bound, are soft-pruned
  rather than forgotten;
- a soft result may receive at most one documented metric-trade-off refinement
  and never becomes validation-best eligible through that permission;
- a clean soft result whose prediction Spearman magnitude is below `0.98` may
  enter one fixed residual-ensemble test from the trusted parent;
- severe regressions, rejected outputs, suspicious/compromised results,
  unstable results, and invalid/retracted nodes are hard-pruned as parents and
  checkpoints;
- a suspicious non-compromised node is quarantined and does not stop search
  while an independent method remains legal from a verified eligible frontier
  node;
- after Person 4's single bounded Trae wiring-repair action still produces a
  no-op, recovery returns a neutral `no_op` node and its unchanged-prediction
  evidence to the planner without emitting a prune decision;
- the legal-choice ranker then receives both one bounded same-mechanism
  reimplementation from the last trusted parent and the available independent
  mechanisms; choosing an independent mechanism retires that branch, while a
  second no-op for the same parent/family/method retires the reimplementation
  option;
- an implementation that becomes `invalid` before protected evaluation gets
  one policy-selected reimplementation from the same trusted parent; a second
  operational failure retires that parent/family/method combination without
  treating either failure as research evidence;
- a stateless LinUCB ranker is reconstructed from verified ledger history and
  reorders only the legal choices emitted by these deterministic gates. It
  cannot invent a family, method, parent, refinement, or ensemble component.

New live deployments use a parallel width of two. Lane zero is the normal
policy choice and may repeat a method only for an explicit outcome-routed
refinement or deepening. Any spare lane is restricted to a method card never
attempted anywhere in the ledger. When no globally untried scouting method
remains, the round automatically contracts to one lane.

`eligible_frontier`, `refinement_frontier_ids`, and `ensemble_candidate_ids`
are separate authoritative context collections. Empty collections remain empty;
Person 1 does not resurrect candidates from generic history. Canonical
`parent_eligible` and `best_eligible` continue to control trusted branching and
checkpoint selection.

Semantic duplicate identity is `parent + family + method cards + ensemble
components`. Rephrasing the same method does not normally authorize another
trial from the same parent. The only exceptions are policy-selected, single
reimplementations after either a verified no-op or an operationally invalid
attempt with no protected evaluation. The planner cannot repeat either after a
second matching failure. A genuine refinement receives a new parent commit and
therefore a distinct identity.

## Interfaces owned by Person 1

- `research.graph_view`: read-only lineage projection;
- `research.search_policy`: deterministic parent/family selection;
- `research.search_eligibility`: derived hard/soft/refinement/ensemble flags;
- `research.linucb`: legal-choice-only contextual-bandit ranker;
- `research.portfolio` and `research.method_cards`: documented method cards;
- `research.duplicate_detection`: stable duplicate identity;
- `research.plan_validation`: pure proposal validator;
- `research.paper_bank`: deterministic retrieval from the frozen 70-paper bank;
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
