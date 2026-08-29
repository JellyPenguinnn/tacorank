# Person 1 — Automated Research and AIDE-Style Planning

## Codex implementation brief

You own TacoRank's research intelligence: interpreting verified evidence, maintaining an AIDE-style view of the experiment tree, deciding which trusted code state should be explored next, and producing one bounded `ExperimentSpec` at a time.

You do **not** own orchestration, durable memory, context assembly, code editing, sandbox execution, recovery, metric computation, or final stop enforcement. Those boundaries are deliberate. Return strict typed objects to Person 2's harness; never directly call another component or write shared state.

## 1. System objective and hard constraints

TacoRank must autonomously reproduce the official recommender baseline, propose code-level improvements, evaluate them on public validation only, learn from verified outcomes, recover from failures, converge under the frozen rule, and submit the validation-best trusted candidate once to hidden test.

The supplied problem material contains conflicting descriptions of the target label and metrics. Therefore:

- never hard-code `click`, `long_view`, GAUC, NDCG, Recall, or K values in research logic;
- read the resolved target and metrics only from the frozen `contract/COMPETITION.md` digest supplied in `PlannerContext`;
- reject planning if the contract is unresolved or its hash does not match the context;
- never request hidden-test information.

The official requirements demand per-iteration hypothesis, code diff, metrics, error/recovery evidence, token/GPU accounting, manual-intervention counting, baseline reproduction, and deterministic convergence. Your output must make the hypothesis and reasoning auditable.

## 2. Shared architecture and ownership

```text
Person 2 ContextBuilder
        ↓ PlannerContext
Person 1 SearchPolicy + ResearchPlanner
        ↓ ExperimentSpec
Person 2 Orchestrator
        ↓ CoderContext
Person 3 Trae → Gate A → sandbox → Gate B
        ↓ RunResult / OutputCheckResult
Person 4 monitors execution and decides recovery when required
Person 5 evaluates valid predictions and creates research feedback
        ↓ verified events
Person 2 appends events and builds the next PlannerContext
```

Only Person 2's orchestrator writes `events.jsonl`. Every other component returns a value. Do not create a private database, graph file, reflection file, or mutable run-state file.

## 3. Shared interface freeze

All shared models live in `src/tacorank/schemas.py`, owned by Person 2. Import them; do not redefine local dataclasses with similar fields.

### 3.1 Models consumed by Person 1

#### `PlannerContext`

```text
schema_version: str                     # exactly 1.0
context_id: str
run_id: str
contract_sha256: str
contract_summary: ContractSummary
baseline: ExperimentSummary
current_best: ExperimentSummary
eligible_frontier: list[ExperimentSummary]
family_history: list[ExperimentSummary]
active_lessons: list[LessonSummary]
method_cards: list[MethodCardSummary]
remaining_budget: BudgetSnapshot
convergence: ConvergenceSnapshot
public_validation_queries: int
source_event_ids: list[str]
context_artifact: ArtifactRef
```

`ExperimentSummary` contains only verified information: experiment ID, parent ID, commit SHA, family, hypothesis summary, highest completed fidelity, metrics when legal, trust verdict, decision, child count, actual cost, and evidence event IDs.

`LessonSummary` contains lesson ID, category, tags, active status, summary, applicability, avoid conditions, confidence, and source event IDs.

`MethodCardSummary` contains method ID, family, status, tags, cost tier, mechanism, prerequisites, expected effect, falsifier, and prohibition conditions.

### 3.2 Model produced by Person 1

#### `ExperimentSpec`

```text
schema_version: str                     # exactly 1.0
run_id: str
experiment_id: str
parent_experiment_id: str
parent_commit_sha: str
context_id: str
hypothesis: str
family: ExperimentFamily
change_summary: str
target_stage: str
target_files: list[str]
fidelity_plan: list[Fidelity]           # ordered subset of smoke, proxy, full
expected_mechanism: str
success_criteria: SuccessCriteria
falsification_condition: str
estimated_cost: CostEstimate
method_card_ids: list[str]
evidence_event_ids: list[str]
duplicate_key: str
```

Enums:

```text
Fidelity = smoke | proxy | full | final
ExperimentFamily = objective | sampling | temporal_history | features |
                   model | multitask | duration_bias | ensemble | other
```

`SuccessCriteria`:

```text
proxy_parent_delta_min: float | null
full_parent_delta_min: float
required_metric_direction: non_decreasing_all | primary_only | contract_defined
```

`CostEstimate`:

```text
llm_tokens_upper_bound: int
wall_time_seconds_upper_bound: int
gpu_seconds_upper_bound: int
cost_tier: low | medium | high
```

### 3.3 Planner return envelope

#### `PlannerOutput`

```text
action: propose | recommend_stop | blocked
spec: ExperimentSpec | null
reason_code: str
reason: str
supporting_event_ids: list[str]
```

This is the sole return type of `ResearchPlanner.propose`. Enforce it as a discriminated contract:

- `action=propose` requires exactly one non-null, fully validated `spec` and uses a proposal reason code;
- `action=recommend_stop|blocked` requires `spec=null`;
- every supporting event must have been present in `PlannerContext`;
- stop and blocked outputs are advisory only; Person 2 owns deterministic stop enforcement.

## 4. Owned repository paths

```text
src/tacorank/research/
  graph_view.py
  search_policy.py
  duplicate_detection.py
  portfolio.py
  method_cards.py
  plan_validation.py

src/tacorank/agents/
  research_planner.py

src/tacorank/providers/
  research_provider.py

research/methods/
  *.md

tests/research/
  test_graph_view.py
  test_search_policy.py
  test_duplicate_detection.py
  test_plan_validation.py
  test_research_planner.py
```

Do not edit Person 2's schemas, event store, context builder, or orchestrator without an agreed interface change. Do not edit candidate code under `solution/`; Person 3/Trae does that.

## 5. What “AIDE-style” means in this project

Implement the useful AIDE principles, not a full reproduction:

- every experiment has one verified parent;
- the baseline is the root;
- a child starts from its parent's Git commit;
- failed or rejected children cannot corrupt their parent;
- planning can return to an older accepted node;
- the current best is preserved while other branches are explored;
- the next hypothesis is chosen using accumulated evidence rather than only the latest result.

Git is the code tree. `PlannerContext.eligible_frontier` is the verified metadata view. Do not create a second persistent graph database.

## 6. Search policy

The three-day core must not depend on LinUCB, Thompson sampling, learned rewards, or a search simulator. There will be too few trustworthy experiments for a contextual bandit to be reliable. Implement a deterministic two-phase beam policy; leave UCB as an optional ablation after the complete system works.

### 6.1 Parent eligibility

A node is eligible only when:

- it is the verified baseline root; or
- it completed full public validation;
- Person 5's trust verdict is `accepted`;
- integrity is `clean`;
- `experiment.decided.parent_eligible` is true;
- its Git commit still exists and matches the context;
- it has not been retracted.

Proxy-only, no-op, suspicious, invalid, pruned, unstable, and hidden-final nodes are never parents.

### 6.2 Phase A: breadth

Before deep exploitation, ensure that high-value families receive one legal probe where budget permits:

1. objective alignment;
2. temporal/history signal;
3. model family;
4. one of multitask or duration-bias modelling when permitted by the data contract.

Each probe normally branches from the baseline or the best compatible frame. A crude proxy failure lowers priority but does not permanently falsify the family unless the implementation was verified and the method's falsification condition was actually met.

### 6.3 Phase B: evidence-guided depth

Maintain a frontier of at most three eligible nodes. Select parent and family using this deterministic ordering:

1. force an untried high-priority family while breadth is incomplete;
2. otherwise sort by trusted primary score descending;
3. prefer a family different from the previous two proposed experiments;
4. prefer nodes with fewer explored children;
5. prefer lower estimated cost when score and novelty are tied;
6. use experiment ID ascending as the final deterministic tie-break.

After two consecutive proposals in one family, the next proposal must use another legal family unless every alternative is blocked by prerequisites or budget.

### 6.4 Confirmation and finalization pressure

- Person 2 and Person 5 manage seed confirmations inside the same experiment. Do not propose a seed-only confirmation as a new experiment. A single-seed result is not parent-eligible and therefore will not enter your eligible frontier until confirmed.
- When convergence pressure equals two non-improving full evaluations, select a genuinely different family, not another minor hyperparameter variant.
- Near the wall-clock/query budget, stop starting high-cost branches and prefer a low-cost ensemble of already confirmed complementary nodes.
- Never recommend stopping simply to reduce resource use before at least one meaningful full experiment has completed.

## 7. Research portfolio

Seed these method cards before the run. Each card must follow the shared Markdown method-card schema.

### 7.1 Root baseline

- Reproduce the shipped official baseline exactly.
- Baseline parity is owned by Person 5 and blocks planning.
- Do not “improve” or refactor the baseline until parity passes.

### 7.2 Ranking-aligned objective — first meaningful candidate

Hypothesis: retaining the stable baseline representation while changing from pointwise loss to within-user pairwise BPR or grouped ranking will better align training with the evaluator's within-user ordering.

Required safeguards:

- positive/negative pairs only within a user;
- skip groups lacking either class;
- deterministic sampling seed and pair cap;
- one score for every original evaluation row;
- no changes to evaluator or split logic.

### 7.3 Temporal/history information

- use only interactions strictly earlier than the target row;
- deterministic sorting, truncation, padding, and cold-start fallback;
- no validation/test future information;
- start with a compact history before attention-heavy designs.

### 7.4 Model family

- compact tree ranker or DeepFM-style candidate;
- change model family only after the objective/data frame is verified;
- record CPU/GPU cost estimate;
- avoid broad capacity sweeps without a mechanism.

### 7.5 Multitask supervision

- auxiliary engagement labels are training targets only when legal;
- never pass target-like columns as validation/test inputs;
- begin with one strong auxiliary signal and fixed weight;
- record how the primary head is used at inference.

### 7.6 Duration-bias/censored-watch-time method

- use only after the frozen contract confirms available legal features;
- document which CWM mechanism is adopted;
- never copy a pipeline whose splits or evaluator differ from the competition;
- keep a simpler direct-ranking control.

### 7.7 Ensemble

- only confirmed, clean candidates;
- rank-average rather than assume score calibration compatibility;
- use at most two or three members;
- ensemble must be a separately evaluated node with exact member commits.

## 8. Research Planner behavior

### 8.1 Deterministic policy before LLM call

`SearchPolicy` chooses:

- eligible parent;
- preferred family;
- maximum cost tier;
- whether the action is probe, improvement, confirmation, or ensemble.

The LLM fills in the bounded technical hypothesis. It cannot override parent eligibility, contract restrictions, target paths, remaining budget, or family forcing.

### 8.2 Prompt contents

The Planner prompt must contain:

- resolved contract summary and hash;
- selected parent and current best;
- applicable method cards;
- verified results from the same family;
- active lessons and prohibitions;
- remaining experiment, query, wall-time, token, and GPU budget;
- explicit `ExperimentSpec` output schema;
- instruction to cite evidence event IDs;
- instruction to change one main mechanism;
- instruction to state how the hypothesis can be falsified.

It must not contain hidden results, raw unrelated logs, invalid feedback as reward, secrets, or the entire ledger.

### 8.3 Provider contract

`ResearchProvider.generate()` returns JSON matching `ExperimentSpec` and nothing else. Bound:

- one primary call;
- one format-repair call only for invalid JSON/schema;
- configured input/output token limits;
- deterministic temperature suitable for structured planning;
- provider/model identity and token usage returned alongside the result.

Do not let provider retry silently create several unlogged hypotheses.

### 8.4 Post-generation validation

`PlanValidator` verifies:

- IDs and context identity;
- parent eligibility and Git SHA match;
- legal family and fidelity sequence;
- normalized editable target files;
- cited evidence exists in the supplied context;
- method cards exist and are not forbidden;
- cost fits budget;
- duplicate key is correct and new;
- hypothesis, mechanism, and falsifier are non-empty and distinct;
- no hidden-test request, protected-file edit, raw command, or external training data.

Invalid plans are never forwarded to Person 3.

## 9. Duplicate detection

Normalize:

```text
parent_commit_sha
family
lowercase(change_summary)
sorted(target_files)
normalized method IDs
```

Remove whitespace and punctuation that do not change meaning, then compute SHA-256. This is `duplicate_key`.

Exact duplicates under the same parent are rejected. Confirmation runs never pass through the planner and therefore never need a special duplicate-key exception.

Semantic near-duplicate detection may use deterministic tags and method IDs. Do not add an embedding/vector store.

## 10. Person 1 event content

You do not append events. Return content that Person 2 records as:

- `context.created` — created by Person 2 before your call;
- `experiment.proposed` — your validated `ExperimentSpec`;
- provider usage in that event's `resource_delta`;
- optional advisory stop/block recommendation recorded as a planner decision event only if Person 2's schema includes it.

Every proposal must cite its `context_id`, `method_card_ids`, and `evidence_event_ids`.

## 11. Integration sequence

```python
planner_context = context_builder.build_planner_context(run_state)
policy_choice = search_policy.choose(planner_context)

spec, provider_usage = research_planner.propose(
    planner_context,
    policy_choice,
)

plan_validator.validate(spec, planner_context, policy_choice)

# Person 2, not Person 1:
event_store.append_experiment_proposed(spec, provider_usage)
```

Person 1 never calls Trae. Person 2 converts the accepted spec into a Coder context and calls Person 3.

## 12. Implementation plan

### P0 — interface-compatible deterministic core

1. Import shared models from `src/tacorank/schemas.py`.
2. Implement `GraphView` using only `PlannerContext` summaries.
3. Implement parent eligibility.
4. Implement breadth tracking by family.
5. Implement deterministic beam selection and tie-breaking.
6. Implement method-card parser and validation.
7. Implement duplicate-key generation.
8. Implement a deterministic fake Planner producing a valid ranking-objective spec.
9. Implement `PlanValidator`.
10. Pass integration with Person 2's fake harness.

### P1 — real research provider

11. Add the real structured LLM provider.
12. Add one bounded format-repair attempt.
13. Capture provider/model/token usage.
14. Seed the method-card portfolio.
15. Add family-aware lesson use.
16. Demonstrate a branch from an older accepted node.

### P2 — only after the full loop works

17. Confirmation scheduling.
18. Ensemble planning.
19. Optional UCB-style selection as an ablation behind a feature flag.
20. Search-policy comparison report; never block the submission on it.

## 13. Required tests

### Unit tests

- baseline is the only eligible parent before accepted experiments;
- rejected, proxy-only, suspicious, invalid, or retracted nodes are excluded;
- breadth phase tries different families;
- two same-family proposals force diversity;
- deterministic tie-breaking is stable;
- duplicate keys are stable and reject exact repeats;
- cost above budget is rejected;
- evidence IDs outside context are rejected;
- protected paths and external-data hypotheses are rejected;
- forbidden method cards are rejected.

### Contract tests

- consume Person 2's golden `PlannerContext` fixture;
- emit JSON accepted by the shared `PlannerOutput` model and its nested `ExperimentSpec`;
- reject `action=propose` with a null spec and advisory actions with a non-null spec;
- unknown fields fail;
- wrong schema/context/run/parent identities fail;
- every proposal includes mechanism, falsifier, evidence, and method source.

### Integration tests

- baseline context → valid first full-stack hypothesis;
- accepted result returns in next context and changes the selected parent;
- negative result prevents exact repetition;
- older accepted node can be selected after the latest branch fails;
- convergence pressure changes family choice;
- mock provider malformed output receives exactly one repair attempt.

## 14. Definition of done

- A valid `PlannerOutput(action=propose)` containing an evidence-citing `ExperimentSpec` is produced from the golden context.
- Search returns to an older trusted node in a deterministic test.
- Invalid or suspicious results never influence parent selection as positive evidence.
- The first meaningful candidate targets ranking alignment rather than random tuning.
- The Planner never writes files outside owned research paths or candidate code.
- Provider tokens and failures are visible to Person 2.
- No hidden-test data, metric, or artifact enters planning.
- The implementation works with fake providers and without a network.

## 15. Handoff checklist

Give Person 2:

- `ResearchPlanner` implementation;
- `SearchPolicy` and deterministic configuration;
- valid/invalid `PlannerOutput` and `ExperimentSpec` fixtures;
- method cards and parser;
- provider configuration template without credentials;
- duplicate-key function and tests;
- a one-page note documenting any intentionally deferred search features.

Do not hand over a runnable script that bypasses the shared orchestrator. The accepted integration surface is `PlannerContext → PlannerOutput` only.
