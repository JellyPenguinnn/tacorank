# Person 5 — Evaluation, Trust, Experiment Decisions and Research Reflection

## Codex implementation brief

You own metric truth. Your component accepts only structurally verified predictions, calls the protected evaluator, compares baseline/parent/best, distinguishes genuine research results from noise, no-op wiring and suspicious behavior, recommends promotion or terminal decisions, creates evidence-backed research lessons, verifies baseline parity, supports final selection evidence, and produces judge-facing evaluation summaries.

Metric adjudication must be deterministic. An LLM may optionally word a reflection after the verdict is fixed, but it must never calculate metrics, set trust status, change acceptance, or select a model.

## 1. Responsibility boundary

You own:

- protected evaluator adapter and metric parity tests;
- internal proxy and public-validation evaluation;
- baseline/parent/best deltas;
- prediction-change/no-op detection;
- trust verdict, stability and integrity flags;
- proxy-to-full promotion recommendation;
- full accept/reject/prune/invalid recommendation;
- multi-seed confirmation logic;
- research `LessonCandidate` creation;
- final evaluation/results/resource evidence and submission-readiness reporting.

You do not:

- modify candidate code or evaluator;
- run Trae or sandbox processes;
- validate patch safety or prediction row structure—Person 3 Gate A/B;
- recover code/infra failures—Person 4;
- choose the next research hypothesis—Person 1;
- append memory, update Git best ref, enforce stop or expose hidden results—Person 2.

## 2. Contract rule and official discrepancy

The supplied competition document contains contradictory task descriptions. Never hard-code target label, metrics, K values, baseline numbers or primary aggregation inside generic evaluator/trust logic.

At run start, Person 2 provides a frozen `ContractSummary` and hashes. Your adapter must:

- load the protected official evaluator selected by that contract;
- validate its hash before every call;
- validate metric names and primary aggregation against the contract;
- refuse evaluation on contract/hash mismatch;
- preserve the contradiction/resolution in reporting;
- keep hidden test outside iterative feedback.

## 3. Owned paths

```text
src/tacorank/evaluation/
  adapter.py
  baseline.py
  proxy.py
  metrics.py
  comparisons.py
  no_op.py
  trust.py
  stability.py
  decisions.py
  final_selection.py

src/tacorank/reflection/
  research.py
  lesson_rules.py

src/tacorank/reporting/
  results.py
  resources.py
  experiment_tree.py
  charts.py

benchmarks/kuairand_pure/
  evaluator_adapter.py
  submission_adapter.py

tests/evaluation/
tests/reflection/
tests/reporting/
```

Import shared schemas from Person 2. Do not create alternative metric/verdict/decision enums.

## 4. Shared interfaces

### 4.1 Input: `EvaluationRequest`

```text
run_id, experiment_id, attempt
output_checked_event_id
prediction_artifact: ArtifactRef
ordered_row_identity_sha256, ordered_prediction_sha256 from verified Gate B
population: internal_proxy | public_validation | hidden_final
fidelity: proxy | full | final
seed
contract_sha256
evaluator_sha256
baseline_summary
parent_summary
previous_best_summary
public_query_index: int | null
```

Precondition: Person 3's `OutputCheckResult.accepted` is true and Person 2 has appended a verified `output.checked` event. Reject any request lacking that evidence.

### 4.2 Output: `EvaluationResult`

```text
run_id, experiment_id, attempt
population, fidelity, seed, public_query_index
evaluator_sha256, contract_sha256
metric_set: MetricSet
baseline_delta, parent_delta, previous_best_delta
prediction_change: PredictionChange
trust: TrustAssessment
seed_evidence_event_ids: verified prior evaluation events for this experiment
```

`MetricSet`:

```text
metrics: dict[str, finite float]
primary_metric_name: str
primary_score: finite float
```

`PredictionChange`:

```text
spearman_vs_parent: float | null
changed_row_fraction: float | null
identical_score_fraction: float | null
unique_score_fraction: float
```

`TrustAssessment`:

```text
verdict: accepted | inconclusive | negative | no_op | suspicious
stability: single_seed | confirmed | unstable | not_applicable
integrity: clean | compromised | inconclusive
flags: list[str]
```

### 4.3 Output: `ExperimentDecision`

```text
run_id, experiment_id
evaluation_event_id: str | null
decision: promote | accept | reject | prune | invalid
reason_code
fidelity_completed
parent_eligible, best_eligible
next_fidelity: proxy | full | null
supporting_event_ids
lesson_candidate: LessonCandidate | null
```

Person 2 appends the evaluation first, then supplies its event ID to your decision method.

### 4.4 Research `LessonCandidate`

```text
origin: research
category: research_result | integrity_warning | process_rule
tags
summary
applicability
avoid_when
confidence in [0,1]
source_event_ids
source_commit_shas
```

Person 2 validates/deduplicates and appends it. You never write `LESSONS.md`.

## 5. Protected evaluator adapter

### 5.1 Isolation

- evaluator resides outside agent-editable roots or is read-only;
- validate SHA-256 before every call;
- receive predictions only after Gate B;
- retrieve labels through a protected evaluator-only data path;
- candidate process never sees protected validation labels;
- call official functions directly where possible;
- execute the official call in a clean isolated interpreter with candidate paths removed;
- never reimplement a “faster equivalent” as the production scoring path;
- any independent metric implementation is a parity test only.

### 5.2 Adapter output

Return all contract-required metrics and the primary score. Preserve raw precision in memory; round only in presentation.

Verify:

- required metric names present;
- finite values;
- ranges if declared by contract;
- primary aggregation reproduces within tolerance;
- population/user IDs align with protected data manifest;
- evaluator/hash/contract identities are recorded.

### 5.3 Metric-agnostic code

Use configuration such as:

```text
required_metrics: list[str]
primary_formula: configured callable/spec
epsilon: float
convergence_patience: int
baseline_metrics: dict[str, float]
```

Dataset-specific adapters may parse the official evaluator, but shared trust logic uses `MetricSet`.

## 6. P0 baseline parity gate

Before autonomous planning:

1. receive Person 3's Gate-B-accepted baseline predictions;
2. run official evaluator;
3. compare every official metric and primary to published baseline under declared tolerance;
4. verify at least the configured seed behavior;
5. record evaluator, contract, data, prediction and commit hashes;
6. return parity outcome to Person 2.

If parity fails, research must stop. Do not “calibrate” thresholds to accept a mismatched implementation.

Also implement an independent small evaluator in tests and compare it with the official evaluator on synthetic fixtures. This catches wrapper/alignment bugs; it does not replace the official scorer.

## 7. Evaluation populations and fidelity

### 7.1 Smoke

No official metric is required. Person 2 may promote a structurally valid smoke result deterministically. Your component is not called unless a diagnostic score is explicitly configured.

### 7.2 Internal proxy

- use legal training-period data only;
- maintain temporal ordering;
- construct train/holdout split deterministically;
- approximate evaluation group geometry when feasible;
- label result as proxy;
- never compare proxy score numerically as if it were full public validation;
- proxy cannot update trusted best or convergence.

Proxy is for pruning/promotion only.

### 7.3 Public validation

- use frozen official validation population;
- increment/query index supplied by Person 2;
- score with official evaluator;
- result affects trust, parent eligibility, best eligibility and convergence;
- candidate code receives predictions but not labels/metric internals.

### 7.4 Hidden final

Legal only after verified `run.stopped` and final selection. Hidden labels/metrics:

- never appear in Planner/Coder/Recovery contexts;
- never update lessons used in the same run;
- never cause another experiment;
- are used only for final reporting after organizer evaluation.

## 8. Comparisons

For each evaluation compute:

- absolute delta versus official baseline;
- delta versus selected parent under the same population/fidelity/seed policy;
- delta versus previous trusted best;
- each raw metric delta;
- optional normalized headroom only for reporting when a defensible ceiling is frozen.

Never compare proxy and full as if they share a scale. Never compare different contracts or evaluator hashes.

## 9. NO_OP detection

Goal: distinguish “the research idea failed” from “the patch did not affect predictions.”

Compute against parent predictions for the same population:

- Spearman rank correlation;
- changed-row fraction under a numeric tolerance;
- identical-score fraction;
- unique-score fraction;
- primary delta relative to expected noise.

Default configurable rule:

```text
no_op if
  spearman_vs_parent >= 0.9999
  and changed_row_fraction <= 0.001
  and abs(parent_delta) <= trust_epsilon
```

Do not hard-code thresholds in logic; expose them in frozen evaluation config and validate on baseline/self-comparison fixtures.

If `no_op`:

- return verdict `no_op`;
- do not create a research-negative lesson;
- Person 2 records evaluation but delays terminal decision;
- Person 2 asks Person 4 for silent implementation recovery;
- after successful wiring repair, reevaluate;
- after repeated no-op, Person 4 may create an implementation constraint.

## 10. Trust assessment

Trust is deterministic and ordered. Apply integrity checks before improvement checks.

### 10.1 Integrity flags

At minimum:

```text
EVALUATOR_HASH_MISMATCH
CONTRACT_HASH_MISMATCH
OUTPUT_GATE_EVIDENCE_MISSING
FORBIDDEN_INPUT_DETECTED
TOO_GOOD_TO_BE_TRUE
METRIC_DIRECTION_CONFLICT
DEGENERATE_SCORES
PREDICTION_ALIGNMENT_SUSPECT
PROXY_FULL_SIGN_CONFLICT
SEED_INSTABILITY
```

`TOO_GOOD_TO_BE_TRUE` threshold must be configured using baseline/noise/domain evidence. It triggers investigation; it does not automatically prove compromise.

### 10.2 Verdict order

1. Hash/evidence/forbidden-data failure → `suspicious`, integrity `compromised` or `inconclusive`.
2. NO_OP rule → `no_op`.
3. Non-finite/invalid metric should already be blocked; treat as suspicious integrity failure if reached.
4. Promising but below trust threshold → `inconclusive`.
5. Verified primary delta at or below zero → `negative`.
6. Verified delta beyond threshold with clean integrity → `accepted` or `single_seed` pending confirmation, according to configured policy.

### 10.3 Trust threshold

Use the frozen contract's convergence epsilon as the minimum practical full improvement unless the contract/evaluation plan specifies a stronger noise-derived threshold.

Do not accept the luckiest seed. For promising candidates:

- request configured confirmation seeds when budget permits;
- aggregate mean and standard deviation;
- ensure required raw metrics do not reveal a prohibited trade-off;
- stability `confirmed` only when the configured rule passes;
- unstable candidates cannot become trusted best.

### 10.4 Metric direction

Always return all raw metric deltas. If one improves and another worsens:

- set `METRIC_DIRECTION_CONFLICT` when contract policy requires it;
- allow primary-based acceptance only if the frozen rule permits and integrity remains clean;
- create a lopsided research reflection rather than hiding the trade-off.

## 11. Decision policy

### 11.1 Proxy

Return `promote` to full when:

- output/evaluator integrity clean;
- implementation not no-op;
- proxy parent delta exceeds configured proxy threshold or shows an explicitly allowed complementary effect;
- full-query and resource budgets remain.

Otherwise return `prune` or `reject` with reason.

Proxy decisions always set `parent_eligible = false`, `best_eligible = false`.

### 11.2 Full public validation

| Verdict | Decision | Parent eligible | Best eligible |
| --- | --- | ---: | ---: |
| accepted + confirmed + clean | `accept` | Yes | Yes only if better than trusted best |
| accepted + single seed, confirmation required | `promote`, `next_fidelity=full`, reason `CONFIRMATION_REQUIRED` | No until confirmed | No |
| inconclusive | `reject` or request confirmation when strategically configured | No | No |
| negative | `reject` | No | No |
| suspicious | `invalid`/`prune` | No | No |
| no_op | Delay terminal decision; route to Person 4 | No | No |

Person 2 expresses confirmation as another full execution/evaluation attempt of the same commit and configuration using the next seed from the frozen seed schedule. It remains the same `experiment_id`, increments the execution attempt, and is fully logged and budgeted. Person 1 is not called. Do not create an internal seed loop inside the evaluator. If the configured confirmation budget is exhausted, return a terminal `reject` with an inconclusive reason; never request another confirmation.

### 11.3 Best eligibility

Requires:

- full public validation;
- exact contract/evaluator hashes;
- accepted, confirmed and clean result;
- reproducible commit/config/seed policy;
- primary better than previous trusted best under frozen comparison;
- no retraction/integrity flag.

## 12. Research reflection

### 12.1 Triggers

Create a research `LessonCandidate` when:

- a confirmed accepted result reveals a reusable mechanism;
- a verified negative result genuinely falsifies the stated hypothesis;
- a metric conflict reveals a reusable trade-off;
- a suspicious result reveals an integrity warning;
- a method result becomes stale under a changed research frame, when supported by events.

Do not create a research lesson for:

- syntax/import/infra failure;
- unverified implementation;
- first no-op result;
- proxy-only weak failure presented as permanent falsification;
- tiny noise-level delta without confirmation.

### 12.2 Structure

A good reflection separates:

1. observation — measured fact;
2. causal hypothesis — explicitly labelled inference;
3. reusable lesson;
4. applicability;
5. avoid condition;
6. recommended research consequence;
7. confidence and evidence IDs.

Example:

```text
Observation: The confirmed pairwise objective improved both contract metrics
relative to the pointwise parent across configured seeds.
Causal hypothesis: Within-user pair construction better matches the evaluator's
ranking geometry.
Applicability: Groups containing both positive and negative examples.
Avoid: Cross-user pairs or single-class groups.
Confidence: 0.90.
```

### 12.3 Optional LLM wording

If an LLM is used to word reflections:

- supply fixed metrics/verdict/evidence;
- use a strict `LessonCandidate` schema;
- one bounded call;
- the model cannot change numerical fields, verdict or confidence ceiling;
- deterministic validation before Person 2 sees the candidate;
- token usage returned in `ResourceDelta`.

Prefer deterministic templates for the three-day core.

## 13. Staleness

When a new accepted experiment changes the research frame—such as objective/group construction—older content-feature negatives may no longer apply.

Do not edit old lessons. Return a recommendation identifying lesson IDs and evidence for Person 2 to append `lesson.status_changed(new_status=stale)`.

Only mark stale when the causal dependency is explicit; do not use staleness to erase inconvenient negative evidence.

## 14. Baseline, seed and reproducibility policy

- baseline parity before research;
- seed identity included in every request/result;
- verify seed actually changes all intended randomness;
- preserve mean/std and individual seed metrics;
- clean final reproduction uses exact commit/config/data/evaluator identities;
- a nonreproducible best becomes unstable/ineligible, not silently accepted.

## 15. Final selection and reporting

Person 2 selects the latest trusted best deterministically. You verify evidence and produce:

- baseline, parent, best and final metric table;
- per-metric and primary deltas;
- seed/stability evidence;
- experiment verdict distribution;
- public-validation query count;
- final clean-reproduction result;
- token, Agent wall-clock, CPU/GPU and intervention summary from Person 2's event projection;
- failure/recovery statistics supplied by events;
- run limitations and contract discrepancy resolution.

Recommended static charts, only after core evaluation works:

1. best primary score versus full evaluation index;
2. experiment tree colored by verdict;
3. per-metric deltas for accepted nodes;
4. verdict distribution including no-op/suspicious/invalid;
5. token/time/GPU usage by role;
6. recovery attempts and successes.

Do not spend core implementation time on an interactive dashboard.

## 16. Person 2 integration

```python
request = orchestrator.build_evaluation_request(output_check)
result = await evaluator.evaluate(request)

# Person 2 appends evaluation.completed and obtains its event ID.

if result.trust.verdict == NO_OP:
    # Person 2 routes to Person 4 before terminal decision.
    return result

decision = await evaluator.decide(
    result,
    decision_context_with_evaluation_event_id,
)

# Person 2 validates/appends experiment.decided,
# lesson.recorded and best.updated when eligible.
```

Never append events or update Git refs yourself.

## 17. Implementation order

### P0 — metric truth first

1. Import shared schemas/fixtures.
2. Implement protected evaluator hash check and adapter.
3. Implement official/independent synthetic parity tests.
4. Integrate Person 3 Gate-B-accepted baseline predictions.
5. Reproduce baseline within tolerance.
6. Implement baseline/parent/best comparisons.

### P1 — trust and decisions

7. Implement proxy/public population routing.
8. Implement prediction-change and no-op detection.
9. Implement integrity flags and deterministic verdict order.
10. Implement proxy promotion/full decisions.
11. Implement multi-seed aggregation/confirmation.
12. Integrate Person 2 event/request flow.

### P2 — reflection and final evidence

13. Implement research LessonCandidate rules/templates.
14. Implement staleness recommendations.
15. Implement final reproduction evidence checks.
16. Generate results/resource/recovery tables and static charts.

## 18. Required tests

### Evaluator/parity

- official evaluator hash mismatch;
- synthetic metric calculations agree with official evaluator;
- published baseline within tolerance;
- primary aggregation exact;
- non-finite/missing/extra metrics rejected;
- wrong population/data manifest rejected.

### NO_OP/trust

- identical predictions → no-op;
- tiny harmless numeric differences under tolerance → no-op;
- high correlation but meaningful ranking changes not falsely no-op;
- positive below threshold → inconclusive;
- verified negative → negative;
- large clean improvement → accepted/confirmation required;
- too-good/hash/forbidden evidence → suspicious;
- metric direction conflict flag;
- unstable seeds not best eligible.

### Decisions

- smoke not handled as official evaluation;
- proxy can promote but never become parent/best;
- full confirmed accepted can become parent;
- accepted but worse than best is parent eligible but not best eligible;
- suspicious/no-op/negative never eligible;
- no-op produces no terminal decision before Person 4 diagnosis;
- hidden-final cannot promote/accept/propose.

### Reflection

- confirmed positive creates reusable lesson;
- verified negative can create falsification lesson;
- proxy failure does not create permanent falsification;
- first no-op produces no research lesson;
- syntax/infra failure rejected as research reflection input;
- metric conflict reflection preserves both metrics;
- stale lesson recommendation requires explicit frame evidence.

### Integration

- baseline pipeline through Person 3 Gate B;
- proxy evaluation → promote;
- full evaluation → accept → Person 2 best update;
- no-op → Person 4 repair → reevaluation;
- suspicious result cannot reach Person 1 as reward;
- final clean reproduction/report generation.

## 19. Definition of done

- Official baseline parity blocks/allows research correctly.
- Every score is tied to prediction, evaluator, contract, data, seed and commit hashes.
- Proxy/full/hidden populations cannot be confused.
- No-op cannot poison research memory as a false negative.
- Suspicious or unstable results cannot become parents/best.
- Multi-seed confirmation is explicit and budget-visible.
- Research reflection is sparse, typed, evidence-linked and separate from operational reflection.
- Final results and resource summary are reproducible from the event ledger.

## 20. Handoff checklist

Give Person 2:

- real and fake evaluator adapters;
- baseline parity command/test;
- proxy split/evaluation configuration;
- trust/no-op/decision configuration;
- valid/invalid `EvaluationRequest`, `EvaluationResult`, `ExperimentDecision` and LessonCandidate fixtures;
- seed confirmation rules;
- report-generation command;
- list of contract-specific assumptions and frozen hashes.

Your accepted integration surfaces are `EvaluationRequest → EvaluationResult`, `EvaluationResult + decision context → ExperimentDecision`, and verified results → optional research `LessonCandidate`.
