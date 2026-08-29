# Person 2 — Agent Harness, Memory, Context Builder and Orchestration

## Codex implementation brief

You own the deterministic control plane that turns five independently developed components into one reliable autonomous research agent. Your work is the integration backbone: shared schemas, append-only memory, state reconstruction, context construction, outer-loop routing, token/resource budgets, convergence, restart, and CLI.

You are the **only writer** of dynamic memory. Other components return typed values; you validate and append them. You do not invent research hypotheses, edit candidate code, waive safety gates, classify GPU failures, compute metrics, or write research conclusions.

## 1. Product objective and contract rule

The system must:

1. freeze and verify the official competition contract;
2. reproduce the official baseline;
3. repeatedly plan, code, verify, execute, recover, evaluate, and learn;
4. preserve every hypothesis, diff, metric, error, recovery, token/GPU cost, and intervention;
5. stop under the frozen convergence or resource rule;
6. select the trusted validation-best candidate;
7. expose hidden test once during finalization only.

The supplied problem material contains conflicting label/metric descriptions. Your harness must be metric-agnostic and refuse to start until humans resolve the conflict in `contract/COMPETITION.md`. No component may hard-code one interpretation independently.

## 2. Your ownership boundary

You own:

- `src/tacorank/schemas.py` and all shared enums/models;
- event validation, canonical JSON serialization, append, hash chain, and replay;
- derived `RunState`, experiment graph view, active lessons, status and summary views;
- exact context bundles for Planner, Coder, Recovery, and evaluation handoffs;
- the outer deterministic state machine;
- adapter protocols for Persons 1, 3, 4, and 5;
- idempotency and resume;
- token/resource budget accounting;
- convergence and final-selection routing;
- CLI entry points;
- integration fixtures and tests.

You do not own:

- parent/hypothesis quality — Person 1;
- Trae prompts, Git worktrees, sandbox, Gate A/B mechanisms — Person 3;
- health/failure classification, repair policy, operational reflection — Person 4;
- metric truth, trust verdicts, promotion/acceptance recommendation, research reflection — Person 5.

## 3. Canonical memory authorities

There are exactly three:

| Authority | Store | Purpose |
| --- | --- | --- |
| Human contract | Markdown | Rules, data boundary, metrics, budgets, protected paths |
| Dynamic evidence | `events.jsonl` | Complete typed append-only run history |
| Code lineage | Git | Exact code, diffs, experiment ancestry and best pointer |

Derived files—`STATUS.md`, `LESSONS.md`, `SUMMARY.md`, context files, dashboard data—are not sources of truth.

Do not add SQLite, Redis, a vector database, a mutable `state.json`, an experiment table, or a separate reflection database.

## 4. Owned repository paths

```text
src/tacorank/
  schemas.py
  config.py
  artifacts.py
  accounting.py
  cli.py

src/tacorank/memory/
  event_store.py
  canonical_json.py
  replay.py
  projections.py
  retrieval.py

src/tacorank/orchestrator/
  ports.py
  state.py
  state_machine.py
  router.py
  convergence.py
  finalize.py

src/tacorank/context/
  builder.py
  redaction.py
  token_estimator.py
  templates.py

runs/<run_id>/
  events.jsonl
  STATUS.md
  LESSONS.md
  SUMMARY.md
  contexts/*.md

tests/schemas/
tests/memory/
tests/orchestrator/
tests/context/
tests/integration/
```

## 5. Shared schema freeze

Implement strict Pydantic v2 models with `extra="forbid"`, finite-number validation, normalized relative paths, enum validation, and schema version `1.0`.

### 5.1 Common sub-models

#### `ArtifactRef`

```text
artifact_id: str
kind: diff | trajectory | context | log | checkpoint | predictions |
      metrics | delta_vector | verification_receipt | submission | report | other
path: str                              # normalized repository-relative path
sha256: str                            # lowercase 64 hex
size_bytes: int >= 0
content_type: str | null
```

Reject absolute paths, `..`, symlinks, paths outside approved artifact roots, missing bytes, or hash mismatch.

#### `ResourceDelta`

```text
llm_input_tokens: int >= 0
llm_output_tokens: int >= 0
token_measurement: provider | estimated | none
wall_time_ms: int >= 0
cpu_time_ms: int >= 0
gpu_time_ms: int >= 0
gpu_count: int >= 0
peak_rss_mb: int | null
peak_gpu_memory_mb: int | null
manual_interventions: int >= 0
```

Provider and estimated tokens must be aggregated separately. GPU-hours are derived as `sum(gpu_time_ms × gpu_count) / 3_600_000`. Agent wall-clock is elapsed time from `run.started` to `run.stopped`, not the sum of action durations.

#### `MetricSet`

```text
metrics: dict[str, finite float]
primary_metric_name: str
primary_score: finite float
```

Metric names and aggregation are validated against the frozen contract.

#### `CostEstimate`

```text
llm_tokens_upper_bound: int >= 0
wall_time_seconds_upper_bound: int >= 0
gpu_seconds_upper_bound: int >= 0
cost_tier: low | medium | high
```

### 5.2 Cross-component models

#### `ExperimentSpec` — Person 1 → Person 2

```text
schema_version, run_id, experiment_id
parent_experiment_id, parent_commit_sha, context_id
hypothesis, family, change_summary, target_stage, target_files
fidelity_plan, expected_mechanism, success_criteria
falsification_condition, estimated_cost
method_card_ids, evidence_event_ids, duplicate_key
```

#### `PlannerOutput` — Person 1 → Person 2

```text
action: propose | recommend_stop | blocked
spec: ExperimentSpec | null
reason_code, reason
supporting_event_ids
```

Invariant: `action=propose` if and only if `spec` is non-null. Advisory stop/block outputs never terminate the run by themselves; the harness checks deterministic stop and budget rules.

#### `PatchCandidate` — Person 3 → Person 2

```text
schema_version, run_id, experiment_id, attempt
experiment_spec_event_id, context_id
base_commit_sha, patch_commit_sha, diff_sha256
changed_files
diff_artifact, trajectory_artifact
trae_version, model_id, steps_used
resource_delta
```

#### `PatchCheckResult` — Person 3 → Person 2

```text
run_id, experiment_id, attempt
patch_commit_sha, diff_sha256
accepted: bool
receipt_id: str | null
receipt_artifact: ArtifactRef | null
checks: list[CheckResult]
violations: list[Violation]
```

`CheckResult.status = pass | fail | not_applicable`.

#### `RunRequest` — Person 2 → Person 3

```text
run_id, experiment_id, attempt, fidelity
command_id, patch_commit_sha, patch_receipt_id
seed, data_manifest_sha256
timeout_seconds, memory_limit_mb, gpu_memory_limit_mb
network_enabled: bool
```

Only named `command_id` values from the protected contract are legal.

#### `TelemetrySample` — Person 3 → Person 4 during run

```text
timestamp, run_id, experiment_id, attempt
elapsed_ms, process_alive, last_output_age_ms
cpu_percent, rss_mb
gpu_utilization_percent, gpu_memory_mb
loss, gradient_norm, disk_free_mb
recent_output_tail
```

#### `MonitorDirective` — Person 4 → Person 3 during run

```text
action: continue | terminate
reason_code: str | null
summary: str | null
```

#### `RunResult` — Person 3 → Person 2

```text
run_id, experiment_id, attempt, fidelity
patch_commit_sha
outcome: success | code_error | interface_error | contract_error |
         numerical_error | oom | timeout | hang |
         infrastructure_error | cancelled
exit_code: int | null
error_class, error_fingerprint, error_summary
log_artifact, telemetry_artifact
checkpoint_artifact, prediction_artifact
resource_delta
```

#### `OutputCheckResult` — Person 3 → Person 2

```text
run_id, experiment_id, attempt
prediction_artifact
accepted: bool
checks: dict[str, pass | fail | not_applicable]
score_stats
violations
```

#### `RecoveryDecision` — Person 4 → Person 2

```text
run_id, experiment_id
failure_event_id
repair_attempt
action: trae_repair | retry_same_commit |
        adjust_approved_runtime_setting | rollback | abandon
reason_code, instructions
same_error_count, remaining_repair_budget
lesson_candidate: LessonCandidate | null
```

#### `EvaluationRequest` — Person 2 → Person 5

```text
run_id, experiment_id, attempt
output_checked_event_id, prediction_artifact
population: internal_proxy | public_validation | unbiased_audit | hidden_final
fidelity, seed
contract_sha256, evaluator_sha256
baseline_summary, parent_summary, previous_best_summary
public_query_index: int | null
```

#### `EvaluationResult` — Person 5 → Person 2

```text
run_id, experiment_id, attempt
population, fidelity, seed, public_query_index
evaluator_sha256, contract_sha256
metric_set
baseline_delta, parent_delta, previous_best_delta
prediction_change
trust: TrustAssessment
```

`TrustAssessment` contains:

```text
verdict: accepted | inconclusive | negative | no_op | suspicious | redundant
stability: single_seed | confirmed | unstable | not_applicable
integrity: clean | compromised | inconclusive
flags: list[str]
```

#### `ExperimentDecision` — Person 5 → Person 2

```text
run_id, experiment_id
evaluation_event_id: str | null
decision: promote | accept | reject | prune | invalid
reason_code
fidelity_completed
parent_eligible, best_eligible
next_fidelity: Fidelity | null
supporting_event_ids
lesson_candidate: LessonCandidate | null
```

#### `LessonCandidate` — Persons 4 or 5 → Person 2

```text
origin: operational | research
category: research_result | resource_constraint |
          implementation_constraint | integrity_warning | process_rule
tags, summary, applicability, avoid_when
confidence: float in [0,1]
source_event_ids, source_commit_shas
```

Person 2 allocates `lesson_id`, checks trigger eligibility/deduplication, and appends `lesson.recorded`.

## 6. Component protocols

Define in `orchestrator/ports.py`:

```python
class ResearchPlanner(Protocol):
    async def propose(self, context: PlannerContext) -> PlannerOutput: ...

class CodingWorker(Protocol):
    async def create_patch(
        self, context: CoderContext, spec: ExperimentSpec
    ) -> PatchCandidate: ...

    async def repair_patch(
        self, context: RecoveryContext, decision: RecoveryDecision
    ) -> PatchCandidate: ...

class PatchGate(Protocol):
    async def check(self, candidate: PatchCandidate) -> PatchCheckResult: ...

class ExecutionRunner(Protocol):
    async def run(
        self, request: RunRequest, observer: HealthObserver
    ) -> RunResult: ...

class HealthObserver(Protocol):
    def observe(self, sample: TelemetrySample) -> MonitorDirective: ...

class RecoveryManager(Protocol):
    async def decide(
        self,
        failure_event_id: str,
        result: RunResult | PatchCheckResult | OutputCheckResult | EvaluationResult,
        context: RecoveryPolicyContext,
    ) -> RecoveryDecision: ...

class OutputGate(Protocol):
    async def check(self, result: RunResult) -> OutputCheckResult: ...

class Evaluator(Protocol):
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...

    async def decide(
        self, result: EvaluationResult, context: EvaluationDecisionContext
    ) -> ExperimentDecision: ...
```

Provide deterministic fake implementations for all ports. The real orchestrator must pass end-to-end tests using only fakes before teammates' adapters are integrated.

## 7. Event ledger

Implement the complete schema in `TacoRank-Memory-Schema-v1.md`. Required properties:

- compact UTF-8 JSON, one object per newline;
- contiguous `seq` and `event_id = evt_{seq:06d}`;
- unique idempotency key;
- causal event reference;
- event-type discriminated payload;
- artifact refs and action-local resource delta;
- `prev_event_hash`/`event_hash` hash chain;
- file lock, flush, and `fsync`;
- complete lines immutable;
- only an incomplete crash tail may be truncated to the last newline.

### 7.1 Event types

```text
run.started
contract.verified
baseline.verified
context.created
planner.recommended
experiment.proposed
patch.created
patch.checked
execution.started
execution.finished
recovery.decided
output.checked
evaluation.completed
experiment.decided
best.updated
lesson.recorded
lesson.status_changed
manual.intervention
run.stopped
final.selected
submission.checked
```

### 7.2 Append transaction

```python
def append(payload_model, metadata) -> Event:
    acquire_exclusive_lock()
    state = replay_and_validate_complete_events()
    reject_duplicate_idempotency_key()
    validate_transition(state, payload_model)
    validate_artifact_hashes(payload_model)
    construct_next_event_and_hash()
    append_single_compact_line()
    flush_and_fsync()
    release_lock()
    return event
```

No component gets a raw file handle.

## 8. Derived state and projections

### 8.1 `RunState`

Derive:

```text
status, phase
active_experiment_id, active_attempt, active_fidelity
best_experiment_id, best_commit_sha, best_primary_score
experiments_proposed, full_evaluations_completed
public_validation_queries
consecutive_non_improving_full_evaluations
remaining budgets
provider/estimated token totals
CPU/GPU/resource totals
manual intervention count
```

Run states: `initializing`, `ready`, `running`, `stopped`, `finalizing`, `finalized`, `failed`.

### 8.2 `ExperimentNode`

Derive experiment identity, parent, hypothesis, family, base/latest commit, attempt count, highest fidelity, status, metrics, trust, eligibility and terminal event. Never persist an independently mutable graph. Verify Git ancestry during replay.

### 8.3 Lessons and views

Canonical lesson content is `lesson.recorded`; status changes are later events. Generate `LESSONS.md` from active lessons. Generate `STATUS.md` and `SUMMARY.md` entirely from events plus Git/artifact metadata.

## 9. State machine

Per experiment:

```text
proposed
  → patch_ready
  → ready_to_run
  → running
  → output_ready
  → output_verified
  → evaluated
  → accepted | rejected | pruned | invalid
```

Recovery path:

```text
patch_ready/running/output_ready/evaluated-no-op
  → recovering
  → patch_ready | ready_to_run | invalid
```

Only allow transitions defined in the memory schema. A successful smoke `output.checked` can promote directly to proxy without a metric. Proxy/full require Person 5 evaluation before decision.

## 10. Outer-loop orchestration

### 10.1 Bootstrap

```python
async def bootstrap(config):
    append_run_started(config)
    verify_contract_and_protected_hashes()
    append_contract_verified()

    baseline_result = await run_baseline_through_person_3()
    output_check = await output_gate.check(baseline_result)
    evaluation = await evaluator.evaluate(baseline_request)
    assert evaluator.baseline_parity_passed(evaluation)
    append_baseline_verified(evaluation)
```

Research cannot start before baseline parity.

### 10.2 One experiment

```python
async def run_one_experiment(state):
    planner_ctx = context_builder.build_planner(state)
    append_context_created(planner_ctx)
    planner_output = await planner.propose(planner_ctx)
    if planner_output.action != PROPOSE:
        append_planner_recommendation(planner_output)
        if deterministic_stop_condition_matches(planner_output):
            stop_with_verified_reason()
        else:
            run_one_bounded_deterministic_fallback_or_stop_no_legal_action()
        return
    spec = planner_output.spec
    append_experiment_proposed(spec)

    coder_ctx = context_builder.build_coder(spec)
    append_context_created(coder_ctx)
    patch = await coding_worker.create_patch(coder_ctx, spec)
    append_patch_created(patch)

    while True:
        patch_check = await patch_gate.check(patch)
        append_patch_checked(patch_check)
        if not patch_check.accepted:
            if not await recover(patch_check):
                append_terminal_invalid()
                return
            patch = repaired_patch
            continue

        stage_queue = validated_stage_queue(spec.fidelity_plan)
        used_confirmation_seeds = set()

        while stage_queue:
            fidelity = stage_queue.popleft()
            seed = select_seed(fidelity, used_confirmation_seeds)
            request = build_run_request(patch_check, fidelity, seed)
            append_execution_started(request)
            result = await runner.run(request, health_observer)
            append_execution_finished(result)

            if result.outcome != SUCCESS:
                if await recover(result):
                    restart_at_recovery_selected_stage()
                    break
                append_terminal_invalid()
                return

            output = await output_gate.check(result)
            append_output_checked(output)
            if not output.accepted:
                if await recover(output):
                    restart_at_recovery_selected_stage()
                    break
                append_terminal_invalid()
                return

            if fidelity == SMOKE:
                decision = deterministic_smoke_promotion(output)
            else:
                evaluation = await evaluator.evaluate(build_eval_request(output))
                append_evaluation_completed(evaluation)

                if evaluation.trust.verdict == NO_OP:
                    if await recover_no_op(evaluation):
                        restart_at_recovery_selected_stage()
                        break

                decision = await evaluator.decide(evaluation, decision_context())

            append_experiment_decided(decision)
            maybe_record_lesson(decision.lesson_candidate)

            if decision.decision == PROMOTE:
                assert decision.next_fidelity is not None
                if fidelity == FULL and decision.next_fidelity == FULL:
                    # Nonterminal confirmation of the same commit/config.
                    # This is a new execution attempt with the next frozen seed,
                    # not a new planner proposal or an internal evaluator loop.
                    used_confirmation_seeds.add(seed)
                    assert confirmation_budget_remaining()
                    stage_queue.appendleft(FULL)
                else:
                    assert legal_fidelity_transition(fidelity, decision.next_fidelity)
                    if not stage_queue or stage_queue[0] != decision.next_fidelity:
                        stage_queue.appendleft(decision.next_fidelity)
                continue
            if decision.best_eligible:
                append_best_updated(decision)
                repair_best_git_ref_projection()
            return
```

Person 1 recommends research. Person 5 determines evaluation truth. Person 4 determines recovery. The harness owns routing and deterministic limits.

The real implementation must reject nonconsecutive duplicate stages in the original plan, cap confirmation attempts from frozen config, and include every confirmation run in the event ledger and resource totals. `select_seed` is deterministic from the frozen seed schedule. Exhausting confirmation budget converts an unresolved candidate to terminal `reject`/`inconclusive`; it must never loop.

### 10.3 Stop and finalization

Before each proposal and after each full evaluation, check convergence, experiment/query budgets, Agent wall-clock, configured token/GPU ceilings, fatal integrity conditions, and whether no legal proposal remains. A planner recommendation can supply evidence, but it cannot bypass these deterministic checks.

On stop:

1. append `run.stopped`;
2. select latest verified `best.updated`;
3. run clean reproduction through Persons 3–5;
4. append `final.selected` only after reproduction;
5. generate/check submission and append `submission.checked`;
6. allow hidden-final evaluation only after stop;
7. never route hidden-final information back to planning;
8. regenerate final evidence views.

## 11. Context builder

### 11.1 General requirements

- deterministic selection and ordering;
- role-specific data minimization;
- exact event/artifact/commit citations;
- redaction before persistence/provider call;
- token estimate and hard input budget;
- immutable context artifact and hash;
- `context.created` records sources and exclusions.

Do not use an LLM summarizer in the core. Use deterministic templates and compact typed events.

### 11.2 Planner context

Include contract digest, budgets, baseline, current best, eligible frontier, same-family verified history, up to five active tag-matching lessons, method cards, public-query count, convergence pressure, and bounded delta-vector orthogonality. Exclude hidden-final, unbiased-audit numbers, inconclusive or invalid/provisional evidence, secrets, full logs, unrelated trajectories, superseded lessons and suspicious metrics as positive rewards. Target 4,000–6,000 tokens.

### 11.3 Coder context

Include one ExperimentSpec, target interfaces/files, editable/protected summary, parent commit, selected method card, applicable lessons and output/budget contract. Target instruction: 1,500–2,500 tokens before Trae tool observations.

### 11.4 Recovery context

Include original hypothesis, accepted patch identity, exact failure, normalized error plus relevant trace tail, failed checks, previous error fingerprints and remaining repair budget. Explicitly prohibit hypothesis drift. Target 2,000–3,000 tokens.

### 11.5 Evaluation handoff

No LLM context. Build a typed request from verified output, hashes, seed and baseline/parent/best references.

### 11.6 Retrieval order

1. mandatory contract/budget;
2. selected parent/current best;
3. exact family/tag match;
4. verified status;
5. integrity rule > accepted result > negative result > failure detail;
6. sequence descending;
7. event ID ascending tie-break.

## 12. Idempotency and restart

Key format:

```text
run_id : experiment_id : stage : attempt : immutable_input_hash
```

On resume:

1. validate JSONL schema, sequence and hash chain;
2. remove only an incomplete trailing fragment;
3. rebuild all projections and idempotency index;
4. revalidate decision-bearing artifacts;
5. reconcile derived `best/<run_id>`;
6. handle artifact-without-event via validation and append or quarantine;
7. continue from last legal state.

Never rerun an expensive action because an acknowledgement was lost.

## 13. Token and resource accounting

- Every adapter returns one action-local `ResourceDelta`.
- Use provider token usage when available; otherwise estimate and label it.
- Aggregate provider and estimated tokens separately.
- P3 supplies run CPU/GPU/memory usage.
- P5 supplies evaluator usage.
- P4 supplies recovery LLM usage when present.
- `manual.intervention` is the only intervention-count source.

## 14. CLI

```text
tacorank run --config <path>
tacorank resume --run-id <id>
tacorank status --run-id <id>
tacorank validate-ledger --run-id <id>
tacorank rebuild-views --run-id <id>
tacorank finalize --run-id <id>
```

After `run.started`, execution is non-interactive except emergency stop. Configuration is frozen and hashed.

## 15. Integration rule

Your first merge must provide `schemas.py`, `ports.py`, valid/invalid fixtures, fake adapters, and a complete fake end-to-end test. Each teammate builds against those fixtures. Schema changes require all affected contract tests in one integration commit. No duplicate enums.

## 16. Implementation order

### P0

1. Freeze schemas and ports.
2. Implement validation and fixtures.
3. Implement canonical JSON/hash-chain append.
4. Implement replay and projections.
5. Implement fake adapters and fake full lifecycle.

### P1

6. Contract verification and baseline bootstrap.
7. Context builder/redaction/token estimator.
8. Idempotency/resume.
9. Convergence/budget enforcement.
10. Integrate Person 5 evaluator first for real baseline parity.

### P2

11. Integrate Person 1 Planner.
12. Integrate Person 3 worker/runner/gates.
13. Integrate Person 4 observer/recovery.
14. Run a real hypothesis, protected-patch rejection and recovery.

### P3

15. Clean reproduction/finalization.
16. Evidence views and resource summary.
17. Failure hardening only; no redesign.

## 17. Required tests

### Schema/ledger

- unknown fields/enums and non-finite metrics rejected;
- `PlannerOutput` discriminator and `planner.recommended` payload invariants;
- path/hash/ID validation;
- hash chain and sequence;
- duplicate idempotency;
- crash-tail handling;
- serialized append;
- artifact hash mismatch.

### State machine

- all legal and illegal transitions;
- no execution without receipt;
- no evaluation before Gate B;
- no best from proxy/no-op/suspicious/invalid;
- maximum-two-repair behavior;
- no-op routed to Person 4 before terminal decision;
- convergence uses verified full validation only.

### Context

- byte-deterministic output;
- token caps;
- hidden-final and secret exclusion;
- invalid/provisional/suspicious reward exclusion;
- role-specific contents and source hashes.

### Restart/integration

- crash after every major event without duplicate side effects;
- fake full lifecycle;
- real baseline;
- complete Planner→Trae→gates→run→evaluation loop;
- code error repair loop;
- timeout exact rerun once;
- accepted result changes next Planner context;
- hidden-final cannot trigger proposal.

## 18. Definition of done

- One schema module is used by all teammates.
- Fake end-to-end loop passes before real adapters.
- Baseline parity is recorded.
- State reconstructs without another database.
- Resume is idempotent at every phase.
- Every LLM call has context hash and token usage.
- Only verified evidence reaches planning.
- Contract/protected hashes are enforced.
- Stop and final selection are deterministic.
- Final report includes hypothesis, diff, metrics, recovery, tokens, GPU-hours and interventions.

## 19. Handoff checklist

Provide versioned schemas/ports, fixtures and test command, config template, fake adapters, vertical-slice test, state-transition diagram, owned-path matrix, schema-change procedure, CLI usage and resume instructions.

Your component is correct when the other four adapters can be replaced independently without changing orchestration logic.
