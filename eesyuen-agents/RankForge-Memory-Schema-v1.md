# RankForge Memory Schema v1.0

**Scope:** implementation contract for memory, state reconstruction, context building, and audit evidence.  
**Storage restriction:** canonical memory uses only Markdown, JSONL, and Git.  
**Runtime assumption:** one sequential orchestrator and at most tens of experiments per run.

## 1. Design decision

RankForge has exactly three canonical memory authorities:

1. **Contract authority — Markdown**  
   Human-frozen rules, benchmark contract, protected paths, budgets, and allowed commands.
2. **Evidence authority — JSONL**  
   An append-only, typed event ledger containing every hypothesis, patch, check, execution, failure, recovery, evaluation, decision, lesson, resource delta, and intervention.
3. **Code authority — Git**  
   Exact candidate code, diffs, experiment ancestry, and the trusted-best pointer.

Everything else is derived:

- current run state;
- experiment graph metadata;
- active lessons view;
- current status page;
- final run summary;
- role-specific LLM contexts.

There is no SQLite, relational database, vector database, mutable experiment table, or second authoritative state file.

## 2. Storage layout

```text
contract/
  COMPETITION.md                 # canonical, human-frozen
  PROTECTED_PATHS.md             # canonical, human-frozen

research/
  methods/
    <method_id>.md               # canonical human-authored knowledge

runs/<run_id>/
  events.jsonl                   # canonical dynamic memory
  STATUS.md                      # derived; safe to regenerate/delete
  LESSONS.md                     # derived; safe to regenerate/delete
  SUMMARY.md                     # derived at convergence
  contexts/
    <context_id>.md              # immutable prompt snapshot, evidence artifact

artifacts/<run_id>/<experiment_id>/
  diff.patch
  trajectory.jsonl
  run.log
  predictions.csv
  checkpoint.*
  receipts/

Git refs:
  experiment/<run_id>/<experiment_id>
  best/<run_id>
```

Predictions, checkpoints, logs, trajectories, and submission CSVs are **artifacts**, not prompt memory. The ledger stores only compact metadata and hash-addressed references to them.

## 3. Authority and write permissions

| Store | Authoritative for | Writer | LLM direct write? |
| --- | --- | --- | --- |
| `COMPETITION.md` | Benchmark rules and data boundary | Human before run | Never |
| `PROTECTED_PATHS.md` | Protected/editable roots and command allowlist | Human before run | Never |
| `methods/*.md` | Seed research knowledge | Human/reviewed team change | Never during a run |
| `events.jsonl` | Dynamic run and research truth | Orchestrator only | Never |
| Git commits/refs | Candidate code and ancestry | Git controller after accepted patch bytes | Trae proposes bytes; controller commits |
| `STATUS.md` | Human-readable current projection | Reporting code | Never authoritative |
| `LESSONS.md` | Human-readable active lessons | Reporting code | Never authoritative |
| `SUMMARY.md` | Judge-facing final projection | Reporting code | Never authoritative |
| `contexts/*.md` | Exact redacted context sent to a role | Context builder | Never edited after creation |

`producer` inside an event records which component supplied the content. It does not mean that component wrote the ledger. The orchestrator validates every object and is the sole appender.

## 4. Identifier rules

All IDs are run-local except `run_id`.

| Field | Format | Example |
| --- | --- | --- |
| `run_id` | `run_YYYYMMDD_<slug>` | `run_20260829_a` |
| `event_id` | `evt_` + six-digit ledger sequence | `evt_000017` |
| `experiment_id` | `exp_` + four digits | `exp_0006` |
| `lesson_id` | `lesson_` + four digits | `lesson_0004` |
| `context_id` | `ctx_` + role + six digits | `ctx_planner_000021` |
| `artifact_id` | `art_` + eight hexadecimal characters | `art_8f31a9c2` |
| Git commit | lowercase 40-character SHA-1 or 64-character SHA-256, matching repository format | `4ac19e2...` |
| SHA-256 | lowercase 64 hexadecimal characters | `9f86d081...` |

IDs are opaque after creation. They are never reused, renumbered, or inferred from filenames other than the explicitly defined event sequence.

## 5. JSONL physical format

`runs/<run_id>/events.jsonl` follows these rules:

- UTF-8.
- Exactly one compact JSON object per line.
- Every committed event ends with `\n`.
- No comments, blank lines, trailing commas, NaN, positive infinity, or negative infinity.
- JSON numbers used for metrics must be finite.
- Object keys are serialized in lexicographic order for hashing.
- Complete lines are never edited, reordered, or deleted.
- Only an incomplete, non-newline-terminated crash tail may be truncated back to the last committed newline.
- The final ledger is committed with the submission evidence. Optional checkpoint commits may copy complete event bytes, but must never rewrite them.

Canonical bytes for hashing:

```python
json.dumps(
    event_without_event_hash,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
).encode("utf-8")
```

`event_hash = sha256(canonical_bytes).hexdigest()`.

`prev_event_hash` equals the preceding event's `event_hash`; the first event uses 64 zeroes. This makes complete-event modification, deletion, and reordering detectable.

## 6. Universal event envelope

Every JSONL line has this exact top-level shape. Unknown top-level fields are rejected in schema v1.

```json
{
  "schema_version": "1.0",
  "seq": 17,
  "event_id": "evt_000017",
  "timestamp": "2026-08-29T03:14:15.123Z",
  "run_id": "run_20260829_a",
  "experiment_id": "exp_0006",
  "attempt": 1,
  "event_type": "evaluation.completed",
  "producer": "evaluator",
  "evidence_status": "verified",
  "causation_event_id": "evt_000016",
  "idempotency_key": "run_20260829_a:exp_0006:evaluation:full:seed_0:pred_9f86d081",
  "payload": {},
  "artifacts": [],
  "resource_delta": {
    "llm_input_tokens": 0,
    "llm_output_tokens": 0,
    "token_measurement": "none",
    "wall_time_ms": 4210,
    "cpu_time_ms": 3890,
    "gpu_time_ms": 0,
    "gpu_count": 0,
    "peak_rss_mb": 512,
    "peak_gpu_memory_mb": null,
    "manual_interventions": 0
  },
  "prev_event_hash": "<64 lowercase hex characters>",
  "event_hash": "<64 lowercase hex characters>"
}
```

### 6.1 Envelope field constraints

| Field | Type | Required | Constraint |
| --- | --- | --- | --- |
| `schema_version` | string | Yes | Exactly `1.0` for this run |
| `seq` | integer | Yes | Starts at 1; contiguous and strictly increasing |
| `event_id` | string | Yes | Must equal `evt_{seq:06d}` |
| `timestamp` | string | Yes | RFC 3339 UTC with `Z` |
| `run_id` | string | Yes | Must equal the run directory identity |
| `experiment_id` | string or null | Yes | Null only for run-level events |
| `attempt` | integer or null | Yes | Null for run-level events; otherwise starts at 0 for planning and 1 for first code/run attempt |
| `event_type` | enum | Yes | One of the event types in section 9 |
| `producer` | enum | Yes | `human`, `orchestrator`, `planner`, `trae`, `gate_a`, `runner`, `recovery`, `gate_b`, `evaluator`, `reporter` |
| `evidence_status` | enum | Yes | `provisional`, `verified`, or `invalid` |
| `causation_event_id` | string or null | Yes | Immediate event that caused this event; null only when none exists |
| `idempotency_key` | string | Yes | Unique in the run; deterministic from stage and input identity |
| `payload` | object | Yes | Discriminated by `event_type`; unknown payload fields rejected |
| `artifacts` | array | Yes | May be empty; every decision-bearing external file must be referenced |
| `resource_delta` | object | Yes | Resource usage attributable to this action only, never cumulative |
| `prev_event_hash` | string | Yes | Hash-chain predecessor |
| `event_hash` | string | Yes | Hash of this event excluding `event_hash` |

### 6.2 Evidence status semantics

- `provisional`: content exists but has not passed the applicable deterministic gate. Example: a Trae patch.
- `verified`: occurrence and content passed the applicable check. A verified failure is still verified evidence.
- `invalid`: content was structurally or contractually invalid and cannot support planning or scoring.

Do not use `evidence_status` to represent whether an experiment improved. Improvement is represented by `evaluation.completed` and `experiment.decided`.

## 7. Shared sub-schemas

### 7.1 `ArtifactRef`

```json
{
  "artifact_id": "art_8f31a9c2",
  "kind": "predictions",
  "path": "artifacts/run_20260829_a/exp_0006/valid.csv",
  "sha256": "<64 lowercase hex characters>",
  "size_bytes": 4821931,
  "content_type": "text/csv"
}
```

Constraints:

- `kind` is one of `diff`, `trajectory`, `context`, `log`, `checkpoint`, `predictions`, `metrics`, `verification_receipt`, `submission`, `report`, `other`.
- `path` is repository-relative, normalized, contains no `..`, and resolves inside an approved artifact root.
- A symlink is rejected.
- `sha256` and `size_bytes` are computed from stored bytes.
- Large bytes are never embedded in `payload`.

### 7.2 `ResourceDelta`

```json
{
  "llm_input_tokens": 2431,
  "llm_output_tokens": 611,
  "token_measurement": "provider",
  "wall_time_ms": 28450,
  "cpu_time_ms": 1320,
  "gpu_time_ms": 0,
  "gpu_count": 0,
  "peak_rss_mb": 384,
  "peak_gpu_memory_mb": null,
  "manual_interventions": 0
}
```

Constraints:

- All counts and durations are non-negative integers.
- `token_measurement` is `provider`, `estimated`, or `none`.
- Estimated and provider-reported tokens must never be silently combined; reporting groups them separately.
- `gpu_time_ms` is allocation time for the action. GPU-hours are derived as `sum(gpu_time_ms × gpu_count) / 3_600_000`.
- Agent wall-clock is derived from `run.started.timestamp` to `run.stopped.timestamp`; summed action wall time is reported separately and may overlap only if parallelism is later introduced.
- Every manual action creates a `manual.intervention` event; `manual_interventions` is 1 only on that event and 0 elsewhere.

### 7.3 `MetricSet`

Metric names are frozen by `COMPETITION.md`, not hard-coded into the memory schema.

```json
{
  "metrics": {
    "gauc": 0.6674,
    "ndcg_at_5": 0.5357
  },
  "primary_metric_name": "primary",
  "primary_score": 0.60155
}
```

Constraints:

- Every required contract metric appears exactly once.
- Extra metrics are permitted only if declared as diagnostic metrics in the contract.
- Values must be finite numbers.
- `primary_score` must reproduce the frozen aggregation formula within tolerance.

### 7.4 `CostEstimate`

```json
{
  "llm_tokens_upper_bound": 6000,
  "wall_time_seconds_upper_bound": 420,
  "gpu_seconds_upper_bound": 0,
  "cost_tier": "low"
}
```

`cost_tier` is `low`, `medium`, or `high`. Upper bounds must fit the remaining run budget when `experiment.proposed` is accepted.

## 8. Contract and method Markdown schemas

### 8.1 `contract/COMPETITION.md`

This file is written by humans before the run and protected by hash. It must contain these headings in this order:

```md
# Competition Contract

## Identity and source precedence
## Required benchmark
## Data and temporal boundary
## Target label and permitted inputs
## Metrics and primary aggregation
## Official baseline
## Convergence and resource limits
## Editable and protected paths
## Allowed commands
## Evaluation isolation
## Submission schema
## Resolved ambiguities
## Human approvals
```

The official-source contradiction must be explicitly resolved under `Resolved ambiguities`. Candidate code never decides whether the task uses one label/metric contract or another.

At `run.started`, its complete SHA-256 is stored. Before every Gate A and evaluation, the hash is checked. A mismatch forces `run.stopped` with reason `contract_changed`.

### 8.2 `research/methods/<method_id>.md`

The first fenced block is machine-readable JSON:

```json
{
  "schema_version": "1.0",
  "method_id": "method_pairwise_bpr",
  "family": "objective",
  "status": "candidate",
  "tags": ["pairwise", "within_user", "ranking"],
  "cost_tier": "medium",
  "sources": ["https://example.org/paper"]
}
```

It is followed by required Markdown headings:

```md
## Mechanism
## Preconditions
## Allowed data
## Expected effect
## Falsification condition
## Do not use when
## Minimal implementation
## Sources
```

`family` is one of `objective`, `sampling`, `temporal_history`, `features`, `model`, `multitask`, `duration_bias`, `ensemble`, `evaluation`, or `other`.

`status` is `candidate`, `blocked`, `known_negative`, or `forbidden`. Dynamic experiment outcomes do not rewrite method cards; they are events and lessons.

## 9. Typed event payloads

### 9.1 Run bootstrap events

#### `run.started`

Run-level event: `experiment_id = null`, `attempt = null`, `producer = orchestrator`.

```json
{
  "sequential": true,
  "contract_path": "contract/COMPETITION.md",
  "contract_sha256": "<sha256>",
  "protected_paths_sha256": "<sha256>",
  "source_commit": "<git commit>",
  "budgets": {
    "max_experiments": 50,
    "max_full_evaluations": 12,
    "max_agent_wall_time_seconds": 21600,
    "max_llm_tokens": null,
    "max_gpu_seconds": null
  },
  "convergence": {
    "epsilon": 0.002,
    "patience": 3,
    "population": "public_validation"
  }
}
```

Limits marked `null` are unbounded only when the frozen contract says they are not fixed. They are still measured.

#### `contract.verified`

```json
{
  "contract_sha256": "<sha256>",
  "protected_paths_sha256": "<sha256>",
  "data_manifest_sha256": "<sha256>",
  "evaluator_sha256": "<sha256>",
  "submission_checker_sha256": "<sha256>",
  "resolved_target_signature": {
    "label": "<contract-defined label>",
    "metrics": ["<metric_1>", "<metric_2>"],
    "primary_formula": "<frozen formula>"
  }
}
```

#### `baseline.verified`

The baseline is the experiment-tree root, conventionally `exp_0000`.
Its envelope uses `experiment_id = exp_0000` and `attempt = 1`.

```json
{
  "baseline_experiment_id": "exp_0000",
  "commit_sha": "<git commit>",
  "seed": 0,
  "metric_set": {
    "metrics": {"<metric_1>": 0.0, "<metric_2>": 0.0},
    "primary_metric_name": "primary",
    "primary_score": 0.0
  },
  "published_primary_score": 0.0,
  "absolute_error": 0.0,
  "tolerance": 0.0,
  "parity_passed": true
}
```

Research cannot begin unless `parity_passed` is true.

### 9.2 Context and planning events

#### `context.created`

```json
{
  "context_id": "ctx_planner_000021",
  "role": "planner",
  "purpose": "propose_next_experiment",
  "source_event_ids": ["evt_000002", "evt_000017"],
  "source_method_ids": ["method_pairwise_bpr"],
  "source_commit_shas": ["<git commit>"],
  "excluded_categories": ["hidden_final", "invalid", "secrets"],
  "input_token_budget": 6000,
  "estimated_input_tokens": 4382
}
```

The exact context is referenced as an artifact with `kind = context`. It is immutable after creation.

`role` is `planner`, `coder`, `recovery`, or `evaluator`.

#### `planner.recommended`

This run-level event records a non-proposal planner outcome. `experiment_id = null`, `attempt = null`, and `producer = planner`.

```json
{
  "context_id": "ctx_planner_000021",
  "action": "recommend_stop",
  "reason_code": "NO_LEGAL_UNTRIED_METHOD",
  "reason": "Every currently applicable method card has a verified terminal result or exceeds the remaining budget.",
  "supporting_event_ids": ["evt_000017", "evt_000020"]
}
```

`action` is `recommend_stop` or `blocked`; proposals use `experiment.proposed` instead. Every supporting event must occur in the referenced context. This event is advisory: only the orchestrator may append `run.stopped`, after validating a frozen deterministic stop condition. If the condition is not met, the orchestrator invokes its single bounded deterministic fallback; failure to find a legal action stops with `no_legal_action` rather than looping.

#### `experiment.proposed`

This payload is the canonical `ExperimentSpec`.

```json
{
  "parent_experiment_id": "exp_0003",
  "parent_commit_sha": "<git commit>",
  "context_id": "ctx_planner_000021",
  "hypothesis": "Within-user pairwise optimization will improve ranking over pointwise loss.",
  "family": "objective",
  "change_summary": "Replace pointwise loss with pairwise BPR while retaining the baseline representation.",
  "target_stage": "training",
  "target_files": ["solution/train.py"],
  "fidelity_plan": ["smoke", "proxy", "full"],
  "expected_mechanism": "The loss directly optimizes positive-negative ordering within each user.",
  "success_criteria": {
    "proxy_parent_delta_min": 0.001,
    "full_parent_delta_min": 0.002,
    "required_metric_direction": "non_decreasing_all"
  },
  "falsification_condition": "A verified full run fails to exceed the parent beyond the configured trust threshold.",
  "estimated_cost": {
    "llm_tokens_upper_bound": 6000,
    "wall_time_seconds_upper_bound": 420,
    "gpu_seconds_upper_bound": 0,
    "cost_tier": "medium"
  },
  "method_card_ids": ["method_pairwise_bpr"],
  "evidence_event_ids": ["evt_000003", "evt_000017"],
  "duplicate_key": "<sha256 of normalized parent+family+change>"
}
```

Validation rules:

- Parent must be baseline or a full, verified, accepted, non-suspicious experiment.
- `parent_commit_sha` must match the Git ref recorded for the parent.
- `target_files` must be normalized editable paths.
- `evidence_event_ids` must exist and be eligible for Planner context.
- `duplicate_key` must not already exist under the same parent unless the new proposal explicitly identifies a different seed-only confirmation.
- Estimated cost must fit the remaining budget.
- One main mechanism per experiment unless `family = ensemble`.

### 9.3 Coding and patch-verification events

#### `patch.created`

```json
{
  "experiment_spec_event_id": "evt_000022",
  "context_id": "ctx_coder_000023",
  "base_commit_sha": "<parent commit>",
  "patch_commit_sha": "<new commit>",
  "diff_sha256": "<sha256 of exact patch bytes>",
  "changed_files": ["solution/train.py"],
  "trae_version": "<pinned version or commit>",
  "model_id": "<provider/model>",
  "must_patch": true,
  "steps_used": 9
}
```

The exact diff and trajectory are artifact references. `evidence_status` is `provisional`.

#### `patch.checked`

```json
{
  "patch_event_id": "evt_000024",
  "patch_commit_sha": "<commit>",
  "diff_sha256": "<sha256>",
  "accepted": true,
  "receipt_id": "receipt_patch_exp_0006_attempt_1",
  "receipt_sha256": "<sha256>",
  "checks": [
    {"name": "protected_paths", "status": "pass", "details": null},
    {"name": "syntax_import", "status": "pass", "details": null},
    {"name": "interface_contract", "status": "pass", "details": null},
    {"name": "data_boundary", "status": "pass", "details": null}
  ],
  "violations": []
}
```

Each check status is `pass`, `fail`, or `not_applicable`. A failed check requires `accepted = false`, at least one violation, and `evidence_status = verified` because the rejection itself is verified evidence.

Violation schema:

```json
{
  "code": "PROTECTED_PATH_MODIFIED",
  "path": "evaluate.py",
  "message": "The candidate patch modifies the protected evaluator."
}
```

### 9.4 Execution and recovery events

#### `execution.started`

```json
{
  "patch_receipt_id": "receipt_patch_exp_0006_attempt_1",
  "patch_commit_sha": "<commit>",
  "fidelity": "proxy",
  "command_id": "candidate_proxy",
  "seed": 0,
  "data_manifest_sha256": "<sha256>",
  "limits": {
    "timeout_seconds": 420,
    "memory_mb": 8192,
    "gpu_memory_mb": null,
    "network_enabled": false
  }
}
```

`fidelity` is `smoke`, `proxy`, `full`, or `final`.

#### `execution.finished`

```json
{
  "execution_started_event_id": "evt_000026",
  "patch_commit_sha": "<commit>",
  "fidelity": "proxy",
  "outcome": "success",
  "exit_code": 0,
  "error_class": null,
  "error_fingerprint": null,
  "error_summary": null,
  "prediction_artifact_id": "art_8f31a9c2",
  "checkpoint_artifact_id": "art_143bc992"
}
```

`outcome` is one of:

- `success`
- `code_error`
- `interface_error`
- `contract_error`
- `numerical_error`
- `oom`
- `timeout`
- `hang`
- `infrastructure_error`
- `cancelled`

On non-success, `error_class`, `error_fingerprint`, and `error_summary` are required. The full trace remains in a log artifact.

`error_fingerprint` is the SHA-256 of normalized error class plus the top relevant stack frames or contract violation codes. It enables repeated-error detection without injecting full logs into planning context.

#### `recovery.decided`

```json
{
  "failure_event_id": "evt_000027",
  "repair_attempt": 1,
  "action": "trae_repair",
  "reason_code": "REPAIRABLE_CODE_ERROR",
  "instructions": "Fix the reported shape mismatch without changing the experiment hypothesis or protected interfaces.",
  "same_error_count": 1,
  "remaining_repair_budget": 1
}
```

`action` is `trae_repair`, `retry_same_commit`, `adjust_approved_runtime_setting`, `rollback`, or `abandon`.

Rules:

- Maximum two Trae code-repair decisions per experiment.
- The same error fingerprint twice forces `abandon`.
- Infrastructure failure may use `retry_same_commit` once without an LLM call.
- A valid low metric can never cause `recovery.decided`.
- A repaired patch creates a new `patch.created` event and must pass Gate A again.

### 9.5 Output and evaluation events

#### `output.checked`

```json
{
  "execution_finished_event_id": "evt_000031",
  "prediction_artifact_id": "art_8f31a9c2",
  "accepted": true,
  "checks": {
    "header": "pass",
    "row_count": "pass",
    "row_id_contiguous": "pass",
    "row_alignment": "pass",
    "duplicates_preserved": "pass",
    "finite_scores": "pass",
    "score_diversity": "pass"
  },
  "score_stats": {
    "rows": 124909,
    "unique_scores": 121344,
    "minimum": -4.812,
    "maximum": 5.114
  },
  "violations": []
}
```

No evaluator call is permitted unless `accepted = true` and this event is verified.

#### `evaluation.completed`

```json
{
  "output_checked_event_id": "evt_000032",
  "prediction_artifact_id": "art_8f31a9c2",
  "population": "public_validation",
  "fidelity": "full",
  "seed": 0,
  "public_query_index": 4,
  "evaluator_sha256": "<sha256>",
  "contract_sha256": "<sha256>",
  "metric_set": {
    "metrics": {"<metric_1>": 0.0, "<metric_2>": 0.0},
    "primary_metric_name": "primary",
    "primary_score": 0.0
  },
  "baseline_delta": 0.0,
  "parent_delta": 0.0,
  "previous_best_delta": 0.0,
  "prediction_change": {
    "spearman_vs_parent": 0.0,
    "changed_row_fraction": 0.0
  },
  "trust": {
    "verdict": "accepted",
    "stability": "single_seed",
    "integrity": "clean",
    "flags": []
  }
}
```

`population` is `internal_proxy`, `public_validation`, or `hidden_final`.

`trust.verdict` is:

- `accepted`
- `inconclusive`
- `negative`
- `no_op`
- `suspicious`

`trust.integrity` is `clean`, `compromised`, or `inconclusive`.

`trust.stability` is `single_seed`, `confirmed`, `unstable`, or `not_applicable`.

Hidden-test isolation rules:

- `hidden_final` is legal only after a verified `run.stopped` event.
- Hidden labels and metrics never appear in a Planner, Coder, or Recovery context.
- Hidden results never cause a new `experiment.proposed` event.

#### `experiment.decided`

```json
{
  "evaluation_event_id": "evt_000033",
  "decision": "accept",
  "reason_code": "TRUSTED_IMPROVEMENT",
  "fidelity_completed": "full",
  "parent_eligible": true,
  "best_eligible": true,
  "next_fidelity": null,
  "supporting_event_ids": ["evt_000025", "evt_000032", "evt_000033"]
}
```

`evaluation_event_id` is nullable only for smoke promotion. In that case,
`supporting_event_ids` must contain the verified smoke execution and Gate-B
events that justify promotion.

`decision` is `promote`, `accept`, `reject`, `prune`, or `invalid`.

- `promote` is non-terminal and requires `next_fidelity`.
- `accept`, `reject`, `prune`, and `invalid` are terminal for that experiment.
- `parent_eligible = true` requires a verified full public-validation result with verdict `accepted` and integrity `clean`.
- `best_eligible = true` additionally requires a primary score greater than the current trusted best under the contract's comparison rule.

#### `best.updated`

```json
{
  "previous_best_experiment_id": "exp_0003",
  "previous_best_commit_sha": "<commit>",
  "previous_best_primary_score": 0.0,
  "new_best_experiment_id": "exp_0006",
  "new_best_commit_sha": "<commit>",
  "new_best_primary_score": 0.0,
  "evaluation_event_id": "evt_000033"
}
```

`best.updated` is canonical; the Git ref `best/<run_id>` is its derived pointer.
After appending the event, the controller moves the ref to
`new_best_commit_sha`. Replay repairs a missing or stale pointer from the latest
verified `best.updated` event.

### 9.6 Lesson events

#### `lesson.recorded`

```json
{
  "lesson_id": "lesson_0004",
  "category": "research_result",
  "status": "active",
  "tags": ["objective", "pairwise", "within_user"],
  "summary": "Pairwise within-user sampling improved both contract metrics over the pointwise parent.",
  "applicability": "Use when each training group contains both positive and negative examples.",
  "avoid_when": "Do not construct pairs across users or from users with only one class.",
  "confidence": 0.9,
  "source_event_ids": ["evt_000033", "evt_000034"],
  "source_commit_shas": ["<commit>"]
}
```

`category` is `research_result`, `resource_constraint`, `implementation_constraint`, `integrity_warning`, or `process_rule`.

A lesson is permitted only when:

- the source is a verified positive or negative experiment result;
- recovery is exhausted and exposes a reusable constraint;
- a suspicious result exposes a reusable integrity rule; or
- a previous lesson needs to be superseded or marked stale.

One-off syntax/import mistakes do not become lessons.

#### `lesson.status_changed`

```json
{
  "lesson_id": "lesson_0004",
  "new_status": "stale",
  "reason": "The result was measured under an objective frame superseded by exp_0011.",
  "source_event_ids": ["evt_000071"]
}
```

`new_status` is `active`, `stale`, `superseded`, or `retracted`. Old lesson events remain unchanged. Retrieval uses the latest status event.

### 9.7 Human intervention and termination events

#### `manual.intervention`

```json
{
  "actor": "team_member_2",
  "reason": "Organizer clarification changed the frozen metric contract.",
  "action": "Stopped the run before changing the contract.",
  "affected_experiment_id": null,
  "code_changed": false,
  "effect": "A new run must be started under a new contract hash."
}
```

Any human edit, manual retry, manual parent choice, manual metric decision, or manual recovery after `run.started` is an intervention and must be logged.

#### `run.stopped`

```json
{
  "reason": "converged",
  "best_experiment_id": "exp_0006",
  "best_commit_sha": "<commit>",
  "best_primary_score": 0.0,
  "experiments_proposed": 9,
  "full_evaluations_completed": 6,
  "consecutive_non_improving_full_evaluations": 3,
  "total_manual_interventions": 0,
  "budget_snapshot": {
    "agent_wall_time_seconds": 15420,
    "llm_input_tokens_provider": 81234,
    "llm_output_tokens_provider": 17452,
    "llm_input_tokens_estimated": 0,
    "llm_output_tokens_estimated": 0,
    "gpu_hours": 0.0
  }
}
```

`reason` is `converged`, `max_experiments`, `max_full_evaluations`, `max_wall_time`, `max_tokens`, `max_gpu_time`, `contract_changed`, `fatal_integrity_failure`, `no_trusted_candidate`, or `manual_emergency_stop`.

#### `final.selected`

```json
{
  "experiment_id": "exp_0006",
  "commit_sha": "<commit>",
  "selection_evaluation_event_id": "evt_000033",
  "clean_reproduction_passed": true,
  "checkpoint_artifact_id": "art_143bc992",
  "validation_predictions_artifact_id": "art_8f31a9c2",
  "selection_reason": "Highest trusted public-validation score at deterministic stop."
}
```

#### `submission.checked`

```json
{
  "final_selected_event_id": "evt_000081",
  "submission_artifact_id": "art_d0f713a2",
  "checker_sha256": "<sha256>",
  "accepted": true,
  "violations": []
}
```

## 10. Derived run-state schema

Run state is reconstructed by folding `events.jsonl`; it is never written as canonical JSON.

```json
{
  "run_id": "run_20260829_a",
  "status": "running",
  "phase": "evaluation",
  "active_experiment_id": "exp_0006",
  "active_attempt": 1,
  "active_fidelity": "full",
  "best_experiment_id": "exp_0003",
  "best_commit_sha": "<commit>",
  "best_primary_score": 0.0,
  "experiments_proposed": 6,
  "full_evaluations_completed": 3,
  "public_validation_queries": 3,
  "consecutive_non_improving_full_evaluations": 1,
  "remaining_budgets": {
    "experiments": 44,
    "full_evaluations": 9,
    "agent_wall_time_seconds": 17240,
    "llm_tokens": null,
    "gpu_seconds": null
  },
  "resource_totals": {
    "llm_input_tokens_provider": 0,
    "llm_output_tokens_provider": 0,
    "llm_input_tokens_estimated": 0,
    "llm_output_tokens_estimated": 0,
    "cpu_time_ms": 0,
    "gpu_weighted_time_ms": 0,
    "manual_interventions": 0
  }
}
```

### 10.1 Run status enum

- `initializing`
- `ready`
- `running`
- `stopped`
- `finalizing`
- `finalized`
- `failed`

### 10.2 Phase enum

- `contract_verification`
- `baseline_reproduction`
- `planning`
- `coding`
- `patch_verification`
- `execution_smoke`
- `execution_proxy`
- `execution_full`
- `output_verification`
- `evaluation`
- `decision`
- `recovery`
- `finalization`
- `complete`

### 10.3 Convergence projection

Only verified `evaluation.completed` events with `population = public_validation` and `fidelity = full` affect convergence.

For each such event in sequence:

1. Let `previous_best` be the trusted best before this event.
2. If the new eligible score improves `previous_best` by more than contract epsilon, reset `consecutive_non_improving_full_evaluations` to 0.
3. Otherwise increment it by 1.
4. Stop when the counter reaches contract patience.

Proxy, smoke, invalid, suspicious, and hidden-final results never affect convergence.

## 11. Derived experiment-node schema

The experiment graph view combines `experiment.proposed`, later events, and Git ancestry:

```json
{
  "experiment_id": "exp_0006",
  "parent_experiment_id": "exp_0003",
  "hypothesis": "...",
  "family": "objective",
  "base_commit_sha": "<commit>",
  "latest_patch_commit_sha": "<commit>",
  "attempts": 1,
  "highest_fidelity_completed": "full",
  "status": "accepted",
  "metric_set": null,
  "trust_verdict": "accepted",
  "parent_eligible": true,
  "best_eligible": true,
  "terminal_event_id": "evt_000034"
}
```

Experiment status is one of:

- `proposed`
- `patch_ready`
- `patch_rejected`
- `ready_to_run`
- `running`
- `recovering`
- `output_ready`
- `output_verified`
- `evaluated`
- `accepted`
- `rejected`
- `pruned`
- `invalid`

Terminal statuses are `accepted`, `rejected`, `pruned`, and `invalid`.

Git consistency rules:

- The experiment branch must descend from `parent_commit_sha`.
- Each repair commit stays on the same experiment branch.
- `latest_patch_commit_sha` is the most recent Gate-A-accepted commit.
- The branch is preserved after terminal decision.
- Only a full accepted experiment or baseline root may be selected as a future parent.

## 12. State transition table

| Current state | Event | Condition | Next state |
| --- | --- | --- | --- |
| none | `experiment.proposed` | valid spec | `proposed` |
| `proposed` / `recovering` | `patch.created` | valid patch metadata | `patch_ready` |
| `patch_ready` | `patch.checked` | accepted | `ready_to_run` |
| `patch_ready` | `patch.checked` | rejected, retries remain | `recovering` |
| `patch_ready` | `patch.checked` | rejected, no retries | `invalid` after terminal decision |
| `ready_to_run` | `execution.started` | receipt and commit match | `running` |
| `running` | `execution.finished` | success | `output_ready` |
| `running` | `execution.finished` | failure, recovery chosen | `recovering` |
| `running` | `execution.finished` | failure, abandon | `invalid` after terminal decision |
| `output_ready` | `output.checked` | accepted | `output_verified` |
| `output_ready` | `output.checked` | rejected, repair chosen | `recovering` |
| `output_verified` | `evaluation.completed` | verified proxy/full result | `evaluated` |
| `output_verified` | `experiment.decided` | smoke promotion | `ready_to_run` at proxy |
| `evaluated` | `experiment.decided` | promote | `ready_to_run` at next fidelity |
| `evaluated` | `experiment.decided` | accept | `accepted` |
| `evaluated` | `experiment.decided` | reject | `rejected` |
| any non-terminal | `experiment.decided` | invalid | `invalid` |

Any transition not listed is rejected before append.

## 13. Context retrieval contract

### 13.1 Planner context

Always include:

- frozen contract digest and remaining budgets;
- baseline root and current trusted best;
- eligible frontier nodes;
- at most three verified results in the proposed family;
- at most five active lessons matching exact tags;
- selected method cards;
- public-validation query count and convergence pressure.

Exclude:

- provisional or invalid evidence;
- suspicious/compromised scores as positive rewards;
- hidden-final events;
- full logs and trajectories;
- secrets or environment dumps;
- superseded/retracted lessons;
- unrelated experiments outside the candidate lineage/frontier.

### 13.2 Coder context

Include only:

- one `ExperimentSpec`;
- parent commit and target interfaces/files;
- editable/protected summary;
- selected method card;
- applicable active lessons;
- output and budget contract.

Do not include competitor-wide score history, hidden data, or unrelated trajectories.

### 13.3 Recovery context

Include only:

- original hypothesis and accepted patch identity;
- exact failing attempt;
- normalized error summary plus relevant trace tail;
- failed check/contract details;
- prior repair fingerprints;
- remaining repair budget;
- instruction not to change the research hypothesis.

### 13.4 Evaluation context

The evaluator is deterministic and does not require an LLM context. It receives only:

- Gate-B-accepted predictions;
- protected labels for the declared population;
- evaluator and contract hashes;
- baseline, parent, and prior-best metric references;
- seed and resource evidence.

### 13.5 Retrieval ordering

When more eligible records exist than the token budget permits, rank deterministically by:

1. mandatory contract/budget records;
2. selected parent and current best;
3. exact family/tag match;
4. verified status;
5. importance: integrity constraint > accepted result > negative result > failure detail;
6. recency by `seq` descending;
7. `event_id` ascending as final tie-break.

Every created context records included event IDs, omitted categories, token estimate, and artifact hash.

## 14. Lesson projection rules

`LESSONS.md` is generated from active `lesson.recorded` events after applying all `lesson.status_changed` events.

Required line format:

```md
- [lesson_0004][objective][active][confidence=0.90] Pairwise within-user
  sampling improved both contract metrics. Applies: groups with both classes.
  Avoid: cross-user pairs. Evidence: evt_000033, evt_000034; commit: 4ac19e2.
```

Deduplication key:

```text
sha256(normalize(category + sorted(tags) + applicability + avoid_when))
```

When a new lesson has the same key:

- do not edit the old event;
- emit a new `lesson.recorded` only if it adds independent support or contradicts it;
- if it supersedes the old lesson, emit `lesson.status_changed` for the old ID;
- context retrieval selects the active latest supported lesson.

## 15. Idempotency and restart rules

Before performing any external or expensive action, derive an idempotency key from immutable inputs:

```text
run_id : experiment_id : stage : attempt : input_identity
```

Examples:

```text
run_20260829_a:exp_0006:trae_patch:1:spec_2d711642
run_20260829_a:exp_0006:execution_full:1:commit_4ac19e2_seed_0
run_20260829_a:exp_0006:evaluation_full:1:pred_9f86d081
```

On restart:

1. Verify every complete line, sequence, event ID, hash chain, schema, and unique idempotency key.
2. Rebuild run state, experiment nodes, lessons, and resource totals.
3. Verify referenced decision-bearing artifacts still match their hashes.
4. Compare the active phase with existing output artifacts.
5. If an event with the desired idempotency key already exists, return its recorded result rather than repeating the action.
6. If an artifact exists without a committed event, validate it and either append the missing event or quarantine the artifact; never assume success.

## 16. Global invariants

The implementation must enforce all of these:

1. The orchestrator is the only ledger writer.
2. Complete JSONL lines are immutable.
3. Event sequence and hash chain are valid.
4. Every idempotency key is unique.
5. No LLM-generated object becomes memory before schema validation.
6. Contract and protected-file hashes match before patch execution and evaluation.
7. No raw secret, authorization header, full environment, or hidden label is stored.
8. Hidden-final information never enters future contexts or proposals.
9. Every score is linked to exact prediction, evaluator, contract, commit, seed, and data-manifest identities.
10. No evaluation occurs before Gate B acceptance.
11. No execution occurs without a receipt for the exact patch hash.
12. A repaired patch must be rechecked.
13. A verified low score is not a runtime failure.
14. Only full, clean, accepted nodes can become parents or trusted best.
15. Proxy and smoke results cannot update trusted best or convergence.
16. Resource totals are sums of event deltas, never mutable counters.
17. Every manual intervention is explicit.
18. Derived Markdown can be regenerated entirely from canonical memory.
19. Git ancestry and ledger parent identities must agree.
20. Final selection occurs only after deterministic stop and clean reproduction.

## 17. Minimum event sequence for one successful experiment

```text
run.started
contract.verified
baseline.verified
context.created                  # planner
experiment.proposed
context.created                  # coder
patch.created
patch.checked                    # Gate A accepts exact hash
execution.started                # smoke
execution.finished
output.checked
experiment.decided               # promote to proxy
execution.started                # proxy
execution.finished
output.checked
evaluation.completed
experiment.decided               # promote to full
execution.started                # full
execution.finished
output.checked                   # Gate B
evaluation.completed
experiment.decided               # accept/reject
best.updated                     # only if eligible and better
lesson.recorded                  # only if reusable
```

Minimum recovery insertion:

```text
execution.finished               # outcome=code_error
recovery.decided                 # action=trae_repair
context.created                  # recovery/coder
patch.created                    # attempt=2
patch.checked
execution.started
...
```

## 18. What each teammate must import and obey

- **Person 1:** envelope validation, append/fold, state transitions, context creation, convergence, idempotency, projections.
- **Person 2:** `ExperimentSpec`, method-card schema, evidence eligibility, duplicate key, parent eligibility.
- **Person 3:** `patch.created`, Git ancestry, artifact references, execution events, exact resource deltas.
- **Person 4:** `patch.checked`, `output.checked`, recovery decisions, receipt identity, violation codes.
- **Person 5:** `MetricSet`, `evaluation.completed`, trust verdicts, `experiment.decided`, lessons, resource/intervention reports.

No teammate may define a private alternative meaning for `experiment_id`, fidelity, verdict, resource usage, parent eligibility, or convergence.

## 19. P0 implementation order

1. Implement strict models for the envelope and shared sub-schemas.
2. Implement the event-type discriminated payload union.
3. Implement append lock, canonical serialization, hash chain, and `fsync`.
4. Implement replay validation and state projection.
5. Add golden valid and invalid JSONL fixtures.
6. Add illegal-transition tests.
7. Add duplicate idempotency tests.
8. Add contract/hash mismatch tests.
9. Add hidden-final retrieval exclusion tests.
10. Add one crash/restart integration test between every major phase.

Only after these pass should the real Planner, Trae, runner, and evaluator replace their deterministic fakes.

## 20. Final recommendation

Use `events.jsonl` as the only dynamic source of truth. Do not also maintain a writable `state.json`, experiment table, or separate reflection database. Git already supplies the experiment tree, while `COMPETITION.md` supplies immutable human intent. This division is sufficient for restart, context retrieval, branching, run-log submission, autonomy evidence, robustness evidence, token/GPU accounting, and final reproducibility without introducing a database coordination problem.
