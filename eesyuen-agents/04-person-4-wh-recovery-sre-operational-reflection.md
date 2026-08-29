# Person 4 — Recovery, SRE Monitoring, Self-Debugging and Operational Reflection

## Codex implementation brief

You own the system's response to operational failure. Your code observes live execution telemetry, identifies unhealthy behavior, classifies completed failures, chooses a bounded recovery action, constructs focused self-debug instructions, prevents repair thrashing, and produces reusable operational lessons when warranted.

You do not run processes, edit worktrees, call Trae directly, evaluate model quality, choose research hypotheses, or append memory. Person 3 owns execution mechanisms; Person 2 routes and records; Person 5 owns metric truth and research reflection.

## 1. Boundary with Persons 3 and 5

### Person 3 versus Person 4

Person 3:

- samples CPU/GPU/process/log telemetry;
- writes telemetry JSONL;
- enforces hard OS/container limits;
- sends each `TelemetrySample` to your observer;
- terminates the process when you return `MonitorDirective(action=terminate)`;
- returns `RunResult` and artifacts.

You:

- interpret telemetry and decide whether it is healthy;
- provide the termination reason;
- classify a failure using `RunResult` plus telemetry history;
- choose repair/retry/adjust/rollback/abandon;
- produce focused repair instructions and optional operational lessons.

### Person 4 versus Person 5 reflection

You write **operational reflection candidates** only:

- reusable OOM/resource thresholds;
- repeated error/repair constraints;
- hang/heartbeat/process behavior;
- implementation constraints discovered after exhausted recovery;
- safety/integrity failures caused by execution behavior.

Person 5 writes **research reflection candidates**:

- confirmed positive or negative model ideas;
- metric trade-offs;
- no-op research diagnosis evidence;
- suspicious evaluation patterns;
- recommendations for future hypotheses.

A one-off syntax/import mistake is an event, not a lesson. A valid low score is a research result, not a recovery failure.

## 2. Contract rule

Read resource limits, allowed runtime adjustments, heartbeat expectations, repair budget, and protected boundaries from frozen configuration supplied by Person 2. Do not hard-code competition metrics or inspect hidden-test information.

Monitoring must work for CPU-only baseline runs. GPU absence is normal, not an error.

## 3. Owned paths

```text
src/tacorank/sre/
  observer.py
  health_policy.py
  telemetry_window.py
  anomaly_detection.py
  heartbeat.py

src/tacorank/recovery/
  classifier.py
  policy.py
  self_debug.py
  fingerprints.py
  runtime_adjustments.py
  operational_reflection.py

tests/sre/
tests/recovery/
tests/failure_injection/
```

Import shared models from `src/tacorank/schemas.py`. Do not redefine them.

## 4. Shared interfaces

### 4.1 Live input: `TelemetrySample`

```text
timestamp, run_id, experiment_id, attempt
elapsed_ms
process_alive
last_output_age_ms
cpu_percent
rss_mb
gpu_utilization_percent: float | null
gpu_memory_mb: int | null
loss: float | null
gradient_norm: float | null
disk_free_mb: int | null
recent_output_tail: str | null
```

### 4.2 Live output: `MonitorDirective`

```text
action: continue | terminate
reason_code: str | null
summary: str | null
```

`summary` must be concise, redacted and safe for `RunResult.error_summary`. The full evidence stays in Person 3's telemetry/log artifacts.

### 4.3 Completed inputs

You may receive one of:

- `RunResult` for execution failure;
- rejected `PatchCheckResult` for patch/safety/interface failure;
- rejected `OutputCheckResult` for malformed prediction output;
- `EvaluationResult` with verdict `no_op` for silent implementation diagnosis.

Person 2 also supplies `RecoveryPolicyContext`:

```text
run_id, experiment_id
original_experiment_spec
current_patch_commit_sha
failure_event_id
attempt_history
prior_error_fingerprints
repair_attempts_used
max_repair_attempts
same_commit_retries_used
remaining_run_budget
allowed_runtime_adjustments
contract_summary
```

### 4.4 Main output: `RecoveryDecision`

```text
run_id, experiment_id
failure_event_id
repair_attempt
action: trae_repair | retry_same_commit |
        adjust_approved_runtime_setting | rollback | abandon
reason_code
instructions
same_error_count
remaining_repair_budget
lesson_candidate: LessonCandidate | null
```

### 4.5 Operational `LessonCandidate`

```text
origin: operational
category: resource_constraint | implementation_constraint |
          integrity_warning | process_rule
tags
summary
applicability
avoid_when
confidence
source_event_ids
source_commit_shas
```

Person 2 validates/deduplicates and appends it. You never allocate `lesson_id` or edit `LESSONS.md`.

## 5. Live SRE observer

Implement:

```python
class SREObserver(HealthObserver):
    def observe(self, sample: TelemetrySample) -> MonitorDirective:
        self.window.add(sample)
        return self.policy.evaluate(self.window)
```

The observer must be deterministic, fast, non-blocking and free of network/LLM calls. It runs inside the execution sampling loop.

### 5.1 Sampling assumptions

Person 3 samples every configurable interval, default two seconds. Your rolling window retains enough samples for the longest detector, normally 20–60 samples. Memory must remain bounded.

### 5.2 Immediate termination signals

Return `terminate` immediately when:

- process is reported dead before normal result collection;
- parsed loss or gradient norm is NaN/Inf;
- explicit CUDA/CPU out-of-memory signature appears;
- disk falls below the configured safe floor;
- contract/safety violation is reported during execution;
- memory exceeds a hard limit that Person 3 has not already enforced.

### 5.3 Hang detection

Do not equate low GPU utilization with a hang.

Classify a likely hang only when multiple signals agree for the configured deadline:

- no stdout/heartbeat progress;
- process remains alive;
- CPU utilization is near idle or unchanged;
- GPU utilization is near idle when GPU is expected;
- no checkpoint/artifact progress when progress tracking exists.

The heartbeat deadline comes from the command profile. Default guidance may be `max(60 seconds, 2 × expected normal output gap)`, but it must be configurable and tested against the baseline.

### 5.4 Numerical anomaly detection

Maintain a rolling finite history:

- immediate terminate on NaN/Inf;
- flag loss explosion when current finite loss exceeds a configured multiple, default 5×, of the median of a sufficiently populated recent window;
- flag gradient explosion similarly only when gradient norm is reliably emitted;
- require persistence for at least two samples for non-NaN spikes to avoid reacting to one noisy minibatch;
- disable a detector when the relevant metric is unavailable.

Return stable reason codes:

```text
PROCESS_DIED
HEARTBEAT_STALE
NUMERICAL_NONFINITE
LOSS_EXPLOSION
GRADIENT_EXPLOSION
CPU_MEMORY_LIMIT
GPU_MEMORY_LIMIT
EXPLICIT_OOM
DISK_LOW
EXECUTION_POLICY_VIOLATION
```

### 5.5 Telemetry privacy

Do not include full stdout, environment variables, paths containing credentials, or secrets in the directive. Person 3 redacts and stores raw artifacts.

## 6. Failure taxonomy

Normalize to the shared execution outcomes:

| Class | Typical evidence | Recovery family |
| --- | --- | --- |
| `code_error` | Traceback points into changed candidate code | Focused Trae repair |
| `interface_error` | Missing function, wrong shape/type, no-op wiring | Focused repair; early abandon on repeat |
| `contract_error` | Hidden access, protected behavior, invalid command | Repair only when clearly accidental; otherwise abandon |
| `numerical_error` | NaN/Inf/explosion | One focused repair; approved LR/precision adjustment only |
| `oom` | OS/CUDA OOM, hard memory limit | Approved batch/resource adjustment or rollback |
| `timeout` | Wall limit exceeded with evidence of progress | Exact retry only if transient; otherwise abandon/high-cost lesson |
| `hang` | Multi-signal no-progress diagnosis | Kill; exact retry once only for likely transient cause |
| `infrastructure_error` | Container/runtime/process loss without candidate-code frame | Same sealed commit retry once |
| `output contract` | Wrong row count/order/schema/finite score | Focused repair |
| `no_op` | Valid run but predictions essentially unchanged | Diagnose wiring; do not falsify hypothesis yet |

### 6.1 Preliminary versus authoritative classification

Person 3 supplies a preliminary `RunResult.outcome`. Your classifier may refine the reason using telemetry/log evidence but may not relabel a successful low metric as failure.

Return a stable reason code, not an essay.

## 7. Recovery policy

### 7.1 Global rules

- Maximum two Trae code-repair attempts per experiment.
- Same normalized error fingerprint twice forces `abandon`.
- At most one `retry_same_commit` for an infrastructure/transient failure.
- Every changed patch must return through Gate A.
- Same-commit retry uses the existing valid receipt and increments execution attempt.
- Runtime adjustments must come from an allowlist and remain within contract budget.
- Recovery must not change the original research hypothesis.
- Valid negative metrics never trigger recovery.

### 7.2 Decision matrix

| Failure | Attempt/history | Decision |
| --- | --- | --- |
| Syntax/import/name error | First occurrence | `trae_repair` |
| Type/shape/interface mismatch | First occurrence | `trae_repair` |
| Same error fingerprint again | Any | `abandon` |
| Gate A protected-path violation | Accidental, first | `trae_repair` with exact prohibition |
| Gate A integrity attack/secret/hidden access | Clear deliberate contract break | `abandon` and integrity lesson candidate |
| Gate B schema/alignment failure | First occurrence | `trae_repair` |
| NaN/Inf | First occurrence | `trae_repair` or approved setting adjustment |
| OOM | Batch/worker knob allowlisted | `adjust_approved_runtime_setting` |
| OOM | No legal smaller setting or repeats | `rollback`/`abandon` plus resource lesson |
| Infrastructure loss | No previous same-commit retry | `retry_same_commit` |
| Infrastructure loss | Already retried | `abandon` |
| Hang | Likely transient and no retry | `retry_same_commit` |
| Hang | Repeated | `abandon` plus process lesson |
| Timeout with steady progress | Higher timeout not allowed/budget insufficient | `abandon` as too costly |
| Evaluation `no_op` | First verified occurrence | `trae_repair` focused on wiring |
| Evaluation `no_op` | After repair or unchanged fingerprint | `abandon`; implementation constraint, not research falsification |

### 7.3 Approved runtime adjustments

Represent adjustments as named keys, never arbitrary command text:

```text
batch_size: decrease to next configured value
num_workers: decrease
mixed_precision: disable
timeout_profile: move to one preapproved higher profile
```

The allowed set and ranges come from run configuration. If an adjustment changes score-bearing semantics, treat it as a new patch/config identity requiring Gate A or a new ExperimentSpec rather than operational recovery.

## 8. Error fingerprinting and thrash prevention

Normalize:

- error class;
- exception type;
- top candidate-code frames with line numbers normalized where appropriate;
- Gate violation codes;
- output contract violation codes;
- monitor termination reason.

Compute SHA-256. Do not include timestamps, random temp paths, memory addresses or entire messages that change every run.

Thrashing rules:

- same fingerprint on two failed attempts → abandon;
- alternating between two fingerprints for three repair attempts is impossible because max repair is two; abandon at budget;
- a repair that introduces a new simple syntax/import error may consume remaining repair budget, but record the error transition;
- never grant extra hidden repairs outside the event log.

## 9. Self-debug repair instruction

Your component constructs instructions; Person 2 builds the full RecoveryContext and Person 3 calls Trae.

Required contents:

1. original hypothesis and expected mechanism;
2. exact accepted patch/commit identity;
3. failure class and normalized fingerprint;
4. concise error/contract evidence and relevant trace tail;
5. failed attempt number and previous repair outcomes;
6. target files/interfaces;
7. explicit requirement to explain the fault briefly before patching;
8. explicit requirement to preserve hypothesis and protected boundaries;
9. exact success check expected after repair;
10. remaining repair budget.

Example:

```text
The hypothesis remains pairwise within-user optimization. Do not change the
objective or add features. The accepted patch fails because train_step returns
shape [batch,1] while the loss contract expects [batch]. Explain where the
extra dimension originates, patch only the candidate training code, and make
the existing interface smoke test pass. Do not edit evaluator, data loader,
contract or command configuration. This is repair attempt 1 of 2.
```

Avoid vague instructions such as “try again,” “improve robustness,” or “fix all issues.”

## 10. Operational reflection

### 10.1 Trigger conditions

Produce a `LessonCandidate` only when:

- recovery is exhausted and evidence reveals a reusable resource/implementation constraint;
- a stable OOM threshold is established;
- repeated hang behavior identifies an invalid command/profile assumption;
- a safety/integrity failure reveals a reusable prohibition;
- a no-op repair shows a feature/config path is not wired.

Do not reflect for one-off syntax/import errors, transient infrastructure that succeeds on retry, or valid model underperformance.

### 10.2 Lesson requirements

- concise evidence-backed observation;
- distinguish fact from causal hypothesis;
- applicability and avoid condition;
- confidence based on support;
- source event IDs and commit SHAs;
- no secret or full trace;
- no metric interpretation beyond the operational fact.

Example:

```text
category: resource_constraint
tags: [gpu, oom, embedding]
summary: Full-fidelity runs at embedding_dim=128 exceeded the configured GPU memory limit twice.
applicability: Current full data and resource profile.
avoid_when: Do not propose dimension >=128 without a lower-memory implementation.
confidence: 0.95
```

## 11. Person 2 integration

```python
directive = sre_observer.observe(sample)       # called by Person 3

# After Person 2 records the failure:
decision = recovery_manager.decide(
    failure_event_id,
    run_or_gate_result,
    recovery_policy_context,
)

# Person 2 validates and appends recovery.decided.
# Person 2 then routes action to Person 3.
```

Your component never invokes `event_store.append`, Git, Trae, subprocess, evaluator or Planner.

## 12. Handling evaluation NO_OP

Person 5 can identify a silent no-op only after comparing predictions. Flow:

1. Person 5 returns `EvaluationResult.trust.verdict = no_op`.
2. Person 2 appends `evaluation.completed` but does not yet append terminal `experiment.decided`.
3. Person 2 constructs a failure-like recovery context and calls you.
4. You inspect experiment intent, prediction-change evidence and prior attempts.
5. First no-op normally routes to `trae_repair` focused on wiring/config consumption.
6. Repeated no-op routes to `abandon` and optional implementation lesson.
7. Person 5's research reflection must not claim the underlying hypothesis is falsified unless a later verified implementation actually tests it.

## 13. Implementation order

### P0 — deterministic recovery core

1. Import shared models/fixtures.
2. Implement bounded telemetry window.
3. Implement observer with process/nonfinite/OOM/hard-limit signals.
4. Implement error normalization/fingerprints.
5. Implement recovery decision matrix and budgets.
6. Implement focused instruction builder.
7. Pass fake failures from Person 2/3.

### P1 — SRE hardening

8. Add multi-signal hang detection.
9. Add rolling loss/gradient anomaly detection.
10. Add GPU/CPU/disk-aware policies.
11. Integrate live observer callback with Person 3.
12. Add exact retry and approved-setting cases.

### P2 — reflection and failure evidence

13. Add operational LessonCandidate triggers.
14. Add no-op routing.
15. Run real failure injections.
16. Generate recovery evidence for demo.

## 14. Required tests

### Observer tests

- healthy CPU baseline with no GPU;
- healthy GPU run with intermittent low utilization;
- process dies;
- stale output plus idle CPU/GPU becomes hang;
- long compute with active CPU/GPU does not become hang;
- NaN/Inf immediate termination;
- single loss spike tolerated; persistent explosion terminated;
- OOM signature;
- hard memory/disk condition;
- bounded telemetry window.

### Classifier/policy tests

- candidate traceback → code repair;
- infra error → same-commit retry once;
- repeated same fingerprint → abandon;
- Gate B alignment failure → focused repair;
- OOM allowlisted adjustment and repeated OOM abandon;
- timeout with progress and insufficient budget abandon;
- no-op first repair, repeated no-op abandon;
- valid negative metric rejected as recovery input;
- repair count never exceeds two.

### Reflection tests

- one-off syntax error produces no lesson;
- transient retry success produces no lesson;
- repeated OOM produces resource lesson;
- hidden access produces integrity lesson;
- exhausted no-op produces implementation lesson;
- sources/confidence/applicability required;
- no metric causal claim from operational reflection.

### Integration/failure injection

- syntax error → repair → Gate A → success;
- timeout → kill whole process group → exact retry;
- NaN → observer terminates → focused repair;
- OOM → approved batch adjustment → rerun;
- repeated fingerprint → branch abandoned;
- evaluation no-op → wiring repair path.

## 15. Definition of done

- Healthy CPU/GPU runs are not killed spuriously under test profiles.
- True hangs, OOM and non-finite training terminate promptly.
- Every failure gets a stable classification and bounded action.
- Repairs cannot exceed two or drift from the hypothesis.
- Same error fingerprint cannot thrash indefinitely.
- Infrastructure retry does not consume an LLM call.
- No valid low score enters operational recovery.
- Reusable operational constraints are emitted as typed, evidence-linked candidates.
- Person 3 controls processes; Person 2 controls routing/memory; boundaries are respected.

## 16. Handoff checklist

Give Person 2/3:

- `SREObserver` and configuration;
- `RecoveryManager` and decision table;
- reason/fingerprint code registry;
- self-debug instruction builder;
- operational reflection builder;
- healthy/failure telemetry fixtures;
- failure-injection tests and expected decisions;
- documented default thresholds and how command profiles override them.

Your integration surfaces are `TelemetrySample → MonitorDirective` and `failure evidence + context → RecoveryDecision`. Nothing else.
