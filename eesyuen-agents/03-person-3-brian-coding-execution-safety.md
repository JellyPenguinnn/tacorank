# Person 3 — Trae Coding, Git Worktrees, Sandboxed Execution and Safety Gates

## Codex implementation brief

You own the mechanism that turns one approved research specification into exact code, verifies that code is safe to run, executes it in isolation, emits telemetry, captures artifacts/resources, and validates prediction structure before evaluation.

Your component is powerful but intentionally not autonomous about research direction. It cannot choose the next experiment, edit the benchmark contract, judge metric quality, decide recovery policy, or write memory directly.

## 1. Responsibility boundary

You own:

- Trae adapter and coding prompts;
- exact patch/diff/trajectory capture;
- Git branch and disposable worktree lifecycle;
- Gate A: deterministic pre-execution patch/safety checks;
- sandbox/container construction and hard resource enforcement;
- allowlisted symbolic commands;
- process launch, process-group termination and artifact capture;
- telemetry collection and callback into Person 4's `HealthObserver`;
- Gate B: prediction schema/alignment/finite-score checks;
- clean reproduction and final inference mechanisms.

Person 4 owns interpretation of health and recovery policy. You collect raw telemetry, enforce hard limits and execute a `terminate` directive; Person 4 decides why the run is unhealthy and whether to retry, repair, adjust an approved runtime setting, roll back or abandon.

Person 5 owns metric evaluation and research verdict. Gate B must pass before Person 5 receives predictions.

Person 2 owns orchestration, memory and event append. Return typed objects only.

## 2. Contract rule

Never hard-code target label, metric names or K values. The competition material is inconsistent. Use only the frozen contract and protected manifests passed by Person 2.

Candidate code may use training labels allowed by the contract. It must not read hidden labels or future information. Public-validation labels belong to the protected evaluator path, not the candidate training/inference process.

## 3. Owned paths

```text
src/rankforge/coding/
  trae_adapter.py
  prompts.py
  output_parser.py
  redaction.py

src/rankforge/git/
  worktrees.py
  refs.py
  patches.py

src/rankforge/safety/
  protected_manifest.py
  patch_gate.py
  path_policy.py
  command_policy.py
  data_access_policy.py
  output_gate.py
  receipts.py

src/rankforge/execution/
  runner.py
  sandbox.py
  commands.py
  process.py
  telemetry.py
  resources.py
  artifacts.py

solution/
  ... Trae-editable candidate pipeline ...

tests/coding/
tests/git/
tests/safety/
tests/execution/
tests/failure_injection/
```

Do not modify `schemas.py`; import Person 2's shared models.

## 4. Shared interfaces

### 4.1 Inputs

#### `CoderContext`

```text
context_id, run_id, experiment_id
contract_sha256
experiment_spec: ExperimentSpec
parent_commit_sha
target_interface_excerpts
editable_roots, protected_paths
allowed_command_ids
selected_method_cards
active_lessons
step_limit, token_limit, wall_time_limit_seconds
context_artifact: ArtifactRef
```

#### `RecoveryContext`

```text
context_id, run_id, experiment_id, repair_attempt
original_experiment_spec
current_patch_commit_sha, accepted_patch_receipt_id
failure_class, error_fingerprint, error_summary
relevant_trace_tail
failed_checks
previous_repair_fingerprints
recovery_instructions
remaining_repair_budget
editable_roots, protected_paths
```

#### `RunRequest`

```text
run_id, experiment_id, attempt, fidelity
command_id, patch_commit_sha, patch_receipt_id
seed, data_manifest_sha256
timeout_seconds, memory_limit_mb, gpu_memory_limit_mb
network_enabled
```

### 4.2 Outputs

#### `PatchCandidate`

```text
schema_version, run_id, experiment_id, attempt
experiment_spec_event_id, context_id
base_commit_sha, patch_commit_sha, diff_sha256
changed_files
diff_artifact, trajectory_artifact
trae_version, model_id, steps_used
resource_delta
```

#### `PatchCheckResult`

```text
run_id, experiment_id, attempt
patch_commit_sha, diff_sha256
accepted
receipt_id, receipt_artifact
checks: list[CheckResult]
violations: list[Violation]
```

#### `TelemetrySample`

```text
timestamp, run_id, experiment_id, attempt
elapsed_ms, process_alive, last_output_age_ms
cpu_percent, rss_mb
gpu_utilization_percent, gpu_memory_mb
loss, gradient_norm, disk_free_mb
recent_output_tail
```

#### `RunResult`

```text
run_id, experiment_id, attempt, fidelity
patch_commit_sha
outcome, exit_code
error_class, error_fingerprint, error_summary
log_artifact, telemetry_artifact
checkpoint_artifact, prediction_artifact
resource_delta
```

#### `OutputCheckResult`

```text
run_id, experiment_id, attempt
prediction_artifact
accepted
checks
score_stats
violations
```

### 4.3 Health callback

Person 4 implements:

```python
class HealthObserver(Protocol):
    def observe(self, sample: TelemetrySample) -> MonitorDirective: ...
```

You sample and call it. A directive is:

```text
action: continue | terminate
reason_code: str | null
summary: str | null
```

Only your runner sends OS/container termination signals. Person 4 never manipulates the process directly.

## 5. Trae adapter

### 5.1 Pinning

Record:

- exact Trae package version or Git commit;
- provider/model ID;
- configuration hash;
- max steps and token limits;
- whether Docker mode is used;
- trajectory output path.

Credentials come from the approved process environment and never enter prompts, trajectories, artifacts, Git or ledger payloads.

### 5.2 Invocation

Use a disposable coding worktree and equivalent of:

```text
trae-cli run <bounded task>
  --working-dir <experiment-worktree>
  --must-patch
  --trajectory-file <artifact-path>
  --max-steps <configured limit>
```

If Docker is used for Trae itself, mount only the experiment worktree and required read-only references. The full training run is a separate controlled execution.

### 5.3 Coding prompt

The task must include:

- one exact ExperimentSpec;
- parent commit and target files;
- editable/protected boundaries;
- required interfaces and output format;
- permitted lightweight commands;
- selected method mechanism and prerequisites;
- applicable lessons;
- instruction to return a patch and concise explanation;
- instruction not to train full data, evaluate official metrics, modify memory or choose a different hypothesis.

For repair, also include exact error evidence and explicit instruction to preserve the original hypothesis.

### 5.4 Tool restrictions

Trae may:

- read files inside its worktree;
- edit only approved solution roots;
- run allowlisted lightweight checks such as formatting, import, unit tests and tiny smoke tests;
- inspect dependency files without changing them unless the spec explicitly permits a reviewed dependency change.

Trae may not:

- modify evaluator, splits, submission checker, contract, event ledger, policies or protected manifests;
- execute arbitrary network calls;
- access hidden/test labels;
- install arbitrary packages;
- run full training/evaluation;
- update Git refs outside its experiment branch;
- append memory or declare success.

### 5.5 Output normalization

After Trae completes:

1. compute `git diff` from exact parent;
2. reject no-diff output unless this is an explicit diagnosis-only task;
3. enumerate normalized changed paths;
4. write exact diff artifact;
5. redact trajectory;
6. commit patch on the experiment branch;
7. compute diff/commit/artifact hashes;
8. return `PatchCandidate` with provider token usage.

If Trae produces malformed output, return a typed coding failure to Person 2; do not invent patch metadata.

## 6. Git experiment and worktree model

### 6.1 Refs

- baseline is root `exp_0000` commit;
- branch: `experiment/<run_id>/<experiment_id>`;
- create branch from `ExperimentSpec.parent_commit_sha`;
- each repair is a new commit on the same experiment branch;
- rejected/invalid branches remain for evidence;
- `best/<run_id>` is a derived pointer controlled by Person 2.

### 6.2 Worktree lifecycle

```python
create_worktree(run_id, experiment_id, parent_commit)
verify_clean_and_exact_parent()
invoke_trae()
commit_patch()
run_gate_a()
execute_if_accepted()
preserve_artifacts()
remove_worktree_only_after_terminal_or_safe_checkpoint()
```

Use deterministic, validated paths. Never construct a destructive command from an unresolved variable or wildcard.

### 6.3 Consistency checks

- branch descends from declared parent;
- worktree clean before Trae;
- changed commit equals returned commit;
- diff bytes equal hashed artifact;
- repair commit descends from previous attempt;
- accepted receipt identifies exact commit and diff;
- execution rechecks commit/hash before launch.

## 7. Gate A — patch verification

Gate A is deterministic. It returns `PatchCheckResult`; it never asks an LLM whether a patch “looks safe.”

### 7.1 Required checks

1. `diff_parse`: patch parses cleanly.
2. `changed_file_match`: reported list equals actual diff paths.
3. `editable_path`: all paths under editable roots.
4. `protected_path`: no protected file modified, deleted or renamed.
5. `path_escape`: no absolute path, `..`, symlink or submodule escape.
6. `contract_hash`: contract/protected hashes unchanged.
7. `syntax_import`: approved syntax and import checks.
8. `interface_contract`: required entry points and signatures exist.
9. `command_policy`: code/config resolves only allowed command capabilities.
10. `data_boundary`: no hidden/test label or future-information access.
11. `network_policy`: no new unapproved network behavior.
12. `secret_scan`: no credential-shaped value.
13. `dependency_policy`: no unreviewed dependency change.
14. `smoke_test`: tiny legal sample completes when applicable.

### 7.2 Protected roots

At minimum protect:

```text
contract/
runs/
src/rankforge/memory/
src/rankforge/orchestrator/
src/rankforge/safety/
official evaluate.py
official data split/load logic
official submit.py
baseline score files
hidden-label storage
```

The exact list comes from `PROTECTED_PATHS.md` and its frozen hash.

### 7.3 Violation codes

Use stable codes such as:

```text
PROTECTED_PATH_MODIFIED
PATH_TRAVERSAL
SYMLINK_ESCAPE
DIFF_MISMATCH
CONTRACT_HASH_MISMATCH
INTERFACE_MISMATCH
HIDDEN_LABEL_ACCESS
FUTURE_INFORMATION_LEAKAGE
UNAPPROVED_COMMAND
UNAPPROVED_NETWORK
SECRET_DETECTED
DEPENDENCY_CHANGE
SYNTAX_IMPORT_FAILURE
SMOKE_FAILURE
```

### 7.4 Receipt

On acceptance, create a receipt artifact containing run/experiment/attempt, commit SHA, diff SHA, contract/protected/data-manifest hashes, checks and timestamp. Hash it. Execution requires the receipt ID and exact identities.

Any repaired patch invalidates the old receipt and must pass Gate A again.

## 8. Symbolic command registry

The runner never accepts raw LLM shell strings. Map `command_id` to reviewed executable/arguments/workdir/environment/resource profile.

Required IDs:

```text
baseline_full
candidate_smoke
candidate_proxy
candidate_full
candidate_final_infer
submission_check
clean_reproduce
```

The registry validates all arguments and injects experiment-specific paths itself. No `shell=True`.

## 9. Sandboxed execution

### 9.1 Isolation

- disposable worktree/container per experiment;
- read-only contract, protected evaluator and data manifests;
- training data view with allowed labels;
- validation features without protected target columns for candidate inference;
- protected evaluator separately sees validation labels;
- hidden input mounted only during final inference;
- network disabled by default;
- sanitized minimal environment;
- explicit working directory;
- process group for reliable termination;
- wall, memory, disk and GPU-memory limits;
- no patching while process runs.

### 9.2 Launch sequence

1. verify patch receipt and commit/diff identity;
2. verify worktree and protected hashes;
3. resolve symbolic command;
4. prepare artifact directory;
5. launch child in new process group/container;
6. begin stdout/stderr capture and telemetry sampling;
7. call Person 4 observer for each sample;
8. terminate on hard limit or `MonitorDirective.terminate`;
9. wait/reap entire process group;
10. compute artifact hashes and resource totals;
11. normalize result to `RunResult`.

### 9.3 Telemetry

Sample at configurable cadence, default two seconds:

- process liveness and elapsed time;
- timestamp of last output/heartbeat;
- CPU percent and RSS;
- GPU utilization/memory when GPU exists;
- disk free space;
- parsed loss/gradient norm when emitted;
- short recent output tail.

Write full samples to a JSONL telemetry artifact, not the main event ledger. The main `RunResult` contains only summary and artifact reference.

Do not treat low GPU utilization alone as a hang. Person 4 combines signals.

### 9.4 Hard enforcement

Your runner unconditionally enforces:

- wall timeout;
- memory limit;
- container/process exit;
- cancellation/termination directive;
- disk exhaustion prevention;
- process-group cleanup.

Return a result rather than raise expected run failures.

## 10. RunResult normalization

Map signals to preliminary outcomes:

- zero exit plus expected artifacts → `success`;
- nonzero exit with candidate-code frame → `code_error`;
- missing interface/output → `interface_error`;
- explicit policy/contract violation → `contract_error`;
- NaN/Inf/loss anomaly directive → `numerical_error`;
- OS/CUDA allocation failure → `oom`;
- hard wall limit → `timeout`;
- Person 4 hang directive → `hang`;
- executor/container/provider failure without candidate-code cause → `infrastructure_error`;
- orchestrator emergency stop → `cancelled`.

Person 4 performs authoritative recovery classification after receiving the result and telemetry summary.

`error_fingerprint` is SHA-256 of normalized error class plus relevant candidate stack frames or violation codes. Full trace remains an artifact.

## 11. Gate B — output verification

Before Person 5 evaluation, check contract-defined submission/prediction structure.

Required checks:

- expected header and column types;
- exact row count for population;
- zero-based contiguous `row_id` when contract requires it;
- official row order;
- user/item identity alignment;
- repeated user-item rows preserved;
- finite numeric scores;
- no missing/extra rows;
- sufficient score diversity to avoid degenerate ordering;
- prediction artifact hash and producer commit;
- no target column or hidden data included.

Return `OutputCheckResult`. Do not compute official metrics.

## 12. Person 4 integration

The boundary is:

```python
sample = telemetry_collector.sample(process)
directive = health_observer.observe(sample)
telemetry_writer.append(sample)

if directive.action == TERMINATE:
    terminate_process_group()
    return normalize_terminated_run(directive, telemetry)
```

After Person 2 records `execution.finished`, it calls Person 4's `RecoveryManager`. You execute the resulting action only when Person 2 routes a validated `RecoveryDecision`:

- `trae_repair` → use RecoveryContext and create new patch commit;
- `retry_same_commit` → same accepted commit/receipt, new execution attempt;
- `adjust_approved_runtime_setting` → change only named allowed runtime knob, generate new request/receipt if required;
- `rollback` or `abandon` → stop executing this experiment.

Person 4 never edits your worktree directly.

## 13. Artifacts

Store under:

```text
artifacts/<run_id>/<experiment_id>/attempt_<n>/
```

Capture:

- exact diff and Trae trajectory;
- Gate A receipt/check details;
- resolved command/configuration;
- stdout/stderr log;
- telemetry JSONL;
- checkpoint/model state;
- predictions;
- Gate B report;
- environment/dependency identity without secrets.

Every artifact returned through a shared interface has relative path, SHA-256, size and content type.

## 14. Resource measurement

Return action-local:

- Trae provider input/output tokens and measurement source;
- coding wall time;
- execution wall and CPU time;
- GPU allocation time and count;
- peak RSS and GPU memory;
- exit/termination overhead when measurable.

Do not calculate run-wide totals; Person 2 aggregates event deltas.

## 15. Implementation order

### P0 — deterministic path

1. Import shared schemas/fixtures.
2. Implement symbolic command registry.
3. Implement worktree creation/verification/cleanup.
4. Implement mock patch generator.
5. Implement Gate A core path/protected/hash checks.
6. Implement baseline runner and artifact capture.
7. Implement Gate B submission checks.
8. Pass fake vertical slice with Person 2 and Person 5.

### P1 — real coding and sandbox

9. Integrate pinned Trae with `--must-patch` and trajectory.
10. Implement diff normalization/redaction/commit.
11. Implement container/process isolation.
12. Implement telemetry JSONL and observer callback.
13. Implement hard resource limits/process-group cleanup.
14. Produce real `PatchCandidate`, `RunResult`, and `OutputCheckResult`.

### P2 — recovery and finalization

15. Integrate Person 4 observer/directives.
16. Implement repair-patch path and receipt invalidation.
17. Implement exact same-commit retry.
18. Implement clean reproduction and final inference.
19. Harden failure injection and secret redaction.

## 16. Required tests

### Trae/coding

- must-patch success and no-patch failure;
- malformed trajectory and missing token usage;
- context/credential redaction;
- step/time cap;
- exact base/patch commit relationship;
- repair commit stays on same experiment branch.

### Gate A

- protected evaluator/split/ledger/contract modification;
- path traversal, symlink and submodule escape;
- diff-list mismatch and patch substitution;
- syntax/import failure;
- correct syntax but wrong interface;
- hidden/test label access and future feature;
- arbitrary shell/network/dependency/secret attempt;
- accepted receipt matches exact diff.

### Runner

- successful CPU baseline;
- nonzero candidate error;
- timeout and full process-group cleanup;
- memory limit/OOM;
- observer termination;
- container/process loss;
- artifact hashing and missing artifact;
- network disabled;
- no zombie process.

### Gate B

- wrong header/count/order;
- row ID gaps;
- user/item misalignment;
- duplicate collapse;
- NaN/Inf/non-numeric score;
- degenerate score vector;
- wrong producer/hash;
- valid official-format predictions accepted.

### Integration

- Planner spec → real Trae patch → Gate A → smoke;
- Gate A rejection → P4 decision → repaired patch → Gate A;
- code error → repair commit → rerun;
- timeout → same sealed commit retry once;
- success → Gate B → Person 5 evaluation;
- clean reproduction of best commit.

## 17. Definition of done

- Trae operates only in the requested worktree and returns an exact patch.
- Git ancestry matches experiment ancestry.
- No score-bearing execution occurs without an exact receipt.
- Protected and hidden boundaries are enforced deterministically.
- Runner cannot hang the orchestrator and always cleans its process group.
- Person 4 receives live telemetry and can terminate through the defined directive.
- Every run produces typed outcomes and hash-addressed artifacts.
- Gate B blocks malformed output before evaluation.
- Clean reproduction and final inference use the same sealed mechanisms.

## 18. Handoff checklist

Provide Person 2:

- real and fake `CodingWorker`, `PatchGate`, `ExecutionRunner`, `OutputGate` adapters;
- symbolic command registry/config;
- worktree lifecycle documentation;
- protected-path and violation-code list;
- sandbox/container setup;
- valid/invalid shared-model fixtures;
- failure-injection test command;
- artifact directory contract;
- pinned Trae version/config without secrets.

Your integration surface is typed inputs and outputs. Never bypass Person 2 by sending patch/run results directly to Person 4 or Person 5.
