# TacoRank Agent Operations Guide

## Purpose

This file is the operating guide for an AI coding agent helping a human set up, test, run, monitor, validate, or troubleshoot TacoRank. Read it completely before changing files or starting a live run.

Your job is to make the requested workflow reproducible and evidence-backed. Do not stop at giving commands when the user has authorized execution and all prerequisites are available. Do not claim that a live provider, Docker, data, convergence, or final submission worked unless you directly observed the relevant command or durable evidence.

For a complete live run, the completion contract is:

1. production preflight passes without creating a ledger;
2. the autonomous loop runs until a frozen stop rule fires;
3. finalization selects the validation-best candidate or protected FM fallback;
4. the official test submission check is accepted;
5. the final projected status is `finalized`; and
6. `validate-ledger` succeeds.

A finalized run can legitimately select the baseline. It proves that the workflow and submission contract completed; it does not by itself prove convergence or an improvement over the baseline.

## First actions in every repository session

Work from the repository root and inspect before acting:

```bash
pwd
git rev-parse --show-toplevel
git status --short --branch
git submodule status --recursive
git log -1 --oneline
```

Then:

1. preserve all pre-existing user changes;
2. confirm the requested mode from the table below;
3. inspect the relevant implementation and tests rather than relying only on this guide;
4. verify prerequisites before starting expensive or stateful work; and
5. record exact commands and outcomes for the final report.

Do not commit, push, merge, delete run evidence, rotate credentials, download data, or start a paid live run unless the user has authorized that action. Normal setup, testing, and execution steps are authorized when the user explicitly asks you to perform that workflow.

## System model

TacoRank is a deterministic, event-sourced harness for autonomous recommender-system research on KuaiRand-Pure:

```text
frozen contract and official FM baseline
  -> ledger-derived planner context
  -> bounded keyless OpenAlex evidence for the selected method
  -> DeepSeek research proposal
  -> Trae edit in a disposable Git worktree
  -> DeepSeek implementation review and bounded Trae revision (at most 5 reviews)
  -> Gate A patch and lineage verification
  -> CPU smoke -> proxy -> full execution
  -> telemetry and bounded recovery
  -> Gate B output verification
  -> protected evaluation and reflection
  -> next experiment or deterministic stop
  -> clean reproduction of the validation best
  -> label-free official test inference
  -> final Gate B and official submission check
```

The deterministic controller owns workflow state, budgets, recovery routing, promotion, rollback, convergence, final selection, and ledger appends. After policy selects a legal method, the planner's bounded online literature skill retrieves a small keyless OpenAlex snapshot without sending run data or metrics; the DeepSeek proposal must cite that exact evidence. Paper text is untrusted data and cannot override policy. DeepSeek proposes a bounded research plan. Trae is an edit-only coding worker and neither coding nor candidate execution receives network access. Before sealing a patch, a separate DeepSeek verifier checks plan-to-code fidelity and may return concrete corrections for at most five internal passes. This verifier is not Gate A, does not receive protected metrics or hidden labels, and does not consume Waihong's two external recovery repairs. Role components return canonical typed records and cannot independently mutate workflow state.

Authoritative surfaces are:

- `contract/COMPETITION.md`, `PROTECTED_PATHS.md`, and generated hash-bound configuration for frozen policy;
- Git commits and Gate A receipts for executable lineage;
- `runs/<run_id>/events.jsonl` for ordered dynamic evidence; and
- the protected official evaluator for metric truth.

`state.json`, `STATUS.md`, reports, lessons, and experiment graphs are replayable views. Never edit them or the ledger to manufacture a result.

## Choose the correct operating mode

| User goal | Mode | Live key | Docker | Official data | Main command |
| --- | --- | --- | --- | --- | --- |
| Change or test repository code | Development | No | Usually no | No | `pytest` |
| Validate real DeepSeek + Trae coding and Gate A | Trae-only | Yes for live preflight/run | Yes | No | `trae-run-example` |
| Run autonomous ML research through final submission | Complete live run | Yes | Yes | Yes | `run` |

Do not present Trae-only validation as ML execution. Do not present deterministic tests as a live provider run. Do not present one completed experiment or a transition into the next experiment as elapsed convergence.

## Non-negotiable safety boundaries

- Never print, paste into a command log, commit, or write `DEEPSEEK_API_KEY` to a file. Ask the human to export it in their shell. Check only presence with `test -n "${DEEPSEEK_API_KEY:-}"`.
- If a credential has been pasted into chat, source, logs, or Git, treat it as exposed and tell the human to rotate it. Do not repeat it.
- Never commit `KuaiRand-Pure/data/`, `.tacorank/`, `.venv/`, run ledgers, submissions, trajectories, predictions, model artifacts, or other generated output.
- Never modify `kuairand-starter-kit/evaluate.py`, official split semantics, labels, submission ordering, contract hashes, seeds, metrics, or baselines merely to make a run pass.
- Candidate code may change only controller-approved editable roots, normally `solution/`. Preserve protected-path, Git-lineage, Gate A, execution-seal, and Gate B checks.
- Never execute an LLM-supplied shell string. Candidate execution must use the reviewed symbolic command registry and hardened Docker runner.
- Never reuse a completed run ID or overwrite an existing deployment/runtime directory. Choose a new identity and preserve prior evidence.
- Never hand-edit `events.jsonl`. Use `rebuild-views` only for derived views; use `resume` only when the CLI accepts the durable checkpoint.
- Keep hidden/test labels and final test output out of planning, search, convergence, and local metric feedback.

## Fresh clone and development setup

Both the superproject and official starter-kit submodule require repository access. Use the human's configured GitHub authentication method:

```bash
git clone --recurse-submodules https://github.com/JellyPenguinnn/tacorank.git
cd tacorank
git submodule update --init --recursive
```

If authenticated HTTPS is unavailable, use the human's approved SSH setup for the clone and ensure the submodule can also be fetched. Do not embed a token in a remote URL.

Create the control-plane environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/tacorank --help
```

The main package supports Python 3.9+. Production Trae setup separately requires an exact Python 3.12 executable.

## Development validation

Run the narrowest relevant tests first, then the complete suite before integration:

```bash
PYTHONPATH=src:. PYTHONPYCACHEPREFIX=/tmp/tacorank-pycache \
  .venv/bin/python -m pytest -q
```

Useful subsystem suites:

```bash
# Research, configuration, deployment, schemas, memory, context, and orchestration
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/research tests/config tests/deployment tests/schemas \
  tests/memory tests/context tests/orchestrator

# Trae, Git, both gates, CPU execution, and failure injection
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/coding tests/git tests/safety tests/execution \
  tests/failure_injection tests/integration

# Health and recovery
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/sre tests/recovery tests/integration/test_recovery_lifecycle.py

# Evaluation, reflection, and reporting
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/evaluation tests/reflection tests/reporting
```

If a listed test directory is absent on a later branch, inspect `tests/` and run the closest current scope. Never claim success for a command that was not executed.

## Trae-only live coding validation

Use this mode to prove DeepSeek model access, the pinned Trae runtime, hardened Docker edit tools, disposable worktrees, patch capture, and Gate A without downloading KuaiRand data or running ML training.

From a clean tracked checkout:

```bash
REPO_ROOT="$(pwd -P)"
TRAE_RUN_ID="trae_trial_001"
TRAE_DEPLOYMENT_DIR="$REPO_ROOT/.tacorank/trae-$TRAE_RUN_ID"
TRAE_RUNTIME_DIR="$(dirname "$REPO_ROOT")/.tacorank-runtime/$(basename "$REPO_ROOT")-trae-$TRAE_RUN_ID"

.venv/bin/tacorank setup-trae \
  --repository-root "$REPO_ROOT" \
  --deployment-dir "$TRAE_DEPLOYMENT_DIR" \
  --runtime-dir "$TRAE_RUNTIME_DIR"
```

If Python 3.12 or Docker is not discoverable, pass canonical executables with `--python312 /absolute/path/to/python3.12` and `--docker /absolute/path/to/docker`.

Validate locally before using a credential:

```bash
.venv/bin/tacorank trae-preflight \
  --config "$TRAE_DEPLOYMENT_DIR/trae-deployment.json" \
  --local-only
```

After the human exports the key in the same shell, run live preflight and one real coding action:

```bash
test -n "${DEEPSEEK_API_KEY:-}"

.venv/bin/tacorank trae-preflight \
  --config "$TRAE_DEPLOYMENT_DIR/trae-deployment.json"

.venv/bin/tacorank trae-run-example \
  --config "$TRAE_DEPLOYMENT_DIR/trae-deployment.json" \
  --input examples/trae/experiment-spec.json \
  --run-id "$TRAE_RUN_ID" \
  --experiment-id exp_0001
```

Accept this stage only when the command reports a real patch, immutable trajectory/diff evidence, an isolated worktree, and an accepted Gate A receipt. State explicitly that dataset execution, Gate B, evaluation, convergence, and finalization were not tested in this mode.

## Complete production run

### 1. Verify prerequisites

Before setup, confirm:

- the tracked checkout is clean;
- the pinned submodule is initialized at the recorded commit;
- control-plane Python is 3.9+;
- Python 3.12 is available for Trae;
- a local Docker-compatible daemon is running and reachable through a local Unix socket or Docker Desktop Windows named pipe;
- the human has authorized live DeepSeek use and exported `DEEPSEEK_API_KEY`;
- outbound HTTPS access to the keyless OpenAlex Works API is available; and
- official KuaiRand-Pure data is present inside this checkout, or setup is authorized to download it.

Safe checks:

```bash
git status --short
git submodule status --recursive
.venv/bin/python --version
python3.12 --version
docker version
docker info
test -n "${DEEPSEEK_API_KEY:-}"
```

Do not print the environment variable value.

### 2. Assign unique run paths

Use a new run ID for every independent attempt. Keep these variables in the same shell for setup, preflight, run, and lifecycle commands:

```bash
REPO_ROOT="$(pwd -P)"
RUN_ID="run_001"
DEPLOYMENT_DIR="$REPO_ROOT/.tacorank/deployments/$RUN_ID"
RUNTIME_DIR="$(dirname "$REPO_ROOT")/.tacorank-runtime/$(basename "$REPO_ROOT")-$RUN_ID"
RUN_CONFIG="$DEPLOYMENT_DIR/run-config.json"
LIVE_CONFIG="$DEPLOYMENT_DIR/live-adapters.json"
DATA_DIR="$REPO_ROOT/KuaiRand-Pure/data"
```

Replace `run_001` if any matching run, deployment, runtime, or worktree already exists. Do not delete prior evidence just to reuse the example ID.

### 3. Build the hash-bound deployment

To download the official dataset when required:

```bash
.venv/bin/tacorank setup-live \
  --repository-root "$REPO_ROOT" \
  --deployment-dir "$DEPLOYMENT_DIR" \
  --runtime-dir "$RUNTIME_DIR" \
  --data-dir "$DATA_DIR" \
  --run-id "$RUN_ID" \
  --download-data
```

If the required raw files already exist in `KuaiRand-Pure/data/`, omit `--download-data`. The data directory must be a real, non-symlinked directory inside the repository. Setup creates credential-free configuration, official data views, baseline predictions, an executable baseline-parity receipt, a data manifest, a pinned Trae environment, and a digest-bound Docker image. Every candidate view contains `fm_baseline_predictions.csv` and its `.sha256` identity; do not remove or substitute them. Preserve the setup JSON result because it reports the exact generated paths and image identity.

Setup requires a clean tracked checkout and creates directories exclusively. If it fails, diagnose the reported prerequisite. Do not partially reuse or overwrite an uncertain deployment.

### 4. Run non-mutating production preflight

```bash
.venv/bin/tacorank preflight \
  --config "$RUN_CONFIG" \
  --live-config "$LIVE_CONFIG"
```

Require exit code 0 and JSON containing:

```json
{"ledger_created": false, "runtime": "live", "status": "passed"}
```

Also verify that `runs/$RUN_ID/events.jsonl` does not yet exist. Preflight checks the clean Git baseline and submodule, frozen contracts, full data manifest, official evaluator and FM baseline, executable candidate parity on all routes, DeepSeek access, keyless OpenAlex search access, pinned Trae installation, Docker isolation, read-only edit tools, execution environment, and output quota.

### 5. Start and allow the actual loop to finish

```bash
.venv/bin/tacorank run \
  --config "$RUN_CONFIG" \
  --live-config "$LIVE_CONFIG"
```

The production command is intentionally quiet during long external calls. Keep it attached and allow it to return. The default deployment requests at most two directions concurrently from one planner snapshot: one outcome-routed primary action and one globally untried scouting method when available. A prior method may repeat only in the primary lane when the deterministic policy explicitly deepens or refines it; spare lanes must not replay previously attempted directions on a new parent. Typed transient provider failures are retried once under a 300-second per-request timeout, and the complete batch is sealed before its Trae worktrees and candidate executions run concurrently. Each lane receives independent evidence and serialized protected-query indices, then independently accepted improvements may pass through one newly gated synthesis candidate before the next round. The controller persists every handoff and automatically finalizes after a legal stop.

Default frozen search limits generated by `setup-live` are:

- improvement threshold `0.002`;
- convergence patience of three consecutive non-improving trusted full-fidelity iterations;
- at most 50 proposed experiments;
- six hours of run wall time;
- at most two repair patches per experiment; and
- CPU-only candidate execution.

Other legitimate stops include experiment or wall-time budget exhaustion and no remaining legal non-duplicate proposal. `fatal_integrity` and a final `failed` status require investigation and are not successful completion.

### 6. Monitor from another terminal

Recreate `REPO_ROOT` and `RUN_ID`, then use the read-only/projected interfaces:

```bash
.venv/bin/tacorank status \
  --run-id "$RUN_ID" \
  --repository-root "$REPO_ROOT"

sed -n '1,220p' "$REPO_ROOT/runs/$RUN_ID/STATUS.md"
sed -n '1,260p' "$REPO_ROOT/runs/$RUN_ID/reports/SUMMARY.md"
sed -n '1,220p' "$REPO_ROOT/runs/$RUN_ID/reports/RESOURCES.md"
```

Meaningful progress fields include `phase`, `last_event_id`, `experiments_proposed`, `full_evaluations_completed`, `convergence_pressure`, `best_experiment_id`, and `best_primary_score`.

For a degraded or failed experiment, inspect its generated file under `experiment-graph/directions/*/experiments/`. Label-free `diagnostic_metrics` show whether scores remain rankable and personalized and how far they moved from the FM parent. `adapter_failures` and `recovery_decisions` identify the exact stage, redacted summary, evidence artifacts, provider tokens, wall time, and bounded action. Treat the ledger as authoritative; reports are replayable views.

Do not infer a hang only because stdout is quiet. Check the status, active controller/Trae process, Docker container, latest ledger event, and resource telemetry before intervening. Do not interrupt a paid run unless the human asks or a verified safety boundary is at risk.

### 7. Validate completion and outputs

After `run` returns:

```bash
.venv/bin/tacorank status \
  --run-id "$RUN_ID" \
  --repository-root "$REPO_ROOT"

.venv/bin/tacorank validate-ledger \
  --run-id "$RUN_ID" \
  --repository-root "$REPO_ROOT"

.venv/bin/tacorank rebuild-views \
  --run-id "$RUN_ID" \
  --repository-root "$REPO_ROOT"
```

Require all of the following before calling the complete workflow successful:

- projected `status` and `phase` are both `finalized`;
- `final_experiment_id` is present;
- ledger validation prints `valid` with an event count and head hash;
- the ledger ends with `final.selected` and a `submission.checked` event whose `accepted` value is `true`;
- the referenced final submission artifact exists and its hash validates;
- `STATUS.md`, `reports/SUMMARY.md`, and `reports/RESOURCES.md` are internally consistent; and
- generated output remains ignored by Git.

Inspect the evidence tree:

```text
runs/<run_id>/
  events.jsonl                 authoritative append-only ledger
  state.json                   replayed state projection
  STATUS.md                    concise human status
  contexts/                    immutable role contexts
  lessons/                     durable research lessons
  experiment-graph/            lineage and direction views
  artifacts/                   patches, receipts, logs, telemetry, predictions
  reports/SUMMARY.md           outcome summary
  reports/RESOURCES.md         resource accounting
```

To locate the final submission without guessing its path, read the `submission_artifact.path` in the final `submission.checked` event. The harness has already run the protected official checker; rerun the official checker only when the user requests independent confirmation and use the same official test data and exact accepted artifact.

Record the source commit, submodule commit, run ID, config hashes, data-manifest hash, Docker image digest, provider/model, seeds, stop reason, selected experiment, metrics, resource totals, ledger head, submission artifact/hash, and exact validation commands. Separate observed results from interpretation.

## Resume, finalization, and recovery

Use the exact immutable configurations generated for that run:

```bash
.venv/bin/tacorank resume \
  --config "$RUN_CONFIG" \
  --live-config "$LIVE_CONFIG"

.venv/bin/tacorank finalize \
  --config "$RUN_CONFIG" \
  --live-config "$LIVE_CONFIG"
```

- Use `resume` only after an interrupted run when the CLI accepts an unambiguous `planning` or `planner_context` checkpoint. It repairs only an incomplete final JSONL fragment and fails closed during ambiguous external-adapter phases.
- Use `finalize` when a valid run has stopped but automatic finalization did not complete. It is idempotent after successful finalization.
- If recovery is rejected, preserve the ledger, artifacts, processes, and exact error. Report the last valid event and request the smallest needed operator decision.
- Never fabricate a missing provider, coding, execution, Gate A, Gate B, evaluation, or submission result.
- Never start another controller against the same run ID concurrently.

## Common failure triage

| Symptom | Required response |
| --- | --- |
| `setup-live` says the checkout is dirty | Preserve user changes; commit only if explicitly authorized, or use a separate clean checkout. |
| Submodule initialization fails | Verify access and the exact pinned gitlink; do not silently track another commit. |
| Python 3.12 or Docker is missing | Install only with authorization, or pass canonical executable paths. |
| Docker daemon/socket preflight fails | Start or repair the approved local daemon, then rerun preflight. |
| DeepSeek authentication/model check fails | Confirm only that the shell variable is present; ask the human to correct or rotate the key. Never inspect or print it. |
| OpenAlex preflight or lookup fails | Preserve run evidence and verify outbound HTTPS/rate-limit status. Do not fabricate or hand-edit literature evidence. |
| Setup directory already exists | Choose a new run/deployment/runtime identity after preserving the existing directory. |
| Run ID already has a ledger | Do not reuse it. Inspect/validate it or select a new run ID. |
| `resume` rejects the phase | Preserve evidence; the state is ambiguous and requires operator review. |
| Candidate fails plan-to-code verification | Inspect `solution_verification.json` and its per-pass trajectory/process artifacts. Do not bypass the verifier; its internal loop is capped at five. |
| Candidate fails Gate A or Gate B | Follow typed bounded recovery; never weaken a gate to admit the candidate. |
| A provider, verifier, gate, evaluator, or execution adapter fails transiently | Retry only the owning stage once against the same immutable input. Send Trae a repair prompt only after a typed result establishes a candidate-code defect. |
| Trae reports malformed or truncated tool JSON | Preserve the `adapter.failed` evidence. The pinned client first requests a smaller valid call in-loop. If a failure remains, follow Waihong's bounded self-recovery classification and its same-commit retry, abandon, or stop decision; do not hand-edit the worktree or ledger. |
| Trae reports a Windows `UnicodeEncodeError` for a status glyph | The isolated Trae process must force Python UTF-8 mode and UTF-8 standard streams. Update and commit the compatibility fix, then create a new hash-bound deployment and run ID; do not reuse the failed deployment. |
| Proxy/full score regresses | Let the deterministic controller prune/reject and continue with the legal search policy. |
| Candidate-scoped integrity rejection | Preserve the rejected evidence; allow only the controller's single clean restart from the declared trusted parent. Never weaken a gate or repair the rejected candidate in place. |
| `fatal_integrity` or final `failed` | Stop, preserve evidence, identify the system-scoped or repeated invariant violation, and do not call the run successful. |

## Architecture and contribution rules

Core code lives under `src/tacorank/`:

- `agents/`, `providers/`, `research/`: bounded research planning and search policy;
- `memory/`, `context/`, `orchestrator/`: ledger, replay, contexts, state machine, and routing;
- `coding/`, `git/`, `safety/`, `execution/`: Trae, bounded implementation verification, worktrees, both gates, Docker, and telemetry;
- `sre/`, `recovery/`: health observation and bounded recovery;
- `evaluation/`, `reflection/`, `reporting/`: protected metrics, selection, lessons, and views;
- `benchmarks/kuairand_pure/`: KuaiRand adapters;
- `solution/`: the normal coding-agent-editable candidate surface; and
- `kuairand-starter-kit/`: pinned official starter-kit submodule.

Shared event and handoff contracts belong in `src/tacorank/schemas.py`. Do not create adapter-local replacements. The controller/memory layer remains the sole ledger writer. Keep changes in the subsystem that owns the behavior and update direct callers, fixtures, and cross-component tests for shared-contract changes.

Follow Python 3.9+ compatibility, PEP 8, type hints on public APIs, deterministic seeds, explicit failures, and existing Pydantic/dataclass patterns. Avoid unrelated refactors and new dependencies.

Before handing off a code change:

1. run focused tests;
2. run the full suite when the blast radius justifies it;
3. review the complete diff and working tree;
4. scan for credentials and generated data/output;
5. document commands actually run and exact results; and
6. commit or push only with explicit user authorization.

## Required agent completion report

End an operational task with a concise evidence report containing:

- mode performed: development, Trae-only, or complete live run;
- repository commit and whether the tracked checkout was clean;
- setup/preflight commands and observed results;
- live run ID, final status, phase, and stop reason when applicable;
- best and final experiment identities and observed metrics when applicable;
- Gate A, execution, Gate B, evaluation, recovery, ledger, and submission evidence actually reached;
- tests and validators executed with pass/fail counts;
- confirmation that secrets, data, and run output were not committed; and
- any unverified claim, residual risk, external dependency, or exact blocker.

Use precise boundaries such as “deterministic integration test passed,” “live one-experiment acceptance passed,” or “live convergence observed.” Never collapse those into a broader claim than the evidence supports.
