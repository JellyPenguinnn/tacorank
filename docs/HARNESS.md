# TacoRank deterministic harness

TacoRank is the integration backbone for the five-person autonomous KuaiRand-Pure research agent. The controller is deterministic and event-sourced; researcher, coding, safety, execution, recovery, and evaluation components return typed values but cannot mutate shared state or choose their own authority.

## End-to-end control flow

```text
frozen contract + official baseline
                |
                v
ledger-derived research feedback -> bounded keyless OpenAlex evidence -> DeepSeek researcher -> ResearchProposal
                |                                      |
                |                                      v
                |                     controller binds code targets + ladder
                |                                      |
                |                                      v
                |                              Trae edits worktree
                |                                      |
                |                                      v
                |                             Gate A + receipt
                |                                      |
                |                                      v
                |                       smoke -> proxy -> full CPU run
                |                                      |
                |                             telemetry / recovery
                |                                      |
                |                                      v
                |                         Gate B -> protected evaluator
                |                                      |
                +------ lesson + result + decision <---+
                |
        next iteration or deterministic stop
                |
                v
clean reproduce validation best -> test inference -> Gate B -> submission check
```

`Harness.run_until_stopped()` uses the configured round width. Production requests at most two directions concurrently from one planner snapshot. The first is the normal outcome-routed policy choice; the optional second is a globally untried method, so parallel fan-out cannot repeat the same portfolio on every new parent. Typed transient provider failures are retried once under the frozen planning timeout. The controller atomically seals the complete batch before fan-out; each direction then gets its own Trae worktree, Gate A receipt, execution seal, Gate B result, protected evaluation, and terminal decision. Protected public-query indices are serialized at the evaluator boundary even though coding and execution are concurrent. When two or more round members independently improve the incumbent, a synthesis-capable research and coding pass receives their verified component patches, creates a fresh candidate from the strongest member, and traverses the same gates and evaluation ladder. Only then does the next planner snapshot begin. `Harness.run_to_completion()` adds finalization after a frozen convergence, experiment, wall-time, token, GPU, integrity, or no-legal-proposal stop.

Confirmation seeds are additional executions of the same experiment and commit. They are fully recorded and charged to resource totals, but only the terminal full-fidelity experiment decision consumes one convergence-patience slot. The default frozen rule is no improvement greater than `0.002` for three consecutive terminal full iterations.

## Authorities and durable state

The only authorities are:

- `contract/COMPETITION.md`, `PROTECTED_PATHS.md`, and the hash-bound run configuration for human-frozen policy;
- Git commits and Gate A receipts for executable candidate lineage;
- `runs/<run_id>/events.jsonl` for ordered dynamic evidence; and
- the protected official evaluator for metric truth.

The ledger uses contiguous event IDs, causal links, idempotency keys, file locking, `flush`/`fsync`, and a SHA-256 chain. Replay produces run state, experiment lineage, lessons, resource totals, and reports. `state.json`, `STATUS.md`, graph files, and reports are disposable derived views. Contexts and hash-addressed artifacts are immutable evidence.

Only the controller appends events. It validates each transition, owns budgets and convergence, routes recovery, updates the best candidate, and selects the final artifact. No LLM can stop the run, promote itself, read labels, write the ledger, or bypass a gate.

Planner memory has two intentionally separate layers. `PlannerContext.family_history`
is bounded working memory built from every visible evaluation, including negative proxy,
`no_op`, inconclusive, redundant, and suspicious results. Each summary retains its
fidelity, population, metrics and deltas, trust assessment, decision reason, prediction
change, and diagnostics so the next proposal can react without treating the result as a
durable fact. `PlannerContext.active_lessons` contains only active `lesson.recorded`
events selected by the deterministic retrieval policy. Hidden-final evaluations enter
neither layer, and `lessons/*.md` remains a generated human-readable projection rather
than a planner input or source of truth.

Suspicious non-compromised experiments are quarantined as non-reward evidence
and cannot become parents, refinements, or ensemble members. The search policy
continues from a verified eligible frontier parent when an independent legal
method remains. Compromised integrity still fails closed.

## Experiment lifecycle and gates

```text
proposed -> patch_ready -> ready_to_run -> running -> output_ready
                 |               ^                         |
                 v               |                         v
             recovering ---------+                  output_verified
                 |                                         |
                 v                                         v
              invalid                                  evaluated
                                                           |
                                     accepted / rejected / pruned
```

- Before a commit reaches Gate A, a strict DeepSeek reviewer compares the cumulative diff and complete target-file source against the exact ExperimentSpec, target interface, selected method cards, and active lessons. A rejection returns only grounded corrections to Trae; this solver/verifier cycle is capped at five reviews within the coding action's wall budget. Every pass records its diff hash, redacted trajectory/process log, verifier result, and provider usage.
- Gate A binds the exact patch commit and diff to the contract, data identity, ExperimentSpec target-file allowlist, allowed imports/dependencies, isolated entrypoint-import result, and verification receipt.
- The runner resolves reviewed symbolic commands; it never executes an LLM-supplied shell string.
- Docker execution is CPU-only, network-disabled, receipt-sealed, resource-bounded, and observed by Person 4.
- Gate B verifies the prediction schema, ordered row identity, finite/diverse scores, producing commit, data manifest, command, and execution seal before evaluation.
- Smoke can promote on structural success. Proxy and full require protected evaluation. Only a trusted public-validation full result can update the best.

`setup-live` gives every candidate route the exact official FM prediction plus its digest and proves the checked-in candidate reproduces it byte-for-byte. This makes the executable parent equal to the scored baseline. Candidate experiments normally preserve that parent and add one bounded train-only residual. Evaluation adds label-free rankability, personalization, residual-scale, and FM-correlation diagnostics to the canonical result and planner context; those diagnostics never replace protected metrics.

FM predictions are unconstrained real-valued ranking scores, not probabilities.
Coder contexts therefore make score-scale preservation, the selected method card,
and compact cited prior-result constraints mandatory; only optional lessons may be
removed to meet the context budget. Gate B also rejects outputs in which one exact
score value occupies more than the hash-bound run limit, preventing clipping-driven
zero/one collapse before protected evaluation.

Execution, Gate A, Gate B, and adapter failures enter the bounded recovery route. A verified evaluation no-op (`NO_PREDICTION_CHANGE`) first receives one implementation-level Trae repair action, capped at 20 internal Trae steps in newly generated production deployments. The repair task preserves the approved hypothesis, begins from the accepted diff and score-output path, and asks for the smallest wiring fix that produces non-identical predictions. The replacement patch must pass Gate A and the same fidelity is rerun. If predictions remain identical, recovery records `return_to_planner` and the node becomes a neutral terminal `no_op`; recovery does **not** emit an experiment prune decision. If Trae exhausts the bounded repair task without producing a valid patch, TacoRank preserves that worker failure and returns the original no-op evidence to planning by the same neutral route instead of stopping the run. The no-op node cannot become a parent or checkpoint. The research tree planner then ranks the legal next directions when available: one modified same-mechanism plan from the last trusted parent, or an independent mechanism that effectively retires the branch. A second planner-selected no-op for that parent/family/method removes the same-mechanism option.

Recovery is owner-aware. A malformed verifier or transient provider response retries that verifier, Gate A and Gate B adapter failures retry their own gate, evaluator failures retry the evaluator, and execution infrastructure failures retry the sealed execution input. Each owner retry is globally bounded and event replay restores its exact stage checkpoint; it does not rerun an unrelated earlier stage. A validated candidate code, interface, numerical, or output-contract defect may instead invoke Trae. That prompt contains the redacted error, fingerprint, failed checks, prior attempts, contract limits, and a bounded proposed correction; Trae must validate the proposal against the evidence before making the smallest edit. Every replacement patch needs a new Gate A receipt.

Candidate-scoped integrity findings—including protected-path edits, path/symlink/submodule escape, hidden or future-label access, unauthorized network use, and protected data in output—are never repaired in place. The controller preserves the rejected commit and violation evidence, resets only the disposable experiment branch to its declared trusted parent, and gives Trae one bounded clean restart with the exact finding as a hard constraint. If that restart repeats the violation or the repair budget is exhausted, only the experiment is abandoned and planning may continue. Credential detection, protected identity/receipt/hash inconsistency, ambiguous controller failure, disk/quota exhaustion, and exhausted run budgets remain fail-closed because the trusted recovery point or operating environment is not safe to assume.

The pinned DeepSeek Responses client catches malformed or truncated function arguments before Trae executes them and asks for one smaller valid JSON call inside the existing step budget. The implementation verifier likewise retries malformed or schema-invalid JSON once with the exact safe parser diagnostic and required key shape. A valid rejection supplies grounded findings and required changes to Trae's bounded implementation-revision loop; a still-malformed verifier response remains a verifier protocol failure rather than evidence that candidate code is defective. An unsuccessful coding trajectory retains a redacted trajectory and process log when available, provider token usage, and elapsed wall time in `adapter.failed`. Generated experiment reports expose these failure and recovery records without changing ledger authority.

The pinned Trae Docker manager is patched during setup to use bounded stateless
Docker exec rather than an interactive host pseudo-terminal. The same reviewed
path works through Docker Desktop's Windows named pipe and macOS/Linux Unix
socket, always starts commands in `/workspace`, converts host paths to POSIX
container paths, and applies an in-container timeout. Runtime identity checks
require this patch, while preflight verifies the timeout utility and read-only
tool mount before any research ledger is created. The isolated host-side Trae
process also forces Python UTF-8 mode and UTF-8 standard streams, preventing
Windows legacy code pages from crashing on Rich status glyphs while preserving
the same deterministic environment on macOS and Linux.

Resource failures are typed and fail closed. OOM and hard memory-limit signals may use an allowlisted runtime adjustment; disk-full, `ENOSPC`, output-quota, and storage-floor signals abandon the experiment without retrying or asking the coding worker to make an unrelated patch. The controller records the stable reason code and an operational lesson, so an operator can reclaim space or choose a fresh runtime before resuming. Evidence and prior run directories are never deleted automatically.

Self-debugging is a bounded plan-before-edit handoff. Trae receives the original hypothesis and mechanism, accepted commit, failure class and fingerprint, safe evidence, prior attempts, target files, contract boundaries, and remaining repair budget. Its instructions require a brief diagnosis and concise repair plan before editing, followed by the smallest scoped patch and the existing Gate A/smoke check. The controller still treats the worker as untrusted: it does not accept a claimed plan as proof, and only Gate A plus the subsequent execution/evaluation results can authorize the next recovery transition.

## Finalization

After `run.stopped`, development proposals are illegal. If a candidate is best, the controller:

1. resolves its exact accepted Gate A receipt and trusted best-score event;
2. runs `clean_reproduce` on full public validation with the selected commit and seed;
3. requires Gate B, clean trust, and exact reproduction of the recorded best score;
4. runs `candidate_final_infer` against the label-free official test view;
5. applies the test-row Gate B contract;
6. runs `submission_check` against that exact accepted artifact; and
7. appends `final.selected` and `submission.checked`.

If the official FM baseline remains best, a protected provider validates and copies the manifest-attested FM test submission. The candidate never receives test labels, and final test output never flows back into planning or convergence.

## Setup and commands

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps -e .

export DEEPSEEK_API_KEY='your-key'
.venv/bin/tacorank setup-live --download-data
.venv/bin/tacorank preflight \
  --config .tacorank/deployment/run-config.json \
  --live-config .tacorank/deployment/live-adapters.json
.venv/bin/tacorank run \
  --config .tacorank/deployment/run-config.json \
  --live-config .tacorank/deployment/live-adapters.json
```

`preflight` verifies the clean baseline, contract, protected paths, official submodule/evaluator/baseline, every manifest file, Trae runtime and model access, Docker image/environment, edit-tool isolation, and output quota without creating a ledger.

Operational commands:

```bash
tacorank resume --config RUN_CONFIG --live-config LIVE_CONFIG
tacorank finalize --config RUN_CONFIG --live-config LIVE_CONFIG
tacorank status --run-id RUN_ID --repository-root .
tacorank validate-ledger --run-id RUN_ID --repository-root .
tacorank rebuild-views --run-id RUN_ID --repository-root .
```

`resume` repairs only an incomplete final JSONL fragment, verifies that both configurations reproduce the frozen run identity, then continues from `planning` or `planner_context`. A malformed complete event is corruption. A crash during an external adapter call leaves an ambiguous intermediate phase and fails closed for operator review; TacoRank does not invent an adapter result. `finalize` is an explicit retry for an already stopped run and is idempotent after successful finalization.

Production exposes no fake runtime flag. Test doubles are constructed directly by tests.

## DeepSeek and credential boundary

The researcher and pinned Trae worker use `deepseek-v4-flash` with high reasoning through separate bounded adapters. `SearchPolicy` still owns the parent, family, phase, and reviewed method card. Once that method is selected, the live planner sends only a static method-derived query to the keyless OpenAlex Works API and snapshots a bounded set of papers; no dataset rows, metrics, run identifiers, or user identifiers leave the controller. DeepSeek receives the research policy, method overviews, experiment feedback, lessons, budget, and untrusted paper snapshot, then must cite an exact retrieved evidence ID in its code-blind hypothesis and intervention. It cannot invent or alter citations. DeepSeek does not receive repository paths, implementation interfaces, commit lineage, pipeline stages, commands, or the execution ladder. After validation, the deterministic controller binds the authorized code targets and frozen smoke/proxy/full ladder before Trae receives the coding context. Candidate coding and execution remain network-disabled. Invalid provider output is recorded at a durable planner checkpoint and raises a resumable error rather than becoming false convergence.

The API key is read only from the configured environment variable. It must never appear in configuration, prompts, Git, logs, trajectories, fixtures, or artifacts.

## Ownership matrix

| Area | Controller owns | Role component owns |
| --- | --- | --- |
| Research | Context, validation, routing, budgets | Hypothesis and bounded plan |
| Coding | Proposal/commit identity and receipt requirement | Trae edit trajectory and patch bytes |
| Execution | Legal request and event lifecycle | Isolation, process control, telemetry, artifacts |
| Recovery | Retry/repair budgets and routing | Classification and typed recovery choice |
| Evaluation | Legal handoff, best update, convergence | Metrics, trust, decision, reflection lesson |
| Finalization | Selected commit and sealed command order | Protected metric/submission validation |

## Verification and evidence boundary

```bash
python -m pytest -q
```

The integration suite deterministically exercises multi-iteration memory feedback, convergence, planner blocking, fatal-integrity recovery, candidate and baseline finalization, replay, and illegal transitions. Live DeepSeek, Docker, official-data CPU execution, and elapsed multi-experiment convergence remain deployment checks; never infer their current availability from mocked tests or from an earlier acceptance receipt.

## Schema changes

1. Change `src/tacorank/schemas.py`; do not create adapter-local duplicate models or enums.
2. Keep schema `1.0` only for backward-compatible additions with replay-safe defaults.
3. Update every affected valid/invalid fixture and cross-component test.
4. Run the full suite and replay a representative ledger before integration.
