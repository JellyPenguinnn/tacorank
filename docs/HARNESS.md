# TacoRank deterministic harness

TacoRank is the integration backbone for the five-person autonomous KuaiRand-Pure research agent. The controller is deterministic and event-sourced; researcher, coding, safety, execution, recovery, and evaluation components return typed values but cannot mutate shared state or choose their own authority.

## End-to-end control flow

```text
frozen contract + official baseline
                |
                v
ledger-derived research feedback -> DeepSeek researcher -> ResearchProposal
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

`Harness.run_until_stopped()` executes one experiment at a time. Each terminal decision and lesson is appended before the next planner context is built, so later proposals consume durable evidence rather than in-memory messages. `Harness.run_to_completion()` adds finalization after a frozen convergence, experiment, wall-time, token, GPU, integrity, or no-legal-proposal stop.

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

- Gate A binds the exact patch commit and diff to the contract, data identity, allowed files/imports/dependencies, and verification receipt.
- The runner resolves reviewed symbolic commands; it never executes an LLM-supplied shell string.
- Docker execution is CPU-only, network-disabled, receipt-sealed, resource-bounded, and observed by Person 4.
- Gate B verifies the prediction schema, ordered row identity, finite/diverse scores, producing commit, data manifest, command, and execution seal before evaluation.
- Smoke can promote on structural success. Proxy and full require protected evaluation. Only a trusted public-validation full result can update the best.

`setup-live` gives every candidate route the exact official FM prediction plus its digest and proves the checked-in candidate reproduces it byte-for-byte. This makes the executable parent equal to the scored baseline. Candidate experiments normally preserve that parent and add one bounded train-only residual. Evaluation adds label-free rankability, personalization, residual-scale, and FM-correlation diagnostics to the canonical result and planner context; those diagnostics never replace protected metrics.

Execution, Gate A, Gate B, and evaluation no-op failures enter the bounded recovery route. Recovery may retry the same commit once, apply an approved runtime adjustment, ask Trae for at most two repair patches, roll back, or abandon. Every replacement patch needs a new Gate A receipt. Repeated fingerprints stop repair cycling, and deliberate credential, hidden-label, target-label, or network boundary violations are recorded and terminate the run.

The pinned DeepSeek Responses client catches malformed or truncated function arguments before Trae executes them and asks for one smaller valid JSON call inside the existing step budget. An unsuccessful coding trajectory retains a redacted trajectory and process log when available, provider token usage, and elapsed wall time in `adapter.failed`. Any failure that remains after the in-loop correction is routed through Waihong's bounded self-recovery policy; only classified transient coding failures receive its one same-commit retry, and all other abandon or stop outcomes remain policy-owned. Generated experiment reports expose these failure and recovery records without changing ledger authority.

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

The researcher and pinned Trae worker use `deepseek-v4-flash` with high reasoning through separate bounded adapters. `SearchPolicy` still owns the parent, family, phase, and reviewed method card. DeepSeek receives only research policy, method overviews, experiment feedback, lessons, and budget, then returns a code-blind hypothesis and intervention. It does not receive repository paths, implementation interfaces, commit lineage, pipeline stages, commands, or the execution ladder. After validation, the deterministic controller binds the authorized code targets and frozen smoke/proxy/full ladder before Trae receives the coding context. Invalid provider output is recorded at a durable planner checkpoint and raises a resumable error rather than becoming false convergence.

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
