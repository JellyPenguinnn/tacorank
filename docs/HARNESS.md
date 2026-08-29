# TacoRank deterministic harness

This is the P0 integration backbone for the five-person TacoRank agent. It is a
working vertical slice, not a replacement for the Planner, Trae worker, safety,
recovery, or evaluator implementations owned by the other team members.

## What is implemented

- One strict Pydantic v2 schema module and one enum set for all adapter boundaries.
- Content-addressed artifact references with approved-root, symlink, size, and hash checks.
- Compact canonical JSONL events with contiguous IDs, causal links, idempotency keys,
  file locking, `flush`/`fsync`, and a SHA-256 hash chain.
- Crash recovery that truncates only an incomplete final fragment. A malformed complete
  line is corruption and is never silently repaired.
- Pure replay projections for run state, experiment lineage, lessons, resource totals,
  and generated Markdown views. There is no mutable state database.
- A deterministic legal-transition validator. It prevents execution without Gate A,
  evaluation without Gate B, untrusted/proxy best selection, hidden-final development
  feedback, repair overflow, and unbounded full-fidelity confirmation.
- A role-specific context compiler for Planner, Coder, and Recovery. It separates
  instructions from untrusted evidence, redacts secrets, excludes hidden-final evidence,
  applies stable ordering, enforces hard token budgets, and persists immutable context
  artifacts with inclusion/exclusion manifests.
- Deterministic convergence and resource stop checks.
- Replaceable adapter protocols and deterministic fakes that complete one full lifecycle.
- Operator commands for run, resume inspection, status, ledger validation, view rebuild,
  and a fail-closed finalization boundary.

## Authority and state flow

```text
Human contract (read only)       Git (code lineage)
             \                    /
              \                  /
               EventStore.append
                      |
             runs/<id>/events.jsonl
                      |
              validate + replay
          ____________|____________
         |             |            |
      RunState    ExperimentNode   Lessons
         |             |            |
         +------ ContextBuilder ----+
                      |
        Planner / Coder / Recovery adapters
```

Only the harness receives raw adapter values and appends events. `STATUS.md`,
`LESSONS.md`, `SUMMARY.md`, and files under `contexts/` are derived artifacts.

## State transitions

```text
proposed -> patch_ready -> ready_to_run -> running
    ^            |                            |
    |            v                            v
    +------ recovering <- failure ------ output_ready
             |                              |
             +-> invalid                    v
                                      output_verified
                                             |
                                             v
                                         evaluated
                                             |
                         +-------------------+-------------------+
                         v                   v                   v
                      accepted            rejected             pruned
```

Smoke output can promote without a metric. Proxy and full stages require a typed
evaluation. Only a trusted public-validation full result can become best eligible.

## Setup and verification

```text
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONPYCACHEPREFIX=/tmp/tacorank-pycache .venv/bin/python -m pytest
.venv/bin/python -m pip install -e .
```

The editable install exposes the `tacorank` command.

## Contract gate

The checked-in `contract/COMPETITION.md` and `PROTECTED_PATHS.md` are intentionally
empty scaffolds. The harness will not edit them and will refuse to start until humans:

1. resolve the label and metric conflict;
2. add exact comma-separated `Allowed command IDs:` and `Artifact roots:` lines;
3. add the exact line `Contract status: FROZEN`;
4. replace the two placeholder hashes in `config.example.json`; and
5. copy the example to a run-specific configuration file.

For example:

```text
Allowed command IDs: run_smoke, run_proxy, run_full
Artifact roots: artifacts, runs
Contract status: FROZEN
```

This is deliberate. A prompt or adapter cannot waive the contract gate.

## Commands

```text
tacorank run --config run-config.json
tacorank resume --run-id run_001 --repository-root .
tacorank status --run-id run_001 --repository-root .
tacorank validate-ledger --run-id run_001 --repository-root .
tacorank rebuild-views --run-id run_001 --repository-root .
tacorank finalize --run-id run_001 --repository-root .
```

`adapter_mode` is currently frozen to `fake`; this lets teammates validate their
schemas and the complete route before integrating real adapters. `finalize` fails closed
until a real clean-reproduction runner/evaluator is connected, because the harness must
not manufacture final evidence.

## Schema-change procedure

1. Change `src/tacorank/schemas.py`; do not create an adapter-local duplicate enum.
2. Update every affected valid and invalid fixture.
3. Update all affected component contract tests in the same change.
4. Keep `schema_version` at `1.0` for backward-compatible additions only. Use a new
   version and an explicit migration/replay decision for incompatible changes.
5. Run the complete suite and validate a real copied ledger before merge.

## Ownership matrix

| Area | Harness owns | External adapter owns |
| --- | --- | --- |
| Research | Validation, append, routing, context | Hypothesis and parent quality |
| Coding | Patch identity, receipt requirement, lineage | Trae prompt, worktree, patch bytes |
| Execution | Legal request and event lifecycle | Sandbox, process, telemetry collection |
| Recovery | Repair count and route | Failure classification and repair choice |
| Evaluation | Legal handoff, trust gating, best route | Metrics, trust verdict, research reflection |

## Intentionally deferred

- Real adapters for Persons 1, 3, 4, and 5.
- Git ancestry/ref reconciliation and clean reproduction.
- Final submission generation and hidden-final execution.
- Provider-native token collection and real CPU/GPU telemetry.
- Optional lexical/vector retrieval beyond deterministic typed retrieval.

Those items need their owner implementations or real evidence. The P0 harness exposes
the stable contracts they plug into without speculating about their internal behavior.
