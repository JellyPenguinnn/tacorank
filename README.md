# TacoRank

TacoRank is a deterministic, evidence-tracked harness for autonomous recommender-system research on KuaiRand-Pure. It coordinates research planning, code generation, guarded execution, failure recovery, evaluation, and reflection without giving any LLM control over shared state, safety policy, or metric truth.

The repository is organized for five developers with one canonical schema and explicit typed handoffs. The controller owns orchestration; role components return values and never compete for control.

> **Current main status:** All five role implementations are merged and the production CLI composes the real Trae, Gate A, Docker execution, SRE/recovery, Gate B, and protected-evaluation adapters. Deterministic fakes require an explicit test-only mode. The integrated test suite is green, while live autonomous training remains fail-closed until humans freeze the contract, protected paths, data, baseline predictions, runtime identities, credentials, and output-quota filesystem.

## System architecture

```mermaid
flowchart LR
    H[Human-frozen contract] --> O[Person 2<br/>deterministic harness]
    O --> P1[Person 1<br/>research planning]
    P1 -->|PlannerOutput / ExperimentSpec| O
    O --> P3[Person 3<br/>Trae, Git, gates, execution]
    P3 -->|TelemetrySample| P4[Person 4<br/>health and recovery]
    P4 -->|MonitorDirective / RecoveryDecision| O
    P3 -->|Patch and run results| O
    O --> P5[Person 5<br/>evaluation and reflection]
    P5 -->|EvaluationResult / decision| O
    O --> E[(events.jsonl)]
    O --> G[(Git lineage)]
    O --> A[(hash-addressed artifacts)]
    E --> C[role-specific contexts]
    C --> O
```

The three authorities are:

- human rules in `contract/COMPETITION.md` and `PROTECTED_PATHS.md`;
- dynamic evidence in the append-only `runs/<run_id>/events.jsonl` ledger; and
- exact code lineage in Git.

Generated status, lesson, summary, context, and chart files are derived views—not independent sources of truth.

## Ownership and implementation status

| Role | Responsibility | Main paths | Current state |
| --- | --- | --- | --- |
| Person 1 | Evidence-grounded experiment planning and deterministic search policy | `src/tacorank/agents/`, `src/tacorank/research/`, `src/tacorank/providers/`, `research/` | Implemented with fake and DeepSeek providers |
| Person 2 | Shared schemas, append-only memory, contexts, orchestration, budgets, replay, resume and CLI | `src/tacorank/schemas.py`, `memory/`, `context/`, `orchestrator/`, `artifacts.py`, `cli.py` | Canonical harness and production composition implemented |
| Person 3 | Trae coding worker, Git worktrees, Gate A, sandboxed execution, telemetry, artifacts and Gate B | `src/tacorank/coding/`, `git/`, `safety/`, `execution/` | Implemented and wired into production mode |
| Person 4 | SRE monitoring, failure classification, bounded recovery and operational reflection | `src/tacorank/sre/`, `src/tacorank/recovery/` | Implemented and failure-injection tested |
| Person 5 | Protected evaluation, trust decisions, stability, final selection and research reflection | `src/tacorank/evaluation/`, `reflection/`, `reporting/`, `benchmarks/kuairand_pure/` | Implemented and covered by evaluator, reflection and reporting tests |

Only Person 2 appends events or changes controller state. Person 1 chooses research direction, Person 3 edits and executes candidate code, Person 4 interprets operational failures, and Person 5 owns metric truth.

## Repository layout

```text
src/tacorank/
  agents/            Person 1 planner adapter
  providers/         fake/provider-specific research model clients
  research/          search policy, experiment graph and method portfolio
  memory/            append-only event store and replay
  context/           bounded role-specific context construction
  orchestrator/      deterministic state machine and adapter routing
  coding/            pinned Trae adapter, prompts and redaction
  git/               experiment refs, patches and disposable worktrees
  safety/            protected manifests, Gate A and Gate B
  execution/         symbolic commands, sandbox, telemetry and artifacts
  sre/               live health observation
  recovery/          classification and bounded recovery policy
  evaluation/        protected metrics, trust and experiment decisions
  reflection/        evidence-linked research lesson generation
  reporting/         reproducible derived views
solution/             only candidate area intended for coding-agent edits
research/methods/     reviewed experiment method cards
tests/                unit, integration and failure-injection coverage
contract/             human-frozen competition contract
runs/                 ignored run evidence and derived views
artifacts/            ignored content-addressed run artifacts
kuairand-starter-kit/ starter-kit Git submodule
```

## Setup

TacoRank supports Python 3.9+. Initialize the starter-kit submodule, create an environment, install dependencies, and install the package:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

Run the complete suite:

```bash
python -m pytest -q
```

Useful component-level checks:

```bash
# Persons 1 and 2
python -m pytest tests/research tests/schemas tests/memory tests/context tests/orchestrator

# Person 3
python -m pytest tests/coding tests/git tests/safety tests/execution tests/failure_injection

# Person 4
python -m pytest tests/sre tests/recovery tests/integration/test_recovery_lifecycle.py

# Person 5
python -m pytest tests/evaluation tests/reflection tests/reporting
```

### Provider keys and Trae coding worker

No API key is needed for the test suite or a fully fake run. Provider keys are never stored in configuration files:

- `research_provider: "fake"` uses the deterministic Person 1 test planner and needs no key.
- `research_provider: "deepseek"` makes only Person 1 planning real and requires `DEEPSEEK_API_KEY` in the process environment.
- `adapter_mode: "live"` selects the reviewed Trae configuration and requires the provider credential names explicitly approved by `live-adapters.json`.

For live integration, install the pinned Trae worker in a separate Python 3.12+ environment:

```bash
python3.12 -m venv .venv-trae
.venv-trae/bin/python -m pip install -r requirements-trae.txt
```

Start from `config/trae-agent.yaml.example`, select a reviewed provider/model, and keep the final credential-free configuration outside Git. Bind the configuration, executable, installation, runtime manifest, and Docker image identities in `live-adapters.json`. The example uses OpenAI and expects `OPENAI_API_KEY` only through the approved process environment. Never place credentials in this repository, YAML, prompts, logs, trajectories, fixtures, or artifacts.

## KuaiRand-Pure starter kit and data

Starter resources are tracked in `KuaiRand-Pure/` and through the `kuairand-starter-kit` submodule. The dataset itself is intentionally excluded from Git. Obtain it separately and follow [`docs/KUAIRAND_STARTER_KIT.md`](docs/KUAIRAND_STARTER_KIT.md) for the official split, baseline, evaluation, and submission contracts.

Do not change the protected evaluator, split logic, submission checker, or published baseline evidence from candidate code.

## Running the harness

The checked-in `contract/COMPETITION.md` and `PROTECTED_PATHS.md` are intentionally empty and fail closed. Before any run, humans must resolve the competition discrepancy, populate and freeze both files, verify the data/evaluator hashes, and copy both `config.example.json` and `live-adapters.example.json` to run-specific configurations.

Production mode requires the live adapter configuration whose SHA-256 is frozen in the run configuration:

```bash
tacorank run --config run-config.json --live-config live-adapters.json
tacorank resume --run-id run_001 --repository-root .
tacorank status --run-id run_001 --repository-root .
tacorank validate-ledger --run-id run_001 --repository-root .
tacorank rebuild-views --run-id run_001 --repository-root .
tacorank finalize --run-id run_001 --repository-root .
```

For deterministic tests only, set `adapter_mode` to `fake` and pass `--allow-test-adapters`; production mode never falls back to it. `resume` currently validates/repairs the ledger tail and reports the recovery phase without restarting adapter execution. `finalize` deliberately refuses to fabricate success because the standalone clean-reproduction/final-selection command is not implemented yet.

## Core guarantees

- All role boundaries use the canonical models in `src/tacorank/schemas.py`.
- Only the deterministic harness writes the event ledger or owns budgets, routing, convergence and final selection.
- Candidate execution requires an exact Gate A receipt bound to the commit, diff, contract and data identities.
- The runner resolves reviewed symbolic commands; it does not accept raw LLM shell commands.
- Candidate code cannot access protected evaluator labels or iterative hidden-final feedback.
- Person 3 owns termination mechanics; Person 4 returns typed health and recovery decisions.
- Gate B validates prediction structure and producer identity before Person 5 evaluation.
- Expected failures return typed results with redacted, hash-addressed evidence.
- Repair attempts and same-commit retries are bounded to prevent recovery loops.

## Collaboration rules

1. Import shared models from `src/tacorank/schemas.py`; do not create component-local replacements.
2. Preserve role ownership and communicate through typed adapter interfaces.
3. Never modify frozen contracts, protected paths, evaluator logic, data splits, or event history from candidate code.
4. Update affected fixtures and cross-component tests with every shared-schema change.
5. Keep datasets, credentials, submissions, model artifacts, sensitive run ledgers and local environments out of Git.
6. Run the complete test suite before requesting integration review.

## Current limitations

- `solution/` does not yet contain the real candidate training pipeline.
- Live Trae, production Docker, GPU and full-data runs cannot be validated until the human-owned deployment inputs are frozen and available.
- GPU commands fail closed until the execution backend can prove a hard per-container GPU-memory limit.
- Human-owned contract, protected-path, data-manifest, evaluator and command-profile inputs must be finalized before live execution.
- The standalone `resume` and `finalize` commands do not yet restart adapter execution or perform clean-reproduction final selection.

## Further documentation

- [`docs/HARNESS.md`](docs/HARNESS.md) — deterministic control plane, event flow and schema-change procedure
- [`docs/person3-handoff.md`](docs/person3-handoff.md) — coding, Git, safety and execution integration contract
- [`docs/research/planning-and-search.md`](docs/research/planning-and-search.md) — planning and search boundary
- [`docs/KUAIRAND_STARTER_KIT.md`](docs/KUAIRAND_STARTER_KIT.md) — dataset, baseline, evaluation and submission details
- [`TacoRank-Memory-Schema-v1.md`](TacoRank-Memory-Schema-v1.md) — event and memory schema reference
- [`research/CURRENT_RUN_IMPROVEMENT_PLAN.md`](research/CURRENT_RUN_IMPROVEMENT_PLAN.md) — reviewed initial research directions
