# Protected Paths Policy

> **Owner:** Human team  
> **Mode:** Immutable policy; all paths are repository-relative.

## Access rules

| Path pattern | Mode | Authorized writer |
|---|---|---|
| `COMPETITION.md` | Immutable | Human only |
| `PROTECTED_PATHS.md` | Immutable | Human only |
| `memory/contracts/**` | Immutable | Human only |
| `memory/schemas/**` | Immutable | Human only |
| `starter_kit/**` | Immutable | Human only |
| `vendor/kuairand-starter-kit/**` | Immutable | Human only |
| `data/raw/**` | Immutable | Human/ingestion only |
| `data/splits/**` | Immutable | Human/ingestion only |
| `data/test/**` | Immutable | Human/ingestion only |
| `data/hidden_labels/**`, `data/test_labels/**` | Deny all | No agent |
| `memory/events/*.jsonl` | Append-only | Ledger writer |
| `runs/*/events.jsonl` | Append-only | Ledger writer |
| `runs/*/metrics.jsonl` | Append-only | Evaluator |
| `memory/run_state.json` | Atomic replace | Orchestrator |
| `memory/summaries/**` | Atomic replace | Memory manager |
| `outputs/final/**` | Create once | Orchestrator/release step |
| `.git/**` | Git operations only | Harness |

`src/**`, `configs/experiments/**`, `data/processed/**`, and worktree-local outputs are editable unless another rule protects them.

## Enforcement

- Resolve canonical paths before every read or write; reject traversal and symlink escapes.
- Reject an entire patch if it touches an immutable, deny-all, or unauthorized managed path.
- For append-only files, existing bytes must remain unchanged; only valid JSONL records may be added.
- Do not overwrite, delete, rename, or truncate protected files.
- A protection exception requires an explicit human commit that updates this policy first.

## Machine-readable protected roots

The harness reads the following path-only bullets as the enforced minimum manifest:

- `PROTECTED_PATHS.md`
- `contract/`
- `runs/`
- `src/tacorank/memory/`
- `src/tacorank/orchestrator/`
- `src/tacorank/safety/`
- `kuairand-starter-kit/data.py`
- `kuairand-starter-kit/baseline.py`
- `kuairand-starter-kit/evaluate.py`
- `kuairand-starter-kit/submit.py`
- `kuairand-starter-kit/baseline_scores.json`
