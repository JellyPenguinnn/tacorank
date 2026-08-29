# TacoRank

TacoRank is an autonomous recommender-system research harness for KuaiRand-Pure. This repository currently defines architecture and ownership paths only; the modules are intentionally empty.

## Architecture

```text
Context + orchestration (Person 2)
  -> research planning (Person 1)
  -> coding, safety gates, and execution (Person 3)
  -> monitoring and recovery (Person 4)
  -> evaluation, trust, and reflection (Person 5)
  -> verified events back to the orchestrator
```

The orchestrator owns run state, budgets, convergence, final selection, and the append-only event ledger. Candidate code lives under `solution/`; protected contracts, evaluator boundaries, and hidden-test data must remain outside its editable surface.

## Repository layout

```text
contract/                    Frozen competition contract
src/tacorank/
  agents/ providers/ research/   Research planning
  memory/ orchestrator/ context/ Control plane and context assembly
  coding/ git/ safety/ execution/ Coding and trusted execution
  sre/ recovery/                 Monitoring and recovery
  evaluation/ reflection/ reporting/ Evaluation and evidence
benchmarks/kuairand_pure/     Dataset-specific evaluator adapters
research/methods/             Research method cards
solution/                     Agent-editable candidate pipeline
runs/                         Generated run ledger and projections
artifacts/                    Generated experiment artifacts
tests/                        Component, contract, and integration tests
```

## Status

Skeleton only. No research, orchestration, execution, recovery, or evaluation logic has been implemented yet.

## Official KuaiRand-Pure starter kit

The official baseline, data loader, evaluator, submission utility, feature ablation, published scores, dataset loader, and dataset license are included. See [`docs/KUAIRAND_STARTER_KIT.md`](docs/KUAIRAND_STARTER_KIT.md) for setup and usage.

Downloaded CSV data belongs in `KuaiRand-Pure/data/`. Git ignores only that dataset directory and the downloaded `KuaiRand-Pure.tar.gz` archive.
