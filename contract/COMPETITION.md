# Competition Contract

> **Owner:** Human team  
> **Mode:** Immutable contract memory; agents may read but must not edit.

## Mission

Build an autonomous ML research agent for the required **KuaiRand-Pure** benchmark. It must reproduce the official baseline, then autonomously propose, implement, run, evaluate, recover, and reflect until convergence or the resource budget is reached. The final artifact is the validation-best checkpoint at that point.

## Non-negotiable rules

1. Develop with the official train and validation splits only. Never access or infer hidden-test labels.
2. Do not use external training data or weights trained on benchmark test labels. Public papers, code, libraries, and otherwise permitted pretrained weights are allowed.
3. Treat the current Starter Kit's split, evaluator, baseline, convergence rule, and submission checker as the executable specification. If organizer materials disagree, stop and record a contract conflict for human resolution.
4. Complete KuaiRand-Pure before attempting the optional KuaiRand-1k or KuaiRand-27k bonuses.
5. Keep persistent agent memory in **Markdown, JSONL, and Git only**; no database or vector database.
6. Run experiment changes in an isolated Git worktree. Never modify protected paths.
7. For every iteration, append: hypothesis, parent experiment/commit, code diff, command, evaluator output, status, errors/recovery, token usage, wall-clock, GPU-hours, and manual interventions.
8. Never invent, hand-edit, or selectively omit metrics. Only official evaluator output may determine experiment and final-checkpoint selection.

## Completion criteria

- Official baseline reproduced.
- Required benchmark reaches convergence or the fixed budget.
- Final checkpoint selected by validation score and submission checker passes.
- Run logs, recovery history, resource totals, and manual-intervention count are complete.
- Repository, reproduction steps, results summary, limitations, and team contributions are ready for submission.

Metrics: GAUC, nDCG@5, primary
Allowed command IDs: candidate_smoke, candidate_proxy, candidate_full
Artifact roots: artifacts, runs
Contract status: FROZEN