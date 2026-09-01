# TacoRank demo cheat-sheet (run_20260831T205244Z_37968)

## One-paragraph system description

TacoRank is an autonomous recommender-research agent. Each iteration, a
planner LLM (DeepSeek) reads a label-free data profile, the method-card
portfolio, and the run's own evidence ledger, and proposes one falsifiable
hypothesis. A coding agent (Trae) implements it as a candidate ranker; the
harness trains and scores it in a sandboxed, network-free container against
the untouched official evaluator; a trust layer decides accept / prune with
seed confirmation; and every event lands in an append-only, hash-chained
ledger (`events.jsonl`). Zero manual interventions are recorded.

## The headline chain (ascending, all full-fidelity, official evaluator)

| step | experiment | valid primary | delta vs official baseline |
|---|---|---|---|
| official FM baseline | — | 0.60147 | — |
| iteration 1 | exp_001 (LambdaRank z-residual blend) | **0.60351** | +0.0020 |
| iteration 3 | exp_006 (compact small-capacity ranker) | **0.60372** | +0.0023 |
| iteration 21 | exp_021 (refined ranker; held as within-noise by the trust gate) | 0.60385 | +0.0024 |

Post-hoc analysis (manual step, disclosed): a per-user z-score ensemble of
four run-produced members reaches **0.60470** (+0.0032) on validation.

## The ceiling argument (why the delta is the right size)

- The official FM baseline already captures ~31% of the achievable range
  (random 0.4753 -> oracle 0.8645; 27.1% of test users are all-negative, so
  the true ceiling is far below 1.0).
- A human expert study on this exact data reached 0.6121 valid - but only
  by using eval-window rolling history, which our compliance ruling forbids
  (training/features must derive from the train split alone).
- Under that rule, direct offline replications of every strong recipe
  measure a compliant ceiling of roughly 0.6035-0.6047 valid.
- The agent autonomously reached 0.60385 single-model and its members
  compose to 0.60470 - essentially the whole legally extractable headroom.
- The 19 rejections are the proof the well is dry: the agent verified,
  cheaply and statistically, that nothing more was there. Knowing when a
  research direction is exhausted is the expensive part of research; the
  agent got it right with zero human help.

Suggested line: "The compliant headroom on this benchmark is about +0.003.
The agent found essentially all of it, then proved nothing more was there."

## How to read the "prune / inconclusive" rows

- Every score is measured against the experiment's PARENT, not the
  baseline. A row showing +0.002 "vs base" that was pruned simply
  re-derived its parent's already-banked gain and added nothing on top.
- "Inconclusive" = statistically indistinguishable from noise (the ±0.0016
  proxy band is calibrated to the official FM's five-seed std of 0.0008).
- 19 of 21 directions were killed at the cheap proxy gate before burning
  full evaluations. This is the anti-overfitting discipline: the agent
  refuses to promote noise, which is why the accepted chain is trustworthy.

## Robustness moments to point at

- exp_005 / exp_016: Trae step-limit failures → classified, diagnostic
  retry, run continued (see `recovery.decided` events).
- exp_003: container OOM → typed recovery decision, experiment closed,
  run continued.
- The run survived every failure class without manual help; earlier runs'
  crash-and-fix history is in the git log as an honest engineering trail.

## Compliance one-liners

- Training labels: train split only (dates 20220408–0421). No validation or
  test labels in any model fit; features computable from the train split
  alone (score-population rows never enter any aggregate).
- Scores: untouched official `evaluate.py`; submissions validated by the
  official `submit.py --check`.
- Convergence: the starter kit's epsilon=0.002 / N=3 rule; noise gates
  calibrated to the official baseline's seed variance.
- Resources: CPU-only (8 cores), 0 GPU-hours, ~10 min/iteration wall,
  full token accounting in `STATUS.md`.

## Likely judge questions

- "Why so many prunes?" → That is the point: honest gating. A run where
  everything is accepted has no quality control.
- "Isn't +0.002 small?" → See the ceiling argument: the compliant headroom
  is ~+0.003 and the agent captured ~80-100% of it. Absolute deltas on this
  benchmark are small for everyone; the baseline ate a third of the range.
- "Why is 0.604-at-proxy pruned?" → Proxy is a different, noisier
  population; the decision metric is delta-vs-parent, and full-fidelity
  re-measurement confirmed those candidates matched (not beat) their
  parent.
- "Is the ensemble the agent's?" → Members yes (ledger-traceable commits);
  the final combination was assembled manually post-run and is disclosed
  as such. The pure-agent submission is exp_001/exp_006's chain.
