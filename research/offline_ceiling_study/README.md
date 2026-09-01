# Offline compliant-ceiling study (2026-08-31 -> 09-01)

Question: with the team's compliance ruling (every feature computable from
the TRAIN split alone; score-population rows never enter any aggregate),
how much headroom over the official FM baseline (valid primary 0.60147)
actually exists?

Method: direct replications and recombinations, all scored with the
untouched official `evaluate.py` on the full validation population from the
run deployments. Member predictions come from ledger-traceable run
candidates (commit hashes below) or from
`compliant_causal_replication.py` in this directory.

## Standalone replications (valid primary)

| recipe | score | note |
|---|---|---|
| official FM baseline | 0.60147 | reference |
| causal-history LambdaRank, leaves 7 (this dir) | 0.60114 | below parent standalone |
| same frame, rank_xendcg | 0.59747 | below parent |
| same frame, leaves 63 | 0.58528 | capacity hurts (matches lab tombstone) |
| CatBoost YetiRank, train-only features (team test) | ~= baseline | rolling eval-window history was the sibling study's edge, and it is non-compliant |
| run candidate exp_021 (run 205244Z, d8378efa) | 0.60385 | best single model |
| run candidate exp_019 (run 205244Z, c38e442d) | 0.60367 | |
| run candidate exp_006 (run 163548Z, 507170a8) | 0.60279 | |
| run candidate exp_001 (run 163548Z, c3c946bc) | 0.60290 | |

## Blends (per-user z-score, equal weight unless stated)

| combination | score |
|---|---|
| FM + 0.7*std_u(FM)*z_u(causal model) | 0.60344 |
| accepted blend card (in-run, best) | 0.60351 |
| z(exp_001)+z(exp_006)+z(replication) | 0.60411 |
| + z(exp_019) | 0.60457 |
| z(exp_021)+z(exp_006)+z(replication)+z(exp_019) | **0.60470** |

## Conclusion

Under the train-split-only rule the measured ceiling is ~0.6035 for a
single model chain and ~0.6047 with member ensembling. The autonomous runs
reached 0.60385 single-model (exp_021) and their members compose to
0.60470 - essentially the entire legally extractable headroom. The human
expert study's 0.6121 required eval-window rolling history, which the
compliance ruling forbids.
