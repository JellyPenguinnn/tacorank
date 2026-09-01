# Method card sources

Where each mechanism in the portfolio comes from, and what was deliberately
left out of scope.

**How this reaches the agents.** It does not, directly. Nothing in
`src/tacorank/` reads `context/papers/` or this file; those are human
reference. The planner's research knowledge arrives through the method cards
themselves — their `mechanism`, `preconditions`, `expected effect`,
`falsification condition`, and `minimal implementation` sections — plus the
playbook and the data profile. So a paper only influences a run once its
mechanism is written into a card. Adding a PDF somewhere is not enough.

**Scope discipline.** Every card keeps the frozen FM parent and adds one
bounded residual. Papers are therefore adopted in part: the mechanism under
test transfers, the surrounding architecture usually does not. Each card states
its own exclusion, and those exclusions matter — they are what keeps one
experiment to one mechanism.

## Objective

| Card | Source |
| --- | --- |
| `objective_pairwise_bpr` | BPR-style within-user pairwise ranking. No external source required for the bounded baseline trial. |
| `objective_lambdarank_ndcg` | Burges, *From RankNet to LambdaRank to LambdaMART: An Overview*, Microsoft Research MSR-TR-2010-82. Weight each pair by the ΔnDCG of swapping it. **Out of scope:** the LambdaMART boosted-tree ensemble. |
| `objective_listwise_user_softmax` | Cao et al., ListNet-style listwise learning to rank, ICML 2007. |

## Duration bias

| Card | Source |
| --- | --- |
| `duration_bias_censored_watch_time` | Duration-aware calibration of a globally fitted score. |
| `duration_bias_quantile_deconfounded` | Zhan et al., *Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation*, KDD 2022, [arXiv:2206.06003](https://arxiv.org/abs/2206.06003). Fit within duration quantile groups so exposure bias is removed and intrinsic effect kept. Deployed at Kuaishou, which is the source platform for KuaiRand-Pure. **Out of scope:** the watch-time target; `duration_ms` is video duration, not observed watch time. |

## Temporal history

| Card | Source |
| --- | --- |
| `temporal_history_compact` | Compact, candidate-agnostic past-only history summary. |
| `temporal_history_target_attention` | Zhou et al., *Deep Interest Network for Click-Through Rate Prediction*, KDD 2018, [arXiv:1706.06978](https://arxiv.org/abs/1706.06978). Weight each past interaction by its relatedness to the candidate, so the user representation varies within a list. **Out of scope:** the deep architecture, mini-batch aware regularizer, and Dice activation. |

## Features

| Card | Source |
| --- | --- |
| `temporal_drift_past_only` | KuaiRand dataset paper and the local chronological split. |
| `features_list_context_relative` | Pei et al., *Personalized Re-ranking for Recommendation*, RecSys 2019, [arXiv:1904.06813](https://arxiv.org/abs/1904.06813). Score an item by its position within the candidate list, which a per-pair model cannot see. **Out of scope:** the transformer encoder. |

## Model

| Card | Source |
| --- | --- |
| `model_compact_ranker` | Compact ranker as a bounded model-family change. |
| `model_stacked_cross_residual` | Wang et al., *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems*, WWW 2021, [arXiv:2008.13535](https://arxiv.org/abs/2008.13535). Explicit bounded-degree feature crossing above a second-order parent. **Out of scope:** the full serving architecture and low-rank production variants. |

## Why these mechanisms

Measurement on this deployment shows the frozen FM parent is at the ceiling of
its feature set: an FM retrained from scratch on the same five fields scores
below it, and residuals that reorder 0.75–1.4% of within-user pairs still net
out to roughly zero. Gains therefore have to come from information the parent
cannot represent, not from re-fitting what it already knows. Three cards target
that directly and are the ones to watch:

- `features_list_context_relative` — the parent scores each pair independently
  and cannot see the candidate set.
- `temporal_history_target_attention` — the parent gives each user one
  embedding, so its user side is constant across that user's list.
- `model_stacked_cross_residual` — the parent is second order and cannot
  represent higher-degree interactions.

## Not yet carded

A residual blend weight, of the kind that turns an accepted residual into a
smaller shrunk one, is a single bounded coefficient rather than an open-ended
sweep and has been observed to compound in practice. It is absent because it
needs a routing decision: it operates on one prior experiment, which is the
ensemble contract, but the ensemble route currently names its preferred card
directly. Worth adding deliberately rather than by accident.
