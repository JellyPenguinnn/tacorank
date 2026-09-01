```json
{"schema_version":"1.0","method_id":"model_lgbm_xendcg","family":"model","status":"candidate","tags":["lightgbm","xendcg","ranking","model"],"cost_tier":"medium","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":["https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree"]}
```

## Mechanism

Train LightGBM with the rank_xendcg objective on the same train-split-only
causal feature frame as model_lgbm_causal_history. XE-NDCG is a listwise
cross-entropy surrogate for NDCG that in the sibling lab study tied or
slightly beat lambdarank on this exact data (0.6133 valid single seed, its
best single number) and converged in fewer boosting rounds. Its loss shape
differs enough from lambdarank to make it a genuinely diverse blend member.

## Preconditions

Best used to deepen an accepted model_lgbm_causal_history parent: keep the
parent's feature frame and swap only the objective. Executable FM parity is
verified.

## Allowed data

Same rules as model_lgbm_causal_history: every feature computable from the
train split alone; score-population rows never enter any aggregate.

## Expected effect

Offline compliant replication (2026-08-31): 0.59747 standalone, 0.60309 as
a z-scored residual at alpha 0.7 on the FM parent. Value is diversity for
blending, not standalone strength; implement as a blend residual.


Match or slightly beat the lambdarank member standalone, and add loss-shape
diversity worth real blend gain (the sibling study's diverse blend reached
0.6172 valid versus 0.6133 best single).

## Falsification condition

No trusted improvement over the lambdarank parent standalone AND no gain
when blended with it.

## Do not use when

No causal-frame parent exists yet (build model_lgbm_causal_history first),
or the evaluator, split, or scored population would need to change.

## Minimal implementation

Identical frame, groups, and small-capacity settings as the accepted
parent; change objective to rank_xendcg, keep label_gain [0,1], learning
rate 0.05, num_leaves 7, min_data_in_leaf 200, lambda_l2 10, feature and
bagging fraction 0.85, num_threads=8 (the full frozen container quota —
never 1), seed bound to invocation.seed, raw-score prediction, finite
scores. Roughly 300 rounds (it converges faster than lambdarank). Train at
full strength for every fidelity.

## Sources

Bruch et al., "An Alternative Cross Entropy Loss for Learning-to-Rank"
(XE-NDCG). Measured in the sibling lab study (lab/PLAYBOOK.md).
