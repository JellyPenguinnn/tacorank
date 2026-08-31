```json
{"schema_version":"1.0","method_id":"model_catboost_yetirank","family":"model","status":"candidate","tags":["catboost","yetirank","categorical","replacement_capable","model"],"cost_tier":"medium","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":["https://arxiv.org/abs/1706.09516"]}
```

## Mechanism

Train CatBoost with the YetiRank pairwise-ranking loss on the same causal
feature frame as model_lgbm_causal_history. CatBoost's ordered target
statistics handle high-cardinality categorical ids natively and without
target leakage, which is exactly where LightGBM's categorical splitting is
weakest on this data. In the sibling lab study CatBoost YetiRank was the
strongest single model measured: 0.6120 test primary standalone (depth 6,
learning rate 0.05, 245 iterations), beating every LightGBM variant.

## Preconditions

A causal-history feature frame already exists in the candidate lineage
(build on the model_lgbm_causal_history parent rather than re-deriving the
frame), and executable FM parity is verified.

## Allowed data

Same rules as model_lgbm_causal_history: outcome columns only through
strictly-past history or leave-one-out training aggregates.

## Expected effect

A diverse, individually strong ranking member: sibling measurement 0.6120
test standalone, and worth about +0.007 when z-blended with the LightGBM
members — diversity beat depth for the last mile in that study.

## Falsification condition

No trusted improvement over the LightGBM causal-history parent standalone
AND no gain when blended with it.

## Do not use when

The wall-clock budget cannot fit its training (about 7 minutes on CPU in
the sibling study); do not shrink training to make it fit.

## Minimal implementation

Same feature frame and group-per-user construction as the parent; CatBoost
parameters measured strong in the sibling study: loss_function YetiRank,
depth 6, learning_rate 0.05, around 245 iterations (frozen — there is no
label access for early stopping in this harness), random_seed bound to
invocation.seed, categorical ids passed as CatBoost cat_features strings,
drop raw video_id and music_id. Predict raw scores, keep them
unconstrained, ensure finiteness. Train at full strength for every
fidelity.

## Sources

Prokhorenkova et al., "CatBoost: unbiased boosting with categorical
features", NeurIPS 2018. Recipe measured in the sibling lab study
(lab/PLAYBOOK.md).
