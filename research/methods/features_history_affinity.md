```json
{"schema_version":"1.0","method_id":"features_history_affinity","family":"features","status":"candidate","tags":["history","affinity","point-in-time","regularized"],"cost_tier":"medium","prerequisites":["baseline_parity","strict_temporal_cutoff","history_affinity_features_legal"],"allowed_data":["train_interactions","date","time_ms","hourmin","user_id","video_id","author_id","tab","duration_ms","long_view","item_tags","upload_date","point_in_time_history_features"],"prohibition_conditions":["future_aggregate_required"],"implementation_targets":["solution/research_scaffold.py"],"configuration_target":"solution/experiment_config.py","capability_status":"verified","implementation_id":"features_history_affinity_v1","active_parameters":["formulation","learning_rate","epochs","l2","residual_scale","max_train_rows","history_shrinkage"],"sources":["https://kuairand.com/","https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction"]}
```

## Mechanism

Rank with a bounded regularized residual built from candidate-conditioned tag,
author, duration, and context affinities computed from strictly earlier events.

## Preconditions

The deployment has emitted hash-bound point-in-time feature views and preserved
exact frozen-FM passthrough on every execution route.

## Allowed data

Training interactions, basic item metadata, request context, and aggregates
whose state is emitted before each training update and frozen before scoring.

## Expected effect

Improve within-user ordering when candidate attributes match a user's observed
positive history without memorizing raw user or item identifiers.

## Falsification condition

No stable multi-seed gain over the frozen FM, a gain confined to one date or
history cohort, or sensitivity to weak shrinkage that disappears under full
evaluation.

## Do not use when

The proposal merely adds static CWM fields, changes embedding capacity, uses
current-row engagement outcomes, or requires validation/test aggregates.

## Minimal implementation

Keep the reviewed feature materializer fixed and vary only the declared typed
regularization, shrinkage, training-budget, and residual-scale parameters.

## Sources

KuaiRand's sequential-data contract and candidate-conditioned interest
modeling from DIN.
