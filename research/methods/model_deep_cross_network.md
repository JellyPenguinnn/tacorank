```json
{"schema_version":"1.0","method_id":"model_deep_cross_network","family":"model","status":"candidate","tags":["dcn","cross_network","deep_ranker"],"cost_tier":"high","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view"],"prohibition_conditions":["baseline_or_objective_unresolved"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/1708.05123"]}
```

## Mechanism

Add a bounded low-order cross network to explicitly model feature conjunctions.

## Preconditions

Baseline parity and the data frame are verified.

## Allowed data

Only contract-permitted training fields and labels.

## Expected effect

Improve ranking when useful feature crosses are sparse but repeatable.

## Falsification condition

The cross network does not improve trusted full validation or causes instability.

## Do not use when

It relies on validation labels, arbitrary feature search, or unbounded depth.

## Minimal implementation

Use a small fixed number of cross layers as an additive residual with finite,
deterministic training and exact parent fallback.

## Sources

[Deep & Cross Network](https://arxiv.org/abs/1708.05123) is the primary
directional source for bounded explicit feature crosses.
