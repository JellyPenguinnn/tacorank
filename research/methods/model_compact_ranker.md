```json
{"schema_version":"1.0","method_id":"model_compact_ranker","family":"model","status":"candidate","tags":["deepfm","dcn","model"],"cost_tier":"high","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["baseline_or_objective_unresolved"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/1703.04247"]}
```

## Mechanism

Capture interactions not represented by the baseline FM.

## Preconditions

Baseline parity and the objective/data frame are verified.

## Allowed data

Only contract-permitted features and labels.

## Expected effect

Improve ranking through additional interactions.

## Falsification condition

No improvement after a bounded, mechanism-driven trial.

## Do not use when

Baseline parity or the objective contract is unresolved.

## Minimal implementation

Train one compact additive residual over the supplied FM parent with fixed
resource bounds. Use all rows or a deterministic representative sample with an
explicit coverage fraction; report user/item/date unknown rates, avoid a
chronologically biased first-N slice, and retain the FM score for unseen
categories. Keep the FM value on its unconstrained ranking-score scale: bound
only the learned residual and never clip, sigmoid, normalize, or rescale the
combined score.

## Sources

[DeepFM](https://arxiv.org/abs/1703.04247) is the primary directional source
for the compact interaction-learning alternative.
