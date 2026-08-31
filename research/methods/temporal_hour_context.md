```json
{"schema_version":"1.0","method_id":"temporal_hour_context","family":"temporal_history","status":"candidate","tags":["hour","context","temporal","residual"],"cost_tier":"low","prerequisites":["baseline_parity","strict_temporal_cutoff"],"allowed_data":["train_interactions","date","hourmin","user_id","tab","long_view","verified_predictions"],"prohibition_conditions":["future_aggregate_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2408.05709"]}
```

## Mechanism

Add a smoothed train-only hour-of-day and tab interaction residual.

## Preconditions

`hourmin` is contract-permitted and training dates precede score dates.

## Allowed data

Training timestamps, users, tabs, labels, and verified parent predictions.

## Expected effect

Capture stable consumption-time context omitted by the FM score.

## Falsification condition

The context residual is constant within users or fails to generalize by date.

## Do not use when

The feature requires future behavior or validation-tuned buckets.

## Minimal implementation

Use fixed coarse hour buckets, smoothing, deterministic fitting, bounded
residuals, and exact fallback for unseen buckets.

## Sources

Moment&Cross: Real-Time Cross-Domain CTR Prediction at Kuaishou.
