```json
{"schema_version":"1.0","method_id":"features_duration_context_interactions","family":"features","status":"candidate","tags":["duration","context","feature_interaction","residual"],"cost_tier":"low","prerequisites":["baseline_parity","duration_features_legal"],"allowed_data":["train_interactions","duration_ms","tab","author_id","long_view","verified_predictions"],"prohibition_conditions":["watch_time_semantics_confused"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":["https://doi.org/10.1145/3580305.3599797"]}
```

## Mechanism

Add fixed duration-bucket interactions with tab and author-frequency context.

## Preconditions

`duration_ms` is treated only as video length, never observed watch time.

## Allowed data

Training duration, tab, author, labels, and verified parent predictions.

## Expected effect

Model context-dependent duration bias missed by a global duration correction.

## Falsification condition

The interaction overfits sparse buckets or fails to change within-user order.

## Do not use when

The proposal conflates content duration with watch time.

## Minimal implementation

Use fixed quantile-free duration buckets, smoothed context rates, bounded
residuals, and exact fallback for unsupported cells.

## Sources

Counterfactual Video Recommendation for Duration Debiasing.
