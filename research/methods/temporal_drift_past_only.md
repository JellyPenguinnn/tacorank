```json
{"schema_version":"1.0","method_id":"temporal_drift_past_only","family":"features","status":"candidate","tags":["temporal","drift","recency"],"cost_tier":"low","prerequisites":["strict_temporal_cutoff","drift_diagnostics_material"],"allowed_data":["train_interactions","date","user_id","video_id","author_id","duration_ms","long_view"],"prohibition_conditions":["future_aggregate_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2208.08696"]}
```

## Mechanism

Represent chronological distribution shift through past-only recency statistics
and candidate-time interactions that can change relative item scores.

## Preconditions

Training-only drift diagnostics show material changes in label rate, item/author
frequency, unknown rate, duration mix, or baseline residuals.

## Allowed data

Contract-permitted timestamps and aggregates computed strictly from rows earlier
than the target interaction.

## Expected effect

Improve ranking on later chronological windows without future-data leakage.

## Falsification condition

No trusted gain at matched budget, or improvement disappears when all
aggregates are recomputed with a strict temporal cutoff.

## Do not use when

The feature is a user/list-wide constant that cannot change within-user order,
or it requires future validation/test statistics.

## Minimal implementation

Test one recency-decayed item/author statistic or one time-by-item interaction
while keeping the model, objective, and split fixed.

## Sources

KuaiRand dataset paper and local chronological split specification.
