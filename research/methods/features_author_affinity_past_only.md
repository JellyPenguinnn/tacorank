```json
{"schema_version":"1.0","method_id":"features_author_affinity_past_only","family":"features","status":"candidate","tags":["features","author","temporal","residual"],"cost_tier":"medium","prerequisites":["baseline_parity","strict_temporal_cutoff"],"allowed_data":["train_interactions","date","author_id","long_view"],"prohibition_conditions":["future_aggregate_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Add a past-only author affinity residual to capture creator-level preference
that the supplied FM score does not fully represent.

## Preconditions

The FM parent is verified and every author statistic is computed only from
interactions strictly earlier than the target row.

## Allowed data

Only contract-permitted training dates, author IDs, and `long_view` labels.

## Expected effect

Improve ranking when a user's recent author preferences contain signal beyond
the baseline features.

## Falsification condition

The residual does not improve the primary score at matched fidelity or changes
scores without a reproducible ranking benefit.

## Do not use when

The implementation needs future rows, test labels, or a user/list-wide constant
that cannot change within-user ordering.

## Minimal implementation

Compute one deterministic, strictly past-only author long-view statistic and
add one bounded residual to the supplied FM score. Keep unseen-author behavior
on the unchanged FM path.

## Sources

No external source required for the bounded trial.
