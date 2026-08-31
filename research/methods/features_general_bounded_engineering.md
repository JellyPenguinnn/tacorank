```json
{"schema_version":"1.0","method_id":"features_general_bounded_engineering","family":"features","status":"candidate","tags":["feature_engineering","residual","train_only"],"cost_tier":"medium","prerequisites":["baseline_parity"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view"],"prohibition_conditions":["future_aggregate_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Add one deterministic train-only feature family that captures a documented
interaction pattern missing from the parent ranker.

## Preconditions

Baseline parity and a strictly bounded residual path are available.

## Allowed data

Only contract-permitted training interactions and fields available at scoring.

## Expected effect

Recover stable user, item, author, tab, or duration signal not represented by
the current parent.

## Falsification condition

The feature changes predictions but does not improve trusted full validation,
or requires future aggregates.

## Do not use when

The feature cannot be computed with a deterministic historical cutoff.

## Minimal implementation

Test one feature family at a time, use smoothing and an exact parent fallback,
and keep the residual bounded on the original score scale.

## Sources

No external source required for the bounded trial.
