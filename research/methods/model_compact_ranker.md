```json
{"schema_version":"1.0","method_id":"model_compact_ranker","family":"model","status":"candidate","tags":["deepfm","dcn","model"],"cost_tier":"high","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view"],"prohibition_conditions":["baseline_or_objective_unresolved"],"sources":[]}
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

Try one compact alternative ranker with fixed training and resource bounds.

## Sources

No external source required for the bounded trial.
