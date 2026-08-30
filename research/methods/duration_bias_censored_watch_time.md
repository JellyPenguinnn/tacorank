```json
{"schema_version":"1.0","method_id":"duration_bias_censored_watch_time","family":"duration_bias","status":"candidate","tags":["cwm","duration","censoring"],"cost_tier":"high","prerequisites":["duration_features_legal"],"allowed_data":["train_interactions","duration_ms","long_view"],"prohibition_conditions":["duration_signal_not_permitted"],"sources":[]}
```

## Mechanism

Use one-sided duration supervision to address watch-time censoring.

## Preconditions

Duration features are explicitly legal under the frozen contract.

## Allowed data

Only contract-permitted duration observations and censoring indicators.

## Expected effect

Improve long-view ranking through duration-bias correction.

## Falsification condition

No primary improvement or mismatch with the competition definition.

## Do not use when

The contract does not permit the duration signal.

## Minimal implementation

Use a bounded duration-bias term with a simpler direct-ranking control.

## Sources

No external source required for the bounded trial.
