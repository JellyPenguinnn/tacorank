```json
{"schema_version":"1.0","method_id":"temporal_history_compact","family":"temporal_history","status":"candidate","tags":["sequence","history","temporal"],"cost_tier":"medium","sources":[]}
```

## Mechanism

Represent recent user interest without using future interactions.

## Preconditions

Each target row has a strict temporal cutoff.

## Allowed data

Interactions strictly earlier than each target row.

## Expected effect

Improve preference modeling for users with useful history.

## Falsification condition

No gain over a no-history control or evidence of temporal leakage.

## Do not use when

The source data has no reliable event ordering.

## Minimal implementation

Deterministic truncation and padding of a compact recent-history window.

## Sources

No external source required for the bounded trial.
