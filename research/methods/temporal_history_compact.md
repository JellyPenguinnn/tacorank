```json
{"schema_version":"1.0","method_id":"temporal_history_compact","family":"temporal_history","status":"candidate","tags":["sequence","history","temporal"],"cost_tier":"medium","prerequisites":["baseline_parity","strict_temporal_cutoff"],"allowed_data":["train_interactions","date","user_id","video_id","author_id","long_view","verified_predictions"],"prohibition_conditions":["unreliable_event_ordering"],"sources":[]}
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

Build a deterministic compact history from earlier positive (`long_view=1`)
training interactions, with negative impressions used only as explicit
negative evidence. Add a bounded similarity/affinity residual to the supplied
FM score; do not replace the parent with an uncalibrated fixed heuristic.
The FM values are unconstrained ranking scores, not probabilities: bound only
the residual and never clip, sigmoid, normalize, or rescale the combined score.

## Sources

No external source required for the bounded trial.
