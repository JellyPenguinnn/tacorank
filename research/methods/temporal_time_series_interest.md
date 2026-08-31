```json
{"schema_version":"1.0","method_id":"temporal_time_series_interest","family":"temporal_history","status":"candidate","tags":["time_series","recency","drift"],"cost_tier":"medium","prerequisites":["baseline_parity","strict_temporal_cutoff"],"allowed_data":["train_interactions","date","user_id","video_id","author_id","long_view"],"prohibition_conditions":["unreliable_event_ordering","future_aggregate_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Model a user's historical engagement as a bounded time series with recency
decay and a small trend summary.

## Preconditions

Strict temporal cutoffs and deterministic date ordering are available.

## Allowed data

Only earlier training interactions and permitted scoring fields.

## Expected effect

Track gradual interest drift while reducing stale-history bias.

## Falsification condition

The time-series residual fails trusted full validation or uses future data.

## Do not use when

Event ordering is ambiguous or the time window cannot be frozen.

## Minimal implementation

Use fixed windows, simple decay/trend statistics, and exact parent fallback for
short or empty histories.

## Sources

No external source required for the bounded trial.
