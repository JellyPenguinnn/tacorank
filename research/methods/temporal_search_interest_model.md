```json
{"schema_version":"1.0","method_id":"temporal_search_interest_model","family":"temporal_history","status":"candidate","tags":["sim","long_term_interest","temporal"],"cost_tier":"high","prerequisites":["baseline_parity","strict_temporal_cutoff"],"allowed_data":["train_interactions","date","user_id","video_id","author_id","long_view"],"prohibition_conditions":["unreliable_event_ordering","future_aggregate_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2006.05639"]}
```

## Mechanism

Separate compact long-term and recent user-interest summaries under a strict
historical cutoff.

## Preconditions

Chronological interaction order and bounded history storage are available.

## Allowed data

Only earlier permitted interactions and scoring-time fields.

## Expected effect

Preserve durable preferences while adapting to recent interest shifts.

## Falsification condition

The separated-interest residual does not improve trusted full validation or
shows temporal leakage.

## Do not use when

Long-term and recent histories cannot be constructed without future data.

## Minimal implementation

Use fixed recent/long-term caps, deterministic pooling, and exact parent fallback
for unsupported histories.

## Sources

[Search-based Interest Model](https://arxiv.org/abs/2006.05639) is the
primary directional source for general-search and exact-search interest units.
