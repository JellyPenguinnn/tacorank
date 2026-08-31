```json
{"schema_version":"1.0","method_id":"temporal_deep_interest_network","family":"temporal_history","status":"candidate","tags":["din","attention","interest_evolution"],"cost_tier":"high","prerequisites":["baseline_parity","strict_temporal_cutoff"],"allowed_data":["train_interactions","date","user_id","video_id","author_id","long_view"],"prohibition_conditions":["unreliable_event_ordering","future_aggregate_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Attend to a user's strictly past interactions conditioned on the candidate item.

## Preconditions

Chronological cutoffs and deterministic history construction are available.

## Allowed data

Only interactions earlier than the scored row and permitted scoring fields.

## Expected effect

Match short-term interest to candidate videos without using future behavior.

## Falsification condition

The attention residual fails trusted full validation or violates temporal order.

## Do not use when

The row has no valid past history or history construction is not deterministic.

## Minimal implementation

Use a capped history, a small attention scorer, and exact parent fallback for
empty or unseen histories.

## Sources

No external source required for the bounded trial.
