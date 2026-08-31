```json
{"schema_version":"1.0","method_id":"temporal_recency_weighted_ranker","family":"temporal_history","status":"candidate","tags":["recency","temporal","within_user","residual"],"cost_tier":"medium","prerequisites":["baseline_parity","strict_temporal_cutoff"],"allowed_data":["train_interactions","date","user_id","video_id","author_id","long_view","verified_predictions"],"prohibition_conditions":["future_aggregate_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2302.02352"]}
```

## Mechanism

Fit a bounded residual ranker with one fixed exponential recency weighting over
strictly historical interactions.

## Preconditions

Training dates are ordered and strictly precede scoring dates.

## Allowed data

Only past training interactions and verified parent predictions.

## Expected effect

Reduce stale-preference bias without requiring a long sequence model.

## Falsification condition

Recency weighting does not improve ranking or concentrates gains in one date.

## Do not use when

Future interactions or validation-derived decay selection would be required.

## Minimal implementation

Use one documented half-life, deterministic weights, and bounded parent-scale
residuals with exact fallback.

## Sources

TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling.
