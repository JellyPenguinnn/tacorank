```json
{"schema_version":"1.0","method_id":"duration_bias_censored_watch_time","family":"duration_bias","status":"candidate","tags":["duration","calibration","residual"],"cost_tier":"high","prerequisites":["baseline_parity","duration_features_legal"],"allowed_data":["train_interactions","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["duration_signal_not_permitted"],"sources":[]}
```

## Mechanism

Use video-duration-aware calibration to address duration-dependent long-view
rates without pretending that content duration is observed watch time.

## Preconditions

Duration features are explicitly legal under the frozen contract.

## Allowed data

Only contract-permitted video duration and `long_view`; `play_time_ms` is not
present in the candidate view and must not be inferred.

## Expected effect

Improve long-view ranking through duration-bias correction.

## Falsification condition

No primary improvement or mismatch with the competition definition.

## Do not use when

The contract does not permit the duration signal.

## Minimal implementation

Add one bounded, train-only log-duration interaction or calibrated residual to
the supplied FM parent. `duration_ms` is video length, not censored watch time;
reject any implementation that treats it as `play_time_ms`. The FM values are
unconstrained ranking scores, so bound only the residual and never clip,
sigmoid, normalize, or rescale the combined score.

## Sources

No external source required for the bounded trial.
