```json
{"schema_version":"1.0","method_id":"multitask_watch_time_auxiliary","family":"multitask","status":"candidate","tags":["multitask","watch_time","auxiliary"],"cost_tier":"medium","prerequisites":["baseline_parity","legal_auxiliary_label"],"allowed_data":["train_interactions","long_view","play_time_ms","duration_ms","auxiliary_engagement_labels","verified_predictions"],"prohibition_conditions":["auxiliary_label_not_permitted"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2306.03392"]}
```

## Mechanism

Regularize the long-view residual with one clipped watch-time auxiliary target.

## Preconditions

Observed `play_time_ms` is available only in the training view.

## Allowed data

Training long-view, play-time, duration, and verified parent predictions.

## Expected effect

Provide graded engagement supervision beyond the binary target.

## Falsification condition

The auxiliary head hurts primary ranking or merely predicts video duration.

## Do not use when

Watch time is unavailable, uncapped, or read from a scoring population.

## Minimal implementation

Clip and normalize watch time from training only, use one fixed auxiliary loss
weight, and retain a bounded primary residual with exact fallback.

## Sources

Tree based Progressive Regression Model for Watch-Time Prediction.
