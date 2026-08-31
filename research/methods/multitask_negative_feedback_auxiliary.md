```json
{"schema_version":"1.0","method_id":"multitask_negative_feedback_auxiliary","family":"multitask","status":"candidate","tags":["multitask","negative_feedback","auxiliary"],"cost_tier":"medium","prerequisites":["baseline_parity","legal_auxiliary_label"],"allowed_data":["train_interactions","long_view","is_hate","auxiliary_engagement_labels","verified_predictions"],"prohibition_conditions":["auxiliary_label_not_permitted"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2308.13249"]}
```

## Mechanism

Use observed negative feedback as one auxiliary penalty while optimizing
long-view ranking as the primary task.

## Preconditions

The training view explicitly permits `is_hate` and the signal is non-empty.

## Allowed data

Training primary and negative-feedback labels plus verified predictions.

## Expected effect

Separate superficially long views from interactions carrying explicit dislike.

## Falsification condition

Sparse negative feedback destabilizes training or hurts both ranking metrics.

## Do not use when

The auxiliary label is absent or would be inferred from validation behavior.

## Minimal implementation

Use one fixed class weight and one fixed auxiliary weight; keep the primary
output deterministic, finite, bounded, and parent-backed.

## Sources

Learning and Optimization of Implicit Negative Feedback for Industrial
Short-video Recommender System.
