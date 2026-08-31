```json
{"schema_version":"1.0","method_id":"multitask_gsu","family":"multitask","status":"candidate","tags":["gsu","gated_sharing","multi_task"],"cost_tier":"high","prerequisites":["legal_auxiliary_label"],"allowed_data":["train_interactions","long_view","auxiliary_engagement_labels"],"prohibition_conditions":["auxiliary_label_not_permitted"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Use a bounded gated shared unit to control what auxiliary-task representation is
shared with the primary ranking task.

## Preconditions

An explicitly legal auxiliary label is available.

## Allowed data

Only permitted primary and auxiliary training targets.

## Expected effect

Share useful multi-task signal while limiting negative transfer.

## Falsification condition

The primary trusted full score regresses or gating is unstable.

## Do not use when

Auxiliary supervision is not allowed by the frozen data contract.

## Minimal implementation

Use one small gate, a fixed loss weight, deterministic initialization, and keep
the official score on the primary head.

## Sources

No external source required for the bounded trial.
