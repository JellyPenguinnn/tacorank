```json
{"schema_version":"1.0","method_id":"multitask_mmoe","family":"multitask","status":"candidate","tags":["mmoe","mixture_of_experts","multi_task"],"cost_tier":"high","prerequisites":["legal_auxiliary_label"],"allowed_data":["train_interactions","long_view","auxiliary_engagement_labels"],"prohibition_conditions":["auxiliary_label_not_permitted"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Use task-specific gates over a small shared mixture of experts.

## Preconditions

An explicitly legal auxiliary label is available.

## Allowed data

Only permitted primary and auxiliary training targets.

## Expected effect

Reduce negative transfer between related engagement objectives.

## Falsification condition

The primary trusted full score regresses or expert routing is unstable.

## Do not use when

Auxiliary supervision is unavailable or the expert budget is not bounded.

## Minimal implementation

Use a small fixed expert count, deterministic gates, and a primary-only official
score with fixed auxiliary weighting.

## Sources

No external source required for the bounded trial.
