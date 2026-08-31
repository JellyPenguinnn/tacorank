```json
{"schema_version":"1.0","method_id":"multitask_esu","family":"multitask","status":"candidate","tags":["esu","expert_sharing","multi_task"],"cost_tier":"high","prerequisites":["legal_auxiliary_label"],"allowed_data":["train_interactions","long_view","auxiliary_engagement_labels"],"prohibition_conditions":["auxiliary_label_not_permitted"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Let related tasks share a bounded subset of compact expert representations.

## Preconditions

An explicitly legal auxiliary label is available.

## Allowed data

Only permitted primary and auxiliary training targets.

## Expected effect

Capture common engagement structure without forcing all experts to be shared.

## Falsification condition

The primary trusted full score regresses or expert sharing is unstable.

## Do not use when

Auxiliary supervision is not permitted.

## Minimal implementation

Use a small fixed expert pool and deterministic routing, with the primary head
remaining the only official scoring head.

## Sources

No external source required for the bounded trial.
