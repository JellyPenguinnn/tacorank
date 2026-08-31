```json
{"schema_version":"1.0","method_id":"multitask_shared_bottom","family":"multitask","status":"candidate","tags":["shared_bottom","multi_task","representation"],"cost_tier":"high","prerequisites":["legal_auxiliary_label"],"allowed_data":["train_interactions","long_view","auxiliary_engagement_labels"],"prohibition_conditions":["auxiliary_label_not_permitted"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://doi.org/10.1016/j.ins.2023.02.021"]}
```

## Mechanism

Share a compact lower representation between the primary ranking head and one
contract-permitted auxiliary engagement head.

## Preconditions

An explicitly legal auxiliary label is available.

## Allowed data

Only the primary target and permitted auxiliary training labels.

## Expected effect

Regularize the primary representation using related engagement supervision.

## Falsification condition

The primary trusted full score regresses or the auxiliary signal is not legal.

## Do not use when

The contract does not expose an auxiliary label.

## Minimal implementation

Use one shared layer, separate heads, and a fixed auxiliary loss weight; never
let the auxiliary head produce the official score.

## Sources

[Knowledge distillation-enhanced shared-bottom recommendation](https://doi.org/10.1016/j.ins.2023.02.021)
is a directional source; this candidate still requires an explicitly legal
auxiliary label.
