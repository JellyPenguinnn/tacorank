```json
{"schema_version":"1.0","method_id":"multitask_ple","family":"multitask","status":"candidate","tags":["ple","progressive_layered_extraction","multi_task"],"cost_tier":"high","prerequisites":["legal_auxiliary_label"],"allowed_data":["train_interactions","long_view","auxiliary_engagement_labels"],"prohibition_conditions":["auxiliary_label_not_permitted"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://doi.org/10.1145/3383313.3412236"]}
```

## Mechanism

Progressively separate task-specific and shared expert representations across a
small number of layers.

## Preconditions

An explicitly legal auxiliary label is available.

## Allowed data

Only permitted primary and auxiliary training targets.

## Expected effect

Preserve common signal while isolating task-specific ranking evidence.

## Falsification condition

The primary trusted full score regresses or progressive routing is unstable.

## Do not use when

Auxiliary supervision is not permitted or depth exceeds the cost budget.

## Minimal implementation

Use one compact extraction layer, fixed gates, deterministic seeds, and score
only the primary head for the official output.

## Sources

[Progressive Layered Extraction](https://doi.org/10.1145/3383313.3412236) is the
primary directional source for separating shared and task-specific experts.
