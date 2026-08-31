```json
{"schema_version":"1.0","method_id":"model_field_aware_fm","family":"model","status":"candidate","tags":["ffm","field_aware","feature_interaction"],"cost_tier":"high","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view"],"prohibition_conditions":["baseline_or_objective_unresolved"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/1701.04099"]}
```

## Mechanism

Use field-aware factor interactions so the effect of one field can depend on
the field it interacts with.

## Preconditions

Baseline parity and the data frame are verified.

## Allowed data

Only contract-permitted fields and training targets.

## Expected effect

Capture user-item, user-context, and item-context interactions that a shared FM
embedding may underfit.

## Falsification condition

The bounded FFM path fails trusted full validation or is unstable across seeds.

## Do not use when

The implementation replaces the parent score or exceeds the approved budget.

## Minimal implementation

Use a small field-aware residual, deterministic initialization, and preserve the
exact parent fallback for unsupported or unseen fields.

## Sources

[Field-aware Factorization Machines in a Real-world Online Advertising
System](https://arxiv.org/abs/1701.04099) is the primary directional source.
