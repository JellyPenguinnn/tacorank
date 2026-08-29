```json
{"schema_version":"1.0","method_id":"objective_pairwise_bpr","family":"objective","status":"candidate","tags":["pairwise","within_user","ranking"],"cost_tier":"medium","prerequisites":["baseline_parity","within_user_positive_negative_pairs"],"allowed_data":["train_interactions","user_id","long_view"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":[]}
```

## Mechanism

Optimize relative positive-versus-negative ordering within users.

## Preconditions

Within-user positive/negative pairs are available.

## Allowed data

Only contract-permitted training rows and features.

## Expected effect

Improve GAUC and nDCG-aligned ordering.

## Falsification condition

No stable primary-score improvement over the pointwise parent.

## Do not use when

The evaluator or split definitions would need to change.

## Minimal implementation

Use deterministic, capped within-user positive/negative pairing.

## Sources

No external source required for the bounded baseline trial.
