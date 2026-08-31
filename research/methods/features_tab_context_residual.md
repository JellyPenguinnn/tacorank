```json
{"schema_version":"1.0","method_id":"features_tab_context_residual","family":"features","status":"candidate","tags":["features","tab","context","residual"],"cost_tier":"medium","prerequisites":["baseline_parity"],"allowed_data":["train_interactions","tab","long_view"],"prohibition_conditions":["evaluator_or_split_change_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Use the observed feed/tab context to model systematic long-view-rate
differences that are not represented by the baseline FM score.

## Preconditions

The FM parent is verified and the tab statistic is estimated from training
rows only.

## Allowed data

Only contract-permitted training rows, tab identifiers, and `long_view` labels.

## Expected effect

Improve calibration of relative scores across feed contexts without changing
the evaluator or split semantics.

## Falsification condition

The bounded context residual produces no primary improvement or is a constant
within every evaluated user's candidate set.

## Do not use when

The change requires hidden labels, a changed split, or a global score
normalization that replaces the FM ranking scale.

## Minimal implementation

Estimate one deterministic train-only tab residual, bound that residual, and
add it to the supplied FM score. Preserve FM scores for unseen tabs.

## Sources

No external source required for the bounded trial.
