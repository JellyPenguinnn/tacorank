```json
{"schema_version":"1.0","method_id":"objective_weighted_cross_entropy","family":"objective","status":"candidate","tags":["weighted_cross_entropy","implicit_feedback","ranking"],"cost_tier":"medium","prerequisites":["baseline_parity"],"allowed_data":["train_interactions","user_id","long_view"],"prohibition_conditions":["evaluator_or_split_change_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Use a fixed, bounded class weighting in a pointwise cross-entropy residual to
address asymmetric implicit-feedback supervision.

## Preconditions

Baseline parity and the primary target definition are verified.

## Allowed data

Only training interactions, user identity, and long_view labels.

## Expected effect

Improve recall of useful positives without destroying within-user rank order.

## Falsification condition

The weighted objective regresses trusted full validation or collapses score
diversity.

## Do not use when

The weight is tuned from validation labels or changes evaluator semantics.

## Minimal implementation

Use one conservative fixed weight, train a bounded residual, and preserve exact
parent fallback and original unconstrained score scale.

## Sources

No external source required for the bounded trial.
