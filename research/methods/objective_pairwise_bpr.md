```json
{"schema_version":"1.0","method_id":"objective_pairwise_bpr","family":"objective","status":"candidate","tags":["pairwise","within_user","ranking"],"cost_tier":"medium","prerequisites":["baseline_parity","within_user_positive_negative_pairs"],"allowed_data":["train_interactions","user_id","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"implementation_targets":["solution/candidate.py"],"sources":[]}
```

## Mechanism

Optimize positive-versus-negative ordering within users as a calibrated parent
correction.

## Preconditions

Executable FM parity is verified and within-user positive/negative pairs are
available.

## Allowed data

Only contract-permitted training rows and features.

## Expected effect

Improve ranking without replacing the parent order.

## Falsification condition

No stable gain after scale checks; regression beyond `eta` or parent Spearman
below `0.995` retires this recipe across parents.

## Do not use when

The evaluator/splits must change, or this scale-correct recipe already severely
regressed in the run.

## Minimal implementation

Keep the official FM score exact. Train only an additive residual on observed
same-user pairs with deterministic capped sampling and non-zero initialization.
On training rows, center within user and freeze a multiplier satisfying
`residual_std <= 0.01 * parent_score_std` and `max_abs_residual <= 0.02 *
parent_score_std`. Output `exact_parent + calibrated_residual`; never add raw
BPR scores or use a fixed absolute clip. Before full promotion verify both
bounds, proxy Spearman at least `0.995`, finite variance, and repeated-item
personalization. Permit one wiring/scale repair; otherwise retire. Wire through
`solution/candidate.py`.

## Sources

No external source required for the bounded baseline trial.
