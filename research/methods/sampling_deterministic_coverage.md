```json
{"schema_version":"1.0","method_id":"sampling_deterministic_coverage","family":"sampling","status":"candidate","tags":["sampling","coverage","within_user"],"cost_tier":"medium","prerequisites":["baseline_parity"],"allowed_data":["train_interactions","user_id","long_view"],"prohibition_conditions":["adaptive_validation_sampling"],"implementation_targets":["solution/candidate.py","solution/train.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Use a deterministic, coverage-preserving training sampler so users with long
and short impression histories contribute bounded, representative evidence.

## Preconditions

The FM parent is verified and the sample is selected before training without
looking at public or hidden evaluation labels.

## Allowed data

Only contract-permitted training user IDs and `long_view` labels.

## Expected effect

Improve generalization by preventing high-volume users from dominating the
bounded residual learner.

## Falsification condition

Coverage balancing does not improve the primary score or removes useful
within-user positive/negative evidence.

## Do not use when

The sample is selected from validation feedback, uses hidden labels, or drops
all positives or negatives for an eligible user.

## Minimal implementation

Select one fixed seed and one deterministic per-user coverage rule, retain the
FM parent for all fallback rows, and record the realized coverage in the
candidate's ordinary diagnostics.

## Sources

No external source required for the bounded trial.
