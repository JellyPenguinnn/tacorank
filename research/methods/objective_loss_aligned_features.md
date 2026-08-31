```json
{"schema_version":"1.0","method_id":"objective_loss_aligned_features","family":"objective","status":"candidate","tags":["features","pairwise","loss_alignment","within_user"],"cost_tier":"medium","prerequisites":["pairwise_tested"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["simultaneous_loss_change","future_or_validation_aggregate_required"],"implementation_targets":["solution/candidate.py"],"sources":[]}
```

## Mechanism

Keep the tested ranking loss fixed and add one bounded feature representation
whose values vary inside that loss's within-user comparison group.

## Preconditions

The pairwise objective has produced one clean evaluated result, so its metric
shape and score behavior are known before feature engineering begins.

## Allowed data

Only contract-permitted training interactions, identifiers, context fields,
video duration, labels from training rows, and the verified FM parent score.

## Expected effect

Give the fixed pairwise or listwise loss discriminative, leakage-safe signals
that can change relative item ordering for the same user.

## Falsification condition

The new features do not create meaningful within-user score variation or do not
improve a trusted evaluation beyond noise at matched training budget.

## Do not use when

The proposal also changes the loss, requires future/validation aggregates, or
uses only user-level constants that cancel in pairwise score differences.

## Minimal implementation

Keep the accepted loss, optimizer, split, seed, FM parent, and training budget
fixed. Add one compact group of training-only features, prioritizing past-only
user-item or user-author affinity, item/author frequency residuals, or bounded
user × item/context interactions. Apply identical deterministic transforms at
scoring time. Any training-label aggregate must be strictly past-only or
out-of-fold, and no aggregate may use validation or score rows.

## Sources

The live OpenAlex skill must attach relevant paper evidence to each proposal.
