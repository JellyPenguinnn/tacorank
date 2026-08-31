```json
{"schema_version":"1.0","method_id":"temporal_causal_history_features","family":"temporal_history","status":"candidate","tags":["temporal","causal","history"],"cost_tier":"medium","prerequisites":["baseline_parity","strict_temporal_cutoff"],"allowed_data":["train_interactions","date","user_id","video_id","author_id","tab","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["future_aggregate_required","ambiguous_within_date_order","unsupported_input_required","validation_tuned_weights"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Explore whether causal history signals from earlier interactions improve
ranking when added to the verified FM parent.

## Preconditions

The FM parent is verified and the candidate view provides a deterministic
temporal cutoff. If the available view cannot establish ordering within a time
bucket, the implementation must not invent that ordering.

## Allowed data

Use only the controller-provided training/scoring views and verified parent
predictions. Do not add external artifacts, hidden labels, or an unapproved
data source.

## Expected effect

Improve ranking under temporal drift while retaining exact FM parent behavior
for unsupported histories.

## Falsification condition

Reject when the candidate relies on current-row or future information, uses an
unapproved input, or fails to produce a trusted full-fidelity gain at matched
budget.

## Do not use when

The implementation requires ambiguous temporal ordering, validation/test
labels, future aggregates, new dependencies, or replacement of the
authenticated FM parent.

## Minimal implementation

The coding agent should choose a bounded, deterministic causal-history
representation from the available contract fields, compare it against the
verified parent, and demonstrate through tests that no current-row or future
information is used. Preserve the parent fallback and the existing output and
fidelity contracts.

## Sources

This card records a research direction only; no result, recipe, or external
artifact is supplied.
