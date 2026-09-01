```json
{"schema_version":"1.0","method_id":"ensemble_diverse_residual_candidate","family":"ensemble","status":"candidate","tags":["ensemble","residual","rank_average","soft_prune"],"cost_tier":"low","prerequisites":["verified_best_prediction","diverse_clean_proxy_member"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["untrusted_or_severely_regressed_member","adaptive_weight_sweep"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Retain the trusted parent score path and add one explicitly identified,
clean, soft-pruned mechanism as a bounded complementary scoring path.

## Preconditions

The parent prediction is verified, the secondary experiment passed integrity
and output checks, changed predictions meaningfully, stayed within the soft
regression floor or exposed a component-metric trade-off, and is sufficiently
different from its parent.

## Allowed data

Only contract-permitted training fields and controller-verified predictions.
The component experiment IDs must be carried in the ExperimentSpec.

## Expected effect

A small fixed convex score or within-user rank blend can improve residual
ordering when the weaker member makes complementary errors.

## Falsification condition

No predeclared blend beats the trusted parent on the internal proxy, or the
gain disappears at full fidelity.

## Do not use when

Any component is suspicious, compromised, a no-op, unstable, severely below
its parent, nearly identical to its parent, or selected through an adaptive
weight sweep on public validation.

## Minimal implementation

Keep the trusted parent implementation unchanged, reconstruct exactly one
identified secondary scoring path, and test one predeclared weight from
`0.90/0.10`, `0.75/0.25`, or `0.50/0.50`. Use one weight per experiment and
the normal smoke/proxy/full ladder; do not search weights inside the candidate.

## Memory discipline

Train members SEQUENTIALLY inside the candidate: build one member's frame,
train, predict, then `del` the frame/dataset and `gc.collect()` before the
next member. Holding two full feature frames simultaneously exceeds the
container memory limit and the experiment dies on an OOM kill (this killed
a prior ensemble attempt). Peak memory must stay under a single member's
footprint plus predictions.

## Sources

No external source required for the bounded trial.
