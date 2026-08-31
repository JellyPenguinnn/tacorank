```json
{"schema_version":"1.0","method_id":"objective_pairwise_bpr","family":"objective","status":"candidate","tags":["pairwise","within_user","ranking"],"cost_tier":"medium","prerequisites":["baseline_parity","within_user_positive_negative_pairs"],"allowed_data":["train_interactions","user_id","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/1205.2618"]}
```

## Mechanism

Optimize relative positive-versus-negative ordering within users.

## Preconditions

Executable FM parity is verified and within-user positive/negative pairs are
available.

## Allowed data

Only contract-permitted training rows and features.

## Expected effect

Improve GAUC and nDCG-aligned ordering.

## Falsification condition

No stable primary-score improvement over the pointwise parent.

## Do not use when

The evaluator or split definitions would need to change.

## Minimal implementation

Keep the supplied official-FM score as the parent and train a bounded additive
pairwise residual from observed positive/negative impressions of the same user.
Use deterministic small non-zero factor initialization (zero-initializing both
factor sides produces zero latent gradients), multiple representative passes,
and capped per-user pairs. Verify that the residual has non-zero variance and
that repeated items can receive different user-conditioned scores. Implement
and wire the mechanism through the approved candidate scaffold. Keep
`solution/candidate.py` as the stable entrypoint that imports and uses the
changed helpers; `solution/train.py` is never an alternate entrypoint.

## Sources

[Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
is the primary directional source for the pairwise objective.
