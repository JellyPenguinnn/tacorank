```json
{"schema_version":"1.0","method_id":"objective_pairwise_bpr","family":"objective","status":"candidate","tags":["pairwise","within_user","ranking"],"cost_tier":"medium","prerequisites":["baseline_parity","within_user_positive_negative_pairs"],"allowed_data":["train_interactions","user_id","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"capability_status":"verified","implementation_id":"objective_bpr_v2","implementation_targets":["solution/research_scaffold.py"],"configuration_target":"solution/experiment_config.py","active_parameters":["formulation","embedding_dim","learning_rate","epochs","negative_count","l2","residual_scale","max_train_rows"],"sources":[]}
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
that repeated items can receive different user-conditioned scores. Select the
mechanism and its bounded hyperparameters through `solution/experiment_config.py`;
the reusable executable scaffold remains unchanged across configuration trials.

## Sources

No external source required for the bounded baseline trial.
