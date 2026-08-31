```json
{"schema_version":"1.0","method_id":"objective_direct_within_user_ranker","family":"objective","status":"candidate","tags":["pairwise","listwise","within_user","parent_replacement"],"cost_tier":"medium","prerequisites":["baseline_parity","within_user_positive_negative_pairs","user_impression_groups"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view"],"prohibition_conditions":["evaluator_or_split_change_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/1205.2618","https://mlanthology.org/icml/2007/cao2007icml-learning/"]}
```

## Mechanism

Replace the FM score path with a directly trained within-user ranker optimized
by pairwise BPR or a bounded pairwise-listwise objective.

## Preconditions

Observed training impressions can form deterministic positive-negative pairs
and user impression groups without reading validation labels.

## Allowed data

Only contract-permitted training interactions, identifiers, context, duration,
and `long_view` labels.

## Expected effect

Learn user-conditioned relative ordering directly instead of asking a small
residual to repair a fixed pointwise FM ranking.

## Falsification condition

The direct ranker does not produce meaningful within-user rank changes or does
not improve trusted ranking metrics over the FM parent.

## Do not use when

The implementation would require evaluator, split, score-population label, or
hidden-test changes.

## Minimal implementation

This is explicitly a `parent_replacement` experiment. Train a compact,
deterministic user-conditioned ranker from within-user positive-negative pairs
or informative user lists and emit its finite unconstrained score directly.
Do not add the learned score as a residual to
`fm_baseline_predictions.csv`, and do not blend FM back into the output. Use
small non-zero factor initialization, bounded representative sampling, and
explicit handling of all-positive or all-negative users. Preserve the output
schema, row order, deterministic seed, and finite fallback for unseen entities.

## Sources

Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback; Cao
et al., Learning to Rank: From Pairwise Approach to Listwise Approach.
