```json
{"schema_version":"1.0","method_id":"objective_pairwise_hinge_margin","family":"objective","status":"candidate","tags":["pairwise","margin","within_user","parent_replacement"],"cost_tier":"medium","prerequisites":["baseline_parity","within_user_positive_negative_pairs"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","duration_ms","long_view"],"prohibition_conditions":["evaluator_or_split_change_required"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2106.06713"]}
```

## Mechanism

Replace FM with a compact within-user ranker trained by a bounded pairwise
hinge loss with one fixed margin.

## Preconditions

Deterministic positive-negative pairs exist within users.

## Allowed data

Only contract-permitted training interactions and `long_view` labels.

## Expected effect

Enforce useful score separation when logistic BPR gradients are too diffuse.

## Falsification condition

The margin collapses scores or fails to improve within-user ranking.

## Do not use when

The evaluator, split, or score-population labels would need to change.

## Minimal implementation

Use one predeclared margin, capped deterministic pairs, non-zero initialization,
and a direct finite score path without an FM blend.

## Sources

AutoLoss: Automated Loss Function Search in Recommendations.
