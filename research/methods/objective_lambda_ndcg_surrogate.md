```json
{"schema_version":"1.0","method_id":"objective_lambda_ndcg_surrogate","family":"objective","status":"candidate","tags":["listwise","ndcg","within_user","parent_replacement"],"cost_tier":"high","prerequisites":["baseline_parity","user_impression_groups"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","duration_ms","long_view"],"prohibition_conditions":["uninformative_lists_unhandled"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2306.02239"]}
```

## Mechanism

Replace FM with a compact ranker whose pair gradients are weighted by their
bounded potential effect on top-five ordering.

## Preconditions

User impression groups can be constructed deterministically from training data.

## Allowed data

Only observed training impressions and `long_view` labels.

## Expected effect

Focus learning on swaps that matter to nDCG@5 while retaining broad ordering.

## Falsification condition

Top-five ranking does not improve or GAUC materially regresses.

## Do not use when

Uninformative all-positive or all-negative lists cannot be skipped explicitly.

## Minimal implementation

Use clipped deterministic delta-nDCG weights, bounded lists, and emit the direct
finite ranker score without blending FM.

## Sources

Generative Flow Network for Listwise Recommendation.
