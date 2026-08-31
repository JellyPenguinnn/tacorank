```json
{"schema_version":"1.0","method_id":"sampling_hard_negative_pairs","family":"sampling","status":"candidate","tags":["hard_negative","pairwise","sampling"],"cost_tier":"medium","prerequisites":["baseline_parity","within_user_positive_negative_pairs"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","long_view","verified_predictions"],"prohibition_conditions":["adaptive_validation_sampling"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://ojs.aaai.org/index.php/AAAI/article/view/38690"]}
```

## Mechanism

Train a bounded residual ranker on deterministic within-user negatives that are
hard under training-only popularity and context similarity.

## Preconditions

Each selected pair is an observed training impression from the same user.

## Allowed data

Training interaction identities, context, labels, and verified predictions.

## Expected effect

Spend limited CPU updates on confusable negatives rather than easy random ones.

## Falsification condition

Hard sampling narrows coverage, destabilizes gradients, or hurts broad GAUC.

## Do not use when

Hardness depends on validation scores or unobserved counterfactual negatives.

## Minimal implementation

Use a fixed hard/easy mixture, capped pairs per user, deterministic ordering,
and a bounded parent-scale residual.

## Sources

IdeFN: Identifying Unclicked Space False Negatives.
