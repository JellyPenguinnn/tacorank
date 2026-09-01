```json
{"schema_version":"1.0","method_id":"objective_lambdarank_ndcg","family":"objective","status":"candidate","tags":["listwise","lambdarank","ndcg","within_user"],"cost_tier":"medium","prerequisites":["baseline_parity","user_impression_groups"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":["https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/"]}
```

## Mechanism

Weight each within-user pair by the nDCG change that swapping it would cause,
so the objective optimises the ranking metric directly instead of a surrogate
that treats every pair alike.

LambdaRank (Burges, Microsoft Research) starts from the observation that
ranking metrics are flat or discontinuous in the model scores and cannot be
optimised by gradient descent directly, but the *gradient* can be defined:
scale each pairwise term by |ΔnDCG| for that swap. Pairs whose order controls
the top of the list get large updates and pairs deep in the list get small
ones.

That is the gap in this portfolio. `objective_pairwise_bpr` weights every
positive-negative pair equally, and `objective_listwise_user_softmax` places
mass on the observed ordering, but neither is shaped by the metric being
scored. nDCG@5 is half the primary score and is dominated by the first five
positions, which is exactly what the lambda weighting emphasises.

## Preconditions

Executable FM parity is verified and users have multi-row impression lists with
both labels present. Users whose list is all-positive or all-negative have a
constant nDCG and contribute no lambda.

## Allowed data

Contract-permitted training rows only. Training dates must strictly precede
every scored date.

## Expected effect

Improve nDCG@5 more than an unweighted pairwise objective does, and improve
GAUC at least as much, because the weighting redistributes existing pairwise
capacity toward the positions the metric rewards rather than adding new
information.

## Falsification condition

No trusted full-fidelity improvement over the parent at matched budget, or
nDCG@5 fails to improve while GAUC does, which would mean the lambda weighting
is not reaching the top of the list.

## Do not use when

The truncation used to compute the lambdas disagrees with the scored metric.
The contract measures nDCG@5, so weighting by a gain defined at a different
cutoff optimises something the evaluator does not reward.

## Minimal implementation

Take the existing within-user pairwise residual and scale each pair's update by
the |ΔnDCG@5| of swapping those two items in the user's current ordering. Keep
the frozen FM score as the parent, add the residual on the original score
scale, and hold the model family, features, evaluator, split, and population
fixed. Verify that the residual reorders within-user pairs rather than shifting
whole lists.

## Sources

Burges, "From RankNet to LambdaRank to LambdaMART: An Overview", Microsoft
Research MSR-TR-2010-82. The lambda weighting is the part in scope; the boosted
tree ensemble of LambdaMART is a separate model-family change and is not.
