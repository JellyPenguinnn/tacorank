```json
{"schema_version":"1.0","method_id":"objective_listwise_user_softmax","family":"objective","status":"candidate","tags":["listwise","within_user","top_k"],"cost_tier":"medium","prerequisites":["pairwise_tested","user_impression_groups"],"allowed_data":["train_interactions","user_id","long_view"],"prohibition_conditions":["uninformative_lists_unhandled"],"sources":["https://mlanthology.org/icml/2007/cao2007icml-learning/"]}
```

## Mechanism

Optimize a probability distribution over each user's observed impression list
so training reflects list ordering and top-rank placement.

## Preconditions

The pairwise objective has been tested and user impression groups are available.
A pure listwise residual is an independent objective formulation; an nDCG@5
weakness is required only for a pairwise-plus-listwise hybrid.

## Allowed data

Only contract-permitted observed impressions and `long_view` labels from the
training population.

## Expected effect

Improve top-5 ordering without discarding broad within-user separation.

## Falsification condition

A trusted full result does not improve nDCG@5 beyond noise or materially
regresses GAUC.

## Do not use when

User lists cannot be constructed deterministically, or the proposed loss gives
all-positive/all-negative lists an invented ordering signal.

## Minimal implementation

Keep the exact parent score and current representation fixed. Add only a
scale-calibrated user-list residual, or a small pairwise-plus-listwise residual,
with explicit handling of uninformative lists. Use the same parent-relative
limits as the pairwise card: residual standard deviation at most `0.01` of the
parent score standard deviation, maximum absolute residual at most `0.02` of
the parent score standard deviation, and proxy Spearman versus parent of at
least `0.995`. Do not reuse a broadly regressing BPR scorer as the hybrid base.

## Sources

Cao et al., Learning to Rank: From Pairwise Approach to Listwise Approach.
