```json
{"schema_version":"1.0","method_id":"objective_listwise_user_softmax","family":"objective","status":"candidate","tags":["listwise","within_user","top_k"],"cost_tier":"medium","sources":["https://mlanthology.org/icml/2007/cao2007icml-learning/"]}
```

## Mechanism

Optimize a probability distribution over each user's observed impression list
so training reflects list ordering and top-rank placement.

## Preconditions

The pairwise objective has been tested, user impression groups are available,
and nDCG@5 remains the diagnosed weakness.

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

Keep the current representation fixed and add a bounded user-list softmax or a
small pairwise-plus-listwise hybrid with explicit handling of uninformative lists.

## Sources

Cao et al., Learning to Rank: From Pairwise Approach to Listwise Approach.
