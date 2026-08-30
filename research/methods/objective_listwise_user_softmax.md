```json
{"schema_version":"1.0","method_id":"objective_listwise_user_softmax","family":"objective","status":"candidate","tags":["listwise","within_user","top_k"],"cost_tier":"medium","prerequisites":["pairwise_tested","user_impression_groups"],"allowed_data":["train_interactions","user_id","long_view"],"prohibition_conditions":["uninformative_lists_unhandled"],"capability_status":"verified","implementation_id":"objective_listwise_full_v2","implementation_targets":["solution/research_scaffold.py"],"configuration_target":"solution/experiment_config.py","active_parameters":["formulation","embedding_dim","learning_rate","epochs","l2","residual_scale","max_train_rows","listwise_strategy"],"sources":["https://mlanthology.org/icml/2007/cao2007icml-learning/"]}
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

Keep the current representation fixed and apply softmax to each complete informative
user list. Normalize target mass uniformly across every positive item in that list;
skip all-positive and all-negative lists rather than inventing an ordering signal.

## Sources

Cao et al., Learning to Rank: From Pairwise Approach to Listwise Approach.
