```json
{"schema_version":"1.0","method_id":"temporal_history_target_attention","family":"temporal_history","status":"candidate","tags":["history","attention","target_aware","within_user","residual"],"cost_tier":"high","prerequisites":["baseline_parity","strict_temporal_cutoff","user_impression_groups"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","duration_ms","date","long_view","verified_predictions"],"prohibition_conditions":["future_aggregate_required"],"sources":["https://arxiv.org/abs/1706.06978","https://dl.acm.org/doi/10.1145/3219819.3219823"]}
```

## Mechanism

Weight each of a user's past interactions by how related it is to the candidate
being scored, instead of compressing that history into one candidate-agnostic
summary.

DIN (Zhou et al., Alibaba, KDD 2018) introduced this because a fixed user
vector forces every candidate to be scored against the same representation of
that user, which cannot express that a user has several distinct interests and
that only some are relevant to the item in front of them. Its local activation
unit gives each historical behaviour a weight that depends on the target item.

This is the one property the frozen parent cannot have. A factorisation machine
gives each user a single embedding, so the user side of its score is identical
for every candidate in that user's list. Only the item side varies, and GAUC
and nDCG@5 are computed strictly inside that list.

It is also distinct from `temporal_history_compact`, which builds a compact
candidate-agnostic history summary. The mechanism under test here is that the
weighting is *target-aware*, not that history is used at all.

## Preconditions

Executable FM parity is verified, users have prior interactions, and every
aggregate is computable from rows strictly preceding the scored date.

## Allowed data

Contract-permitted training rows only. History for a scored row must be drawn
exclusively from dates earlier than that row; the split makes this natural,
since training dates all precede the scored window.

## Expected effect

Improve within-user ordering by making the user representation vary across
candidates in the same list, which is the only way a user-side term can affect
these metrics at all.

## Falsification condition

No trusted full-fidelity improvement over the parent at matched budget, or the
learned attention is close to uniform, in which case the mechanism has
collapsed to the compact-history card and adds nothing over it.

## Do not use when

History is too short to weight. Users with a single prior interaction give the
attention nothing to select between, and the planner data profile reports the
list-size distribution and single-row fraction needed to judge this before
proposing.

## Minimal implementation

For each scored row, take a bounded window of that user's most recent prior
interactions, score each against the candidate by a simple relatedness term
over permitted fields such as shared author or duration bucket, normalise those
weights, and use the weighted history summary in one bounded additive residual
on the frozen parent. Keep the objective, evaluator, split, and population
fixed, and do not introduce a sequence model in the same experiment.

## Sources

Zhou et al., "Deep Interest Network for Click-Through Rate Prediction", KDD
2018. The local activation unit transfers; the full deep architecture,
mini-batch aware regularizer, and Dice activation are separate changes and are
not in scope for a bounded residual.
