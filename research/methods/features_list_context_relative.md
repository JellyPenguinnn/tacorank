```json
{"schema_version":"1.0","method_id":"features_list_context_relative","family":"features","status":"candidate","tags":["list_context","re_ranking","within_user","residual"],"cost_tier":"medium","prerequisites":["baseline_parity","user_impression_groups"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","duration_ms","date","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":["https://arxiv.org/abs/1904.06813","https://dl.acm.org/doi/10.1145/3298689.3347000"]}
```

## Mechanism

Score each impression using features computed *relative to the other items in
the same user's list*, rather than from the pair alone.

The frozen FM parent scores every (user, item) pair independently, so it cannot
represent anything about the composition of the candidate set. Personalized
Re-ranking (Pei et al., RecSys 2019) is built on exactly this observation: a
ranker that encodes the whole list can capture mutual influence between items
that a point-scored model structurally cannot, and it is designed to run as a
modular stage on top of an existing ranker's outputs rather than replacing it.

That is the same shape as this contract: keep the parent, add one bounded
residual over list-relative features.

## Preconditions

Executable FM parity is verified and users have multi-row impression lists. The
planner data profile reports the list-size distribution and
`score_single_row_user_fraction`; single-row users contribute no within-user
pair and are unaffected by construction.

## Allowed data

Contract-permitted fields only. List-relative features are computed from the
scored population's own rows, which contain no labels; any statistic that
carries a label must come from training rows strictly preceding every scored
date.

## Expected effect

Improve GAUC and nDCG@5 by supplying the one class of signal the parent cannot
express. Both metrics are computed strictly inside a user's list, so a feature
defined by an item's position within that list is directly aligned with what is
scored.

## Falsification condition

No trusted full-fidelity improvement over the parent at matched budget, or the
residual reorders fewer within-user pairs than the no-op threshold, which would
show the list-relative terms collapsed to a per-user constant.

## Do not use when

The chosen statistic does not vary inside a list. A user-level mean, count, or
rate is identical for every item that user is scored on, so it cannot change
within-user order however predictive it looks marginally. This is the most
common way this direction fails.

## Minimal implementation

For each user's scoring list, compute a small set of within-list relative terms
over permitted fields, such as an item's duration percentile within the list
and its deviation from the list mean. Learn one bounded additive residual over
those terms and add it to the frozen FM parent on the original score scale. Do
not add a new model family, change the evaluator or split, or introduce a
sequence model in the same experiment.

## Sources

Pei et al., "Personalized Re-ranking for Recommendation", RecSys 2019. The
list-context motivation transfers; the transformer architecture does not, and
is explicitly out of scope for a bounded residual.
