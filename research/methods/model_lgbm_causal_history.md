```json
{"schema_version":"1.0","method_id":"model_lgbm_causal_history","family":"model","status":"candidate","tags":["lightgbm","lambdarank","causal_history","target_encoding","replacement_capable","model"],"cost_tier":"medium","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":["https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree"]}
```

## Mechanism

Replace the served score with a small-capacity LambdaRank GBDT whose inputs
are three signal classes the FM parent literally cannot see:

1. **Causal per-impression history** — for each TRAINING row, aggregates of
   that user's TRAINING interactions strictly before the row's `time_ms`: previous
   long_view of this exact video, rolling mean of the last 5 and last 20
   long_views, running user rate, running user-by-author rate, position
   within the current session (a >30-minute gap in `time_ms` starts a new
   session), time since the previous impression, and causal video/author
   popularity (running long_view rate over all users, ordered by time). In
   the sibling lab study this signal class alone broke a long tabular
   plateau, worth about +0.005 primary. Score rows all fall after the
   training window, so their history state is the user's state at the end
   of the train split — computed from TRAIN rows only. Never let any
   score-population row (even unlabeled) enter any aggregate, count,
   session state, or encoding: features must be identical whether or not
   the score file exists.
2. **Leave-one-out target encodings** from the training window: video and
   author long_view rates (subtract the row's own label inside the window;
   plain aggregates for score rows), a user-by-duration-bucket rate, and a
   leave-one-out mean play ratio per video from `play_time_ms/duration_ms`.
3. **Static item statistics** from video_features_statistic_pure.csv:
   `long_time_play_cnt/play_cnt` was the single best static feature in the
   sibling study; play_progress, complete/valid/like/share rate and log
   show/play counts also earn gain. Join on video_id.

## Preconditions

Executable FM parity is verified. train.csv exposes time_ms/hourmin/
is_click/play_time_ms and the side feature files are present in the input
root.

## Allowed data

Contract-permitted columns only. is_click and play_time_ms are outcomes of
their own row: they may enter ONLY through strictly-past history or
leave-one-out aggregates over training rows, never as same-row features —
a same-row outcome feature cannot exist at serve time and is rejected.

## Expected effect

Sibling lab measurements on the same data and metrics: this frame with a
small LambdaRank reached 0.6056–0.6122 valid-primary territory versus the
0.6016 FM parent. Expect a clearly positive full-fidelity delta; treat the
exact magnitude as the experiment's question.

## Falsification condition

No trusted full-fidelity improvement over the current best, or one feature
absorbing nearly all tree gain while the score collapses (the signature of
a leaked own-label feature — see Do not use when).

## Do not use when

Do not re-test these measured dead ends from the sibling study: raw
(user,video) aggregates without leave-one-out (score collapses to random);
the FM score as a plain input feature (0.5978, the FM was fit on the same
rows); graded play-ratio labels with label_gain (0.6032 vs 0.6122); wide
extra history windows such as m50/day/author rolling stats (0.6104 vs
0.6122 — m5/m20 saturate the signal); truncation level 40 (0.6036 vs
0.6122).

## Minimal implementation

Build one pandas frame over train plus score rows; compute history
features with the cumulative-minus-own-row trick after a stable sort by
(user_id, time_ms), letting only training rows contribute to sums and
counts; fill history that does not exist yet with -1. Sibling-measured
strong hyperparameters to start from (validate, do not blindly retune):
lambdarank objective, label_gain [0,1], truncation level 8, learning rate
0.05, **num_leaves 7** (capacity is the enemy here: 255 leaves scored
0.5953, 7 scored 0.6122), min_data_in_leaf 200, lambda_l2 10, feature and
bagging fraction 0.85, roughly 400 rounds, and drop raw video_id (its
signal enters through the target encodings; keep author_id as the one raw
id worth having). Categorical dtype for the kept id columns, groups built
per user with a stable sort and restored by inverse permutation, raw-score
prediction, seed bound to invocation.seed, num_threads=8 (the full frozen container
quota — never 1), and finite scores everywhere.
Train at full strength for every fidelity.

## Sources

Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
NeurIPS 2017. Feature-frame findings and dead-end measurements reproduced
from the sibling lab study of this dataset (lab/PLAYBOOK.md).
