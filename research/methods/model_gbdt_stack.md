```json
{"schema_version":"1.0","method_id":"model_gbdt_stack","family":"model","status":"candidate","tags":["lightgbm","gbdt","stacking","replacement_capable","model"],"cost_tier":"high","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":["https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree","https://arxiv.org/abs/1706.09516"]}
```

## Mechanism

Replace the served score with a gradient-boosted decision tree model whose
input features include the setup-verified FM parent score itself.

This card is **replacement-capable**: the candidate's output score is the GBDT
prediction directly, not the FM score plus a bounded residual. Because the FM
score enters the trees as an ordinary input feature, the GBDT hypothesis class
strictly contains the parent ordering — a stump on that single feature already
reproduces it — so replacement here is a superset, not a gamble. Trees then add
what a second-order factorisation machine cannot express: non-monotone
threshold effects, arbitrary-order categorical interactions (user × tab,
author × duration bucket, date-derived recency), and automatic feature
selection under a within-user ranking-aligned objective.

## Preconditions

Executable FM parity is verified and the aligned parent prediction file is
supplied, so the FM score can be joined as a training feature and as the
fallback for rows with unseen categories.

## Allowed data

Contract-permitted fields only. Every engineered aggregate (per-user, per-item,
per-author positive rates or counts) must be computed strictly from training
rows dated before the scored rows, with deterministic smoothing; never from the
scored population.

## Expected effect

Improve within-user ordering by a larger margin than any bounded residual can:
the model may reorder items freely where the trees find signal, while the FM
feature anchors it to the parent ordering where they do not.

## Falsification condition

No trusted full-fidelity improvement over the FM parent at matched budget, or a
Spearman correlation with the parent so high (for example above 0.995) that the
trees learned nothing beyond the FM feature.

## Do not use when

The evaluator, split, or scored population would need to change, or the wall
clock budget cannot train the boosted model honestly at every fidelity. The
fidelity views share identical training data, so train at full strength for
smoke, proxy, and full alike.

## Minimal implementation

Start from the reference implementation in the model_lgbm_lambdarank_blend
card when it is supplied: keep its data loading, feature frame, category
alignment, grouping, and output plumbing, and change only the objective and
the combine step (output the model's own score instead of the blended
residual, with the FM score added as an input feature).

Train one LightGBM model (binary objective on long_view, or lambdarank grouped
by user) over the permitted raw fields, a small set of strictly past-only
aggregate features, and the aligned FM parent score as one input feature. Use
pandas categorical dtypes for user_id, video_id, author_id, and tab so LightGBM
handles them natively; set deterministic single-threaded parameters and a fixed
seed. Output the model's raw margin score for every scored row on its own
unconstrained scale — do not sigmoid, clip, normalize, or blend back into the
parent by hand. For rows whose user or item was never seen in training, the FM
feature alone drives the prediction, which is the intended fallback. Keep the
feature count modest (tens, not hundreds) and rely on early rounds and depth
limits, not feature volume, to stay inside the wall-clock bound.

## Sources

Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
NeurIPS 2017. Prokhorenkova et al., "CatBoost: unbiased boosting with
categorical features", NeurIPS 2018 (for ordered target-statistic caution on
categorical aggregates).
