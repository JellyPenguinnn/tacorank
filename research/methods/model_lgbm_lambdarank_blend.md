```json
{"schema_version":"1.0","method_id":"model_lgbm_lambdarank_blend","family":"model","status":"candidate","tags":["lightgbm","lambdarank","residual","proven_recipe","model"],"cost_tier":"medium","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":["https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree"]}
```

## Mechanism

Blend a LightGBM LambdaRank ranker into the FM parent on the parent's own
per-user scale:

    final = FM + ALPHA * std_user(FM) * z_user(raw_ranker)

with ALPHA = 0.70. The ranker's raw scores are z-scored within each user and
rescaled by that user's FM score spread, so the correction is strong enough to
reorder items within a user but can never blow up the parent's scale. This
exact construction produced a trusted, seed-confirmed +0.0023 full-fidelity
primary gain over the verified parent in a sibling evaluation of the same
contract (run_017_global_repro50 exp_001), and a rowwise mean of the seeds
0, 1, 2 members added roughly +0.0002 more (exp_002).

## Preconditions

Executable FM parity is verified and the aligned parent prediction file is
supplied so the FM score is joinable per scored row.

## Allowed data

Contract-permitted fields only. The proven feature frame, restricted to this
contract's columns: categorical user_id, video_id, author_id, tab, and a
deterministic duration bucket; numeric day (from date), and log1p(duration_ms).
The sibling run also used hour-of-day features from a column this contract does
not expose; omit them, do not substitute label-derived aggregates.

## Expected effect

Improve within-user ordering by roughly +0.002 primary over the FM parent,
with a small further gain from a fixed rowwise seed-mean of independently
refit members.

## Falsification condition

No trusted full-fidelity improvement over the FM parent with the frozen
parameters, or the gain disappears when the residual scale is held at the
predeclared ALPHA (which would indicate the earlier result was tuned noise).

## Do not use when

The evaluator, split, or scored population would need to change. Do not tune
ALPHA or the LightGBM parameters on evaluation feedback inside one experiment;
a deliberate ablation is a separate proposal.

## Minimal implementation

Train LightGBM (the pinned 4.x) with the frozen audited parameter set:
objective lambdarank, lambdarank_norm false, lambdarank_truncation_level 8,
label_gain [0, 1], sigmoid 1.0, learning_rate 0.04, num_leaves 63, max_depth
10, min_data_in_leaf 300, min_data_per_group 100, lambda_l1 0.2, lambda_l2
8.0, cat_l2 10.0, cat_smooth 40.0, max_bin 127, max_cat_threshold 64,
max_cat_to_onehot 4, no bagging or feature subsampling, deterministic true,
force_col_wise true, every seed field bound to invocation.seed, and 260
boosting rounds. Build ranking groups from the actual user groups with a
stable sort and restore row order by the inverse permutation. Predict with
raw_score=True, z-score the raw scores within each user (population ddof=0),
rescale by that user's FM standard deviation (ddof=0), multiply by ALPHA=0.70,
and add to the untouched FM parent score. Users with a single row or zero FM
spread keep the plain FM score. Never sigmoid, clip, normalize, or rescale the
combined result. Train at full strength for every fidelity.

## Sources

Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
NeurIPS 2017. Recipe reproduced from the sibling evaluation
run_017_global_repro50 (exp_001 accepted at full fidelity, exp_002 seed-mean).
