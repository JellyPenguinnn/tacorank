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

## Reference implementation

The following complete candidate is reviewed against this contract's exact
interface (train.csv, score.csv, aligned fm_baseline_predictions.csv, output
row_id,user_id,video_id,score). Transcribe it into solution/candidate.py and
adapt only what the approved ExperimentSpec changes; do not re-derive the
plumbing. Common from-scratch failures this avoids: pandas category alignment
(`union_categoricals` lives in `pandas.api.types`, not on the top-level
module), LightGBM `feature_name`/`num_feature` mismatches, unstable group
permutations, and ddof mistakes in the per-user z-score.

```python
"""LambdaRank residual blended onto the frozen FM parent (ALPHA = 0.70)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

ALPHA = 0.70
NUM_BOOST_ROUND = 260
CATEGORICAL = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
FEATURES = CATEGORICAL + ["day", "log_duration"]
ID_COLUMNS = {"user_id": str, "video_id": str, "author_id": str, "tab": str}


def _lgb_params(seed: int) -> dict:
    return {
        "objective": "lambdarank",
        "lambdarank_norm": False,
        "lambdarank_truncation_level": 8,
        "label_gain": [0, 1],
        "sigmoid": 1.0,
        "metric": "None",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": 10,
        "min_data_in_leaf": 300,
        "min_data_per_group": 100,
        "lambda_l1": 0.2,
        "lambda_l2": 8.0,
        "cat_l2": 10.0,
        "cat_smooth": 40.0,
        "max_bin": 127,
        "max_cat_threshold": 64,
        "max_cat_to_onehot": 4,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "feature_fraction": 1.0,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "objective_seed": seed,
        "seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _add_features(frame: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    duration = frame["duration_ms"].astype(float).clip(lower=0.0)
    frame = frame.assign(
        duration_bucket=np.searchsorted(edges, duration.to_numpy()).astype(str),
        day=frame["date"].astype(int),
        log_duration=np.log1p(duration),
    )
    return frame


def _align_categories(train: pd.DataFrame, score: pd.DataFrame) -> None:
    for name in CATEGORICAL:
        categories = pd.Index(
            sorted(set(train[name].astype(str)) | set(score[name].astype(str)))
        )
        train[name] = pd.Categorical(train[name].astype(str), categories=categories)
        score[name] = pd.Categorical(score[name].astype(str), categories=categories)


def run(invocation: Any) -> None:
    root = invocation.input_root
    train = pd.read_csv(root / "train.csv", dtype=ID_COLUMNS)
    score = pd.read_csv(root / "score.csv", dtype=ID_COLUMNS)
    parent = pd.read_csv(
        root / "fm_baseline_predictions.csv",
        dtype={"user_id": str, "video_id": str},
    )
    if len(parent) != len(score) or not (
        parent["row_id"].to_numpy() == score["row_id"].to_numpy()
    ).all():
        raise ValueError("fm_baseline_predictions.csv does not align with score.csv")

    edges = np.quantile(
        train["duration_ms"].astype(float).clip(lower=0.0).to_numpy(),
        [index / 8.0 for index in range(1, 8)],
    )
    train = _add_features(train, edges)
    score = _add_features(score, edges)
    _align_categories(train, score)

    user_codes = train["user_id"].cat.codes.to_numpy()
    order = np.argsort(user_codes, kind="stable")
    sorted_codes = user_codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    group_sizes = np.diff(np.concatenate(([0], boundaries, [len(sorted_codes)])))

    dataset = lgb.Dataset(
        train.iloc[order][FEATURES],
        label=train["long_view"].astype(int).to_numpy()[order],
        group=group_sizes,
        categorical_feature=CATEGORICAL,
        free_raw_data=True,
    )
    model = lgb.train(
        _lgb_params(int(invocation.seed)),
        dataset,
        num_boost_round=NUM_BOOST_ROUND,
    )

    raw = pd.Series(model.predict(score[FEATURES], raw_score=True), dtype=float)
    fm = parent["score"].astype(float)
    users = score["user_id"].cat.codes
    raw_mean = raw.groupby(users).transform("mean")
    raw_std = raw.groupby(users).transform("std", ddof=0)
    fm_std = fm.groupby(users).transform("std", ddof=0)
    z = (raw - raw_mean) / raw_std
    final = fm + ALPHA * fm_std * z
    final = final.where(np.isfinite(final), fm)

    result = pd.DataFrame(
        {
            "row_id": score["row_id"].astype(int),
            "user_id": score["user_id"].astype(str),
            "video_id": score["video_id"].astype(str),
            "score": final.astype(float),
        }
    )
    if not np.isfinite(result["score"].to_numpy()).all():
        raise ValueError("candidate produced non-finite scores")
    with open(invocation.output_path, "x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
```

## Sources

Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
NeurIPS 2017. Recipe reproduced from the sibling evaluation
run_017_global_repro50 (exp_001 accepted at full fidelity, exp_002 seed-mean).
