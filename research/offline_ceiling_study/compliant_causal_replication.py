"""Compliant causal-history LambdaRank candidate: train-split-only features.

History/state features come from TRAIN rows only. Score rows receive the
user's end-of-train state, so every feature is identical whether or not the
score file exists. Serves the model's raw score (replacement); FM is only a
per-row fallback for users with no training history at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

NUM_ROUNDS = 404
SESSION_GAP_S = 1800.0
CATS = ["author_id", "tab", "dur_bucket", "hour"]
HIST = [
    "h_u_n", "h_u_rate", "h_lv1", "h_lv2", "h_lv3", "h_pr1", "h_pr_m5",
    "h_lv_m5", "h_lv_m20", "h_dt", "h_sess_pos", "h_sess_rate",
    "h_ua_n", "h_ua_rate", "h_uv_lv", "h_click1", "h_click_m5",
    "h_vid_n", "h_vid_rate", "h_auth_n", "h_auth_rate",
]
NUMS = HIST + [
    "te_video", "te_video_n", "te_author", "te_author_n", "ud_rate",
    "v_playratio", "rel_dur", "log_dur", "day",
    "st_long_rate", "st_short_rate", "st_play_progress", "st_valid_rate",
    "st_like_rate", "st_share_rate", "st_show", "st_play", "st_avg_play_ms",
]


def _params(seed: int) -> dict:
    return {
        "objective": "lambdarank", "lambdarank_norm": False,
        "lambdarank_truncation_level": 8, "label_gain": [0, 1],
        "sigmoid": 1.0, "metric": "None", "learning_rate": 0.05,
        "num_leaves": 7, "max_depth": -1, "min_data_in_leaf": 200,
        "min_data_per_group": 50, "lambda_l1": 0.1, "lambda_l2": 10.0,
        "cat_l2": 10.0, "cat_smooth": 30.0, "max_bin": 255,
        "max_cat_threshold": 128, "feature_fraction": 0.85,
        "bagging_fraction": 0.85, "bagging_freq": 1, "seed": seed,
        "bagging_seed": seed, "feature_fraction_seed": seed,
        "data_random_seed": seed, "deterministic": True,
        "force_col_wise": True, "num_threads": 8, "verbosity": -1,
    }


def _causal_train(train: pd.DataFrame) -> pd.DataFrame:
    """Strictly-past aggregates over TRAIN rows only (lab cum-minus-own)."""
    df = train.sort_values(["user_id", "time_ms"], kind="stable")
    g = df.groupby("user_id", sort=False)
    lv = df["long_view"].astype(float)
    pr = (df["play_time_ms"] / (df["duration_ms"] + 1)).clip(0, 3)
    ck = df["is_click"].astype(float)
    gmean = float(lv.mean())

    cnt = g.cumcount().astype(float)
    cum = g["long_view"].cumsum() - lv
    df["h_u_n"] = np.log1p(cnt)
    df["h_u_rate"] = (cum + 5.0 * gmean) / (cnt + 5.0)
    for k in (1, 2, 3):
        df[f"h_lv{k}"] = g["long_view"].shift(k).fillna(-1)
    df["h_pr1"] = pr.groupby(df["user_id"], sort=False).shift(1).fillna(-1)
    sh_lv = lv.groupby(df["user_id"], sort=False).shift(1)
    sh_pr = pr.groupby(df["user_id"], sort=False).shift(1)
    sh_ck = ck.groupby(df["user_id"], sort=False).shift(1)
    roll = lambda s, w: s.groupby(df["user_id"], sort=False).rolling(
        w, min_periods=1).mean().reset_index(level=0, drop=True).fillna(-1)
    df["h_lv_m5"] = roll(sh_lv, 5)
    df["h_lv_m20"] = roll(sh_lv, 20)
    df["h_pr_m5"] = roll(sh_pr, 5)
    df["h_click1"] = sh_ck.fillna(-1)
    df["h_click_m5"] = roll(sh_ck, 5)

    dt = g["time_ms"].diff() / 1000.0
    df["h_dt"] = np.log1p(dt.clip(lower=0)).fillna(-1)
    new_sess = (dt.isna() | (dt > SESSION_GAP_S)).astype(int)
    sess_id = new_sess.groupby(df["user_id"], sort=False).cumsum()
    sg = df.assign(_s=sess_id).groupby(["user_id", "_s"], sort=False)
    spos = sg.cumcount().astype(float)
    df["h_sess_pos"] = np.log1p(spos)
    scum = sg["long_view"].cumsum() - lv
    df["h_sess_rate"] = np.where(spos > 0, scum / np.maximum(spos, 1), -1)

    ga = df.groupby(["user_id", "author_id"], sort=False)
    acnt = ga.cumcount().astype(float)
    acum = ga["long_view"].cumsum() - lv
    df["h_ua_n"] = np.log1p(acnt)
    df["h_ua_rate"] = np.where(acnt > 0, acum / np.maximum(acnt, 1), -1)
    gv = df.groupby(["user_id", "video_id"], sort=False)
    df["h_uv_lv"] = gv["long_view"].shift(1).fillna(-1)

    df = df.sort_values("time_ms", kind="stable")
    lv2 = df["long_view"].astype(float)
    gvid = df.groupby("video_id", sort=False)
    vn = gvid.cumcount().astype(float)
    vcum = gvid["long_view"].cumsum() - lv2
    df["h_vid_n"] = np.log1p(vn)
    df["h_vid_rate"] = (vcum + 20.0 * gmean) / (vn + 20.0)
    gauth = df.groupby("author_id", sort=False)
    an = gauth.cumcount().astype(float)
    aucum = gauth["long_view"].cumsum() - lv2
    df["h_auth_n"] = np.log1p(an)
    df["h_auth_rate"] = (aucum + 20.0 * gmean) / (an + 20.0)
    return df.sort_index()


def _tail_state(train: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    """Map each user's end-of-train state onto the score rows."""
    df = train.sort_values(["user_id", "time_ms"], kind="stable")
    lv = df["long_view"].astype(float)
    pr = (df["play_time_ms"] / (df["duration_ms"] + 1)).clip(0, 3)
    ck = df["is_click"].astype(float)
    gmean = float(lv.mean())
    g = df.groupby("user_id", sort=False)
    tail = pd.DataFrame(index=g.size().index)
    n = g.size().astype(float)
    s = g["long_view"].sum().astype(float)
    tail["h_u_n"] = np.log1p(n)
    tail["h_u_rate"] = (s + 5.0 * gmean) / (n + 5.0)
    tail["h_lv1"] = g["long_view"].nth(-1).astype(float)
    tail["h_lv2"] = g["long_view"].nth(-2).astype(float)
    tail["h_lv3"] = g["long_view"].nth(-3).astype(float)
    tail["h_pr1"] = pr.groupby(df["user_id"], sort=False).nth(-1)
    tail["h_pr_m5"] = pr.groupby(df["user_id"], sort=False).apply(lambda x: x.tail(5).mean())
    tail["h_lv_m5"] = lv.groupby(df["user_id"], sort=False).apply(lambda x: x.tail(5).mean())
    tail["h_lv_m20"] = lv.groupby(df["user_id"], sort=False).apply(lambda x: x.tail(20).mean())
    tail["h_click1"] = ck.groupby(df["user_id"], sort=False).nth(-1)
    tail["h_click_m5"] = ck.groupby(df["user_id"], sort=False).apply(lambda x: x.tail(5).mean())
    tail["last_time_ms"] = g["time_ms"].max().astype(float)
    for c in ("h_lv2", "h_lv3"):
        tail[c] = tail[c].fillna(-1)

    out = score[["user_id"]].merge(tail, left_on="user_id", right_index=True, how="left")
    out.index = score.index
    out["h_dt"] = np.log1p(
        ((score["time_ms"].astype(float).to_numpy() - out["last_time_ms"].to_numpy()) / 1000.0)
    )
    out["h_sess_pos"] = 0.0
    out["h_sess_rate"] = -1.0

    ua = df.groupby(["user_id", "author_id"])["long_view"].agg(["count", "sum"])
    key = pd.MultiIndex.from_frame(score[["user_id", "author_id"]])
    ua_c = ua["count"].reindex(key).fillna(0).to_numpy()
    ua_s = ua["sum"].reindex(key).fillna(0).to_numpy()
    out["h_ua_n"] = np.log1p(ua_c)
    out["h_ua_rate"] = np.where(ua_c > 0, ua_s / np.maximum(ua_c, 1), -1)
    uv = df.groupby(["user_id", "video_id"])["long_view"].last()
    kuv = pd.MultiIndex.from_frame(score[["user_id", "video_id"]])
    out["h_uv_lv"] = uv.reindex(kuv).fillna(-1).to_numpy()

    vid = df.groupby("video_id")["long_view"].agg(["count", "sum"])
    vc = vid["count"].reindex(score["video_id"]).fillna(0).to_numpy()
    vs = vid["sum"].reindex(score["video_id"]).fillna(0).to_numpy()
    out["h_vid_n"] = np.log1p(vc)
    out["h_vid_rate"] = (vs + 20.0 * gmean) / (vc + 20.0)
    auth = df.groupby("author_id")["long_view"].agg(["count", "sum"])
    ac = auth["count"].reindex(score["author_id"]).fillna(0).to_numpy()
    asum = auth["sum"].reindex(score["author_id"]).fillna(0).to_numpy()
    out["h_auth_n"] = np.log1p(ac)
    out["h_auth_rate"] = (asum + 20.0 * gmean) / (ac + 20.0)
    for c in HIST:
        out[c] = out[c].fillna(-1)
    return out[HIST]


def _static(train: pd.DataFrame, frame: pd.DataFrame, root) -> pd.DataFrame:
    """LOO target encodings from train + item statistics, applied to frame."""
    lv = train["long_view"].astype(float)
    gmean = float(lv.mean())
    is_train = np.arange(len(frame)) < len(train)

    def loo(keys, values_train, name, prior, fallback):
        groupers = [train[k] for k in keys] if len(keys) > 1 else train[keys[0]]
        g = values_train.groupby(groupers).agg(["sum", "count"])
        idx = (
            pd.MultiIndex.from_frame(frame[list(keys)])
            if len(keys) > 1
            else frame[keys[0]]
        )
        s = np.nan_to_num(g["sum"].reindex(idx).to_numpy(), nan=0.0)
        c = np.nan_to_num(g["count"].reindex(idx).to_numpy(), nan=0.0)
        own = np.zeros(len(frame))
        if name != "v_playratio":
            own_vals = frame["long_view"].fillna(0).astype(float).to_numpy()
        else:
            own_vals = (frame["play_time_ms"].fillna(0) / (frame["duration_ms"] + 1)).clip(0, 3).to_numpy()
        own = np.where(is_train, own_vals, 0.0)
        s = s - own
        c = np.maximum(c - is_train.astype(float), 0.0)
        frame[name] = (s + prior * fallback) / (c + prior)
        return c

    c1 = loo(("video_id",), lv, "te_video", 50.0, gmean)
    frame["te_video_n"] = np.log1p(c1)
    c2 = loo(("author_id",), lv, "te_author", 50.0, gmean)
    frame["te_author_n"] = np.log1p(c2)
    loo(("user_id", "dur_bucket"), lv, "ud_rate", 20.0, gmean)
    ratio_train = (train["play_time_ms"] / (train["duration_ms"] + 1)).clip(0, 3)
    loo(("video_id",), ratio_train, "v_playratio", 5.0, float(ratio_train.mean()))

    stat = pd.read_csv(root / "video_features_statistic_pure.csv", dtype={"video_id": str})
    eps = 1.0
    f = pd.DataFrame({"video_id": stat["video_id"]})
    f["st_long_rate"] = stat["long_time_play_cnt"] / (stat["play_cnt"] + eps)
    f["st_short_rate"] = stat["short_time_play_cnt"] / (stat["play_cnt"] + eps)
    f["st_play_progress"] = stat["play_progress"]
    f["st_valid_rate"] = stat["valid_play_cnt"] / (stat["play_cnt"] + eps)
    f["st_like_rate"] = stat["like_cnt"] / (stat["show_cnt"] + eps)
    f["st_share_rate"] = stat["share_cnt"] / (stat["show_cnt"] + eps)
    f["st_show"] = np.log1p(stat["show_cnt"])
    f["st_play"] = np.log1p(stat["play_cnt"])
    f["st_avg_play_ms"] = stat["play_duration"] / (stat["play_cnt"] + eps)
    frame = frame.merge(f, on="video_id", how="left")
    frame["rel_dur"] = np.log1p(frame["duration_ms"]) - np.log1p(
        frame.groupby("user_id")["duration_ms"].transform("median"))
    frame["log_dur"] = np.log1p(frame["duration_ms"])
    frame["day"] = frame["date"].astype(int)
    return frame


def run(invocation: Any) -> None:
    root = invocation.input_root
    ids = {"user_id": str, "video_id": str, "author_id": str, "tab": str}
    train = pd.read_csv(root / "train.csv", dtype=ids)
    score = pd.read_csv(root / "score.csv", dtype=ids)
    parent = pd.read_csv(root / "fm_baseline_predictions.csv",
                         dtype={"user_id": str, "video_id": str})
    if len(parent) != len(score) or not (
        parent["row_id"].to_numpy() == score["row_id"].to_numpy()
    ).all():
        raise ValueError("fm predictions misaligned")

    edges = np.quantile(train["duration_ms"].astype(float).clip(lower=0),
                        np.linspace(0, 1, 21)[1:-1])
    for f in (train, score):
        f["dur_bucket"] = np.searchsorted(edges, f["duration_ms"].astype(float)).astype(str)
        f["hour"] = (f["hourmin"] // 100).astype(str)

    train = _causal_train(train)
    score_hist = _tail_state(train, score)
    for c in HIST:
        score[c] = score_hist[c].to_numpy()

    both = pd.concat([train, score], axis=0, ignore_index=False, sort=False)
    both = _static(train, both, root)
    tr = both.iloc[: len(train)]
    sc = both.iloc[len(train):]

    for c in CATS:
        cats = pd.Index(sorted(set(tr[c].astype(str)) | set(sc[c].astype(str))))
        both[c] = pd.Categorical(both[c].astype(str), categories=cats)
    tr = both.iloc[: len(train)]
    sc = both.iloc[len(train):]

    feats = CATS + NUMS
    codes = tr["user_id"].astype("category").cat.codes.to_numpy()
    order = np.argsort(codes, kind="stable")
    scodes = codes[order]
    b = np.flatnonzero(np.diff(scodes)) + 1
    groups = np.diff(np.concatenate(([0], b, [len(scodes)])))
    dataset = lgb.Dataset(
        tr.iloc[order][feats], label=tr["long_view"].astype(int).to_numpy()[order],
        group=groups, categorical_feature=CATS, free_raw_data=True)
    model = lgb.train(_params(int(invocation.seed)), dataset, num_boost_round=NUM_ROUNDS)

    raw = model.predict(sc[feats], raw_score=True)
    fm = parent["score"].astype(float).to_numpy()
    known = sc["h_u_n"].to_numpy() > 0
    final = np.where(np.isfinite(raw) & known, raw, fm)

    result = pd.DataFrame({
        "row_id": score["row_id"].astype(int).to_numpy(),
        "user_id": score["user_id"].astype(str).to_numpy(),
        "video_id": score["video_id"].astype(str).to_numpy(),
        "score": final.astype(float),
    })
    if not np.isfinite(result["score"].to_numpy()).all():
        raise ValueError("non-finite scores")
    with open(invocation.output_path, "x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
