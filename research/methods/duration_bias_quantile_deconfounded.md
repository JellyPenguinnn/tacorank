```json
{"schema_version":"1.0","method_id":"duration_bias_quantile_deconfounded","family":"duration_bias","status":"candidate","tags":["duration","deconfounding","quantile","residual"],"cost_tier":"medium","prerequisites":["baseline_parity","duration_features_legal"],"allowed_data":["train_interactions","duration_ms","user_id","video_id","author_id","tab","long_view","verified_predictions"],"prohibition_conditions":["duration_signal_not_permitted"],"sources":["https://arxiv.org/abs/2206.06003","https://dl.acm.org/doi/10.1145/3534678.3539092"]}
```

## Mechanism

Fit the residual *within* duration quantile groups rather than globally, so the
learned preference is free of duration's confounding effect on exposure while
duration's intrinsic effect on engagement is preserved.

Kuaishou's D2Q framework (Zhan et al., KDD 2022) treats duration as a
confounder that acts on two paths at once: it affects which videos are exposed,
which is bias and must be removed, and it affects engagement through the
video's intrinsic character, which is signal and must be kept. D2Q separates
them by grouping training rows into duration quantiles and fitting the target
within each group, so the model never learns "longer is better" from exposure
imbalance alone.

This is not the post-hoc calibration in `duration_bias_censored_watch_time`.
That card applies a monotone correction to a globally fitted score; this card
changes what is fitted, which is the part of D2Q that removes the confound.

## Preconditions

Executable FM parity is verified, duration is contract-permitted, and duration
varies materially inside a user's impression list rather than only between
users. The planner data profile reports this directly as
`score_within_user_duration_dispersion`; on this deployment it is 0.61, so the
axis is material.

## Allowed data

Contract-permitted training rows only. Duration quantile boundaries must be
computed from training rows strictly preceding every scored date; they must
never be fitted on the scored population.

## Expected effect

Improve within-user ordering by removing a duration-driven ranking distortion
the pointwise parent inherits from exposure imbalance. KuaiRand-Pure is drawn
from Kuaishou, the platform D2Q was designed and deployed for, so the confound
the paper describes is expected to be present in this data.

## Falsification condition

No trusted full-fidelity improvement over the parent at matched budget, or the
gain disappears once quantile boundaries are recomputed under a strict temporal
cutoff, which would indicate the boundaries themselves leaked scored-period
information.

## Do not use when

The label being modelled is reinterpreted as watch time. `duration_ms` is video
duration, not observed watch time, and the contract forbids treating it as
such. D2Q's original target is watch time; here the adaptation is that
long-view propensity is itself duration-confounded, and only that adaptation is
in scope.

## Minimal implementation

Bucket training impressions into a fixed number of duration quantiles from
past-only rows. Learn one bounded additive residual per group over the existing
permitted fields, then add it to the frozen FM parent on the original score
scale. Verify that the residual reorders items inside users' lists rather than
shifting whole lists, since a per-user constant cannot move GAUC or nDCG@5.
Keep the parent, evaluator, split, and population fixed.

## Sources

Zhan et al., "Deconfounding Duration Bias in Watch-time Prediction for Video
Recommendation", KDD 2022. Deployed on Kuaishou, the source platform for
KuaiRand-Pure.
