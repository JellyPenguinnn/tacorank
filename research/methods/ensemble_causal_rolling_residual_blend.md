```json
{"schema_version":"1.0","method_id":"ensemble_causal_rolling_residual_blend","family":"ensemble","status":"candidate","tags":["ensemble","rolling_feedback","causal_history","residual","out_of_time"],"cost_tier":"high","prerequisites":["baseline_parity","strict_temporal_cutoff","standard_public_evaluation_complete","rolling_feedback_mode_declared"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","hourmin","duration_ms","long_view","is_click","is_like","is_follow","is_comment","is_forward","is_hate","play_time_ms","profile_stay_time","comment_stay_time","is_profile_enter","verified_predictions"],"prohibition_conditions":["rolling_feedback_mode_undeclared","future_or_self_outcome_leakage","adaptive_validation_weight_search","test_label_or_hidden_feedback"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["ROLLING_BLEND_062_PLAYBOOK.md","EXPERIMENT_SUMMARY.md","PLAYBOOK.md"]}
```

## Mechanism

Construct one strict causal rolling-history feature mode. Every user, user-video,
user-author, session, item, time-gap, and engagement feature must use only rows
strictly earlier than the scored row, with a deterministic policy for rows that
share a timestamp. Train a diverse compact stage-one set consisting of multiple
seeded LambdaRank members, rank_xendcg, and CatBoost YetiRank with user query
groups. Then fit frozen-history LightGBM, rank2, and a compact DIN-style
positive/negative sequence and time-context correction member.

Per user, use the exact sample z-score with `ddof=1` and the frozen sparse blend
`Z(lab_base) - 0.40*Z(frozen_lgb) - 0.10*Z(rank2) + 0.15*Z(DIN50)`. Negative
weights are correction vectors and must not invert labels. This is one integrated
mechanism, not a license for per-slice blend search.

## Preconditions

Baseline parity, a strict chronological cutoff, and a completed protected public
evaluation are available. The serving contract explicitly names whether earlier
feedback is available while scoring later rows. If it does not, use a separate
train-window-only causal-history method and do not silently consume online
outcomes.

## Allowed data

Only the listed contract-approved interaction, time, duration, engagement, and
verified-parent-prediction fields. No public-validation labels, test labels,
future rows, or hidden feedback may enter feature construction, fitting,
normalization, weight selection, or early stopping.

## Expected effect

Improve within-user GAUC and nDCG@5 by combining causal preference signals with
diverse compact rankers and correcting complementary residual ordering under
temporal drift.

## Falsification condition

Reject on any self/future leakage, timestamp ambiguity, invalid per-user
normalization, no trusted full-fidelity gain beyond `epsilon`, later-temporal
regression, or a gain concentrated in a small date slice.

## Do not use when

Rolling feedback is not explicitly authorized, a chronological or same-timestamp
policy cannot be reconstructed, a member is in-sample or future-fitted, or the
result depends on validation-selected weights. Do not use test labels or hidden
final feedback for any decision.

## Minimal implementation

Freeze the history cutoff, batch-order rule, stage-one member list, seeds,
per-user `ddof=1` z-score, sparse coefficients, and fallback behavior before
running the smoke → proxy → full ladder. Audit leakage and row alignment before
protected evaluation; compare both metric components, later temporal slices,
gain concentration, and within-user rank movement. Treat the supplied local
notes' validation and one-shot test scores as hypothesis evidence only.

## Sources

- `ROLLING_BLEND_062_PLAYBOOK.md`
- `EXPERIMENT_SUMMARY.md`
- `PLAYBOOK.md`
