# CONTEXT — decision log

Last updated: 2026-08-31

## Merge of main into final_version (2026-08-31)

Merged origin/main into `final_version` with `-X ours` (user directive:
final_version wins on conflicts). One breakage: our branch's
`classify_failure` calls `_bounded_evidence` whose definition lived only on
main; the ours-side resolution dropped it. Grafted main's definition back
(commit 68bfb29). Focused suite (tests/schemas, tests/sre, tests/recovery)
green: 111 passed. Note the 49 recovery failures existed on final_version
before the merge too — the branch was committed calling a helper it never
defined.

## Manual baseline-improvement effort in `lab/` (2026-08-31)

Goal: user asked for a hands-on iterative improvement of the
`kuairand-starter-kit` baseline (FM, test primary 0.5946), target 0.7.
Copied kit into `lab/` (evaluate.py untouched; it is the sole scorer).
Oracle ceiling is primary 0.8645, so 0.7 means eating ~54% of the
random→oracle interval — for calibration, FM ate 31%.

### What worked (in order of discovery)

- LightGBM LambdaRank with **causal per-impression history features**
  (`lab/rank3.py`): every row scored using only logs strictly before its
  `time_ms` — user long_view EWMAs, previous play ratios, session position,
  seen-this-video-before, user×author running rate, causal video popularity.
  Plus LOO target encodings and `video_features_statistic_pure.csv` ratios
  (`long_time_play_cnt/play_cnt` etc.).
- Key hyper findings: **small trees win** (num_leaves 7–15, lr 0.05);
  dropping `video_id`/`music_id` as raw categoricals (keep their TEs) gave
  +0.002; lambdarank truncation 8 beats 40; capacity increases always hurt.
- ~~Train on train+valid for the test model~~ — **retracted**: violates FAQ
  2.9.2 (see compliance correction above).
- Best single model: valid 0.6122, **test 0.6045** (GAUC 0.6729,
  nDCG@5 0.5362) — beats FM by +0.010 and the entire prior harness plateau
  (0.6015).

### Dead ends (do not retry without new ideas)

- Any train-window aggregate feature **including the row's own label**
  (uv_pos etc.) — model collapses onto it; LOO versions required.
- FM score as a GBDT input feature: in-sample FM train scores poison
  training (valid 0.5978 < no-fm).
- BPR-MF embedding dot as feature: memorizes (u,v) train pairs even with
  2-fold OOF; DIN-style profile cosines leak window membership. 0.5878.
- Causal GRU sequence model (`lab/seq.py`): trains (loss 0.64→0.47) but
  valid stays ~random (0.49) — memorizes, transfers nothing. Needs real
  research investment (regularization, ranking loss) to be viable.
- Big trees / more capacity / truncation 40 all regress.

### Compliance correction (2026-08-31, after user flag)

FAQ 2.9.2: training data for KuaiRand-Pure is the **train split only**
(0408–0421); valid supplies feedback (tuning/early-stop/blend weights), not
training rows, and `log_random_*` is banned as training data at any date.
The first final artifact below trained on ≤0428 and was **discarded**;
`rank3.py`/`cb.py` now default hist_end to 20220421 for every target.
Open question sent upward: causal features read valid-window logs as
inference-time history for test rows — ask organizers if that is allowed,
else rebuild features train-window-only.

### FINAL — compliance rebuild (2026-08-31 night): sealed test 0.5960

FAQ 2.9.3 bans any use of test labels incl. feature statistics → all
rolling eval-window history features illegal; every earlier test number
(0.6045–0.6144) is a dev artifact, not reportable. `lab/rank_frozen.py`:
train rows roll within train, eval rows get user state frozen at 0421.
Members re-tuned on valid (blend 0.6039 = ens + 0.6×xen + 0.2×FM; frozen
catboost regressed to 0.5996 and was dropped). Test scored ONCE, sealed:
**primary 0.5960** (GAUC 0.6623, nDCG@5 0.5297) — +0.0014 over FM.
`lab/submission.csv` = this sealed blend, format-checked. Legal next
levers in lab/PLAYBOOK.md header (recency-weighted TEs, staleness
features, train-window sequence embedding).

### Superseded research history (rolling features, NOT compliant)

### Third wave (2026-08-31 night): 0.6144 test

User confirmed eval-window feature input is allowed. CatBoost push:
keeping ALL categoricals (incl. video_id/music_id) + exact 300-tree cut
found by sweeping ntree_end on true primary → single model 0.6170 valid /
0.6129 test (sharp overfit cliff after 300; transfers exactly since
valid/test share the trained model). Frozen blend 0.5×lgb-ens + 1.0×xendcg
+ 2.0×cb-allcats → **test 0.6144**, submission.csv rebuilt and checked.
Depth 8 regressed; old DROP_CATS catboost weight went to 0.

### Second wave (2026-08-31 evening): 0.6121 test, compliant

Diversity ensemble broke 0.61: lgb-5-seed-ens + 1.0×rank_xendcg +
1.0×CatBoost-YetiRank (per-user z, weights frozen on valid at 0.6172).
Test: **primary 0.6121** (GAUC 0.6830, nDCG@5 0.5411). CatBoost alone is
0.6120 test / 0.6151 valid — strongest single model. All members train on
the train split only. New dead ends: graded play-ratio labels (0.6032),
extra causal features h_lv_m50/h_day/h_auth (0.6104). `lab/PLAYBOOK.md` is
the living recipe file. Catboost seeds 1–2 (3-seed member, est. +~0.001) were aborted on user
request — first untried item if resuming.

### Superseded first wave (2026-08-31)

Frozen recipe: 5-seed rank3 (leaves 7, lr 0.05, trunc 8, fixed 404 rounds,
DROP_CATS=video_id,music_id, hist_end 20220428 for test), per-user z-average,
+ 0.1 × per-user-z FM (weight chosen on valid where 0.1–0.2 tied at 0.6134).

- valid primary 0.6134 | **test primary 0.6056** (GAUC 0.6745, nDCG@5 0.5367)
- vs FM baseline 0.5946 (+0.0110) and prior harness plateau 0.6015 (+0.0041)
- `lab/submission.csv` written and passes `submit.py --check` (the checker's
  final print crashes on cp1252 console encoding only — set
  PYTHONIOENCODING=utf-8 to see the checkmark).

Target 0.7 was not reached and is judged unreachable with tabular GBDT over
these features: 0.7 = 58% of the random→oracle interval (oracle 0.8645);
current best eats 33%. The one direction with plausible headroom left is a
properly regularized sequence model (the naive causal GRU memorized and
transferred nothing).
