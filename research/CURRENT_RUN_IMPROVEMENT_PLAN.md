# TacoRank Current-Run Improvement Playbook

**Schema version:** 1.0
**Status:** predefined, human-reviewed direction for the current run
**Primary task:** within-user ranking of `long_view`
**Primary score:** contract-defined mean of GAUC and nDCG@5
**Core rule:** one experiment changes one research mechanism.

```json
{
  "schema_version": "1.0",
  "rule_order": [
    "output_rejected",
    "suspicious_or_compromised",
    "no_op",
    "unstable",
    "promotion_required",
    "non_public_or_incomplete",
    "pairwise_gauc_up_ndcg_down",
    "pairwise_gauc_down_ndcg_up",
    "pairwise_both_up",
    "meaningful_no_gain",
    "trusted_improvement",
    "trusted_regression"
  ],
  "family_order": [
    "objective",
    "temporal_history",
    "multitask",
    "duration_bias",
    "features",
    "model",
    "sampling",
    "ensemble",
    "evaluation",
    "other"
  ],
  "method_order": {
    "objective": ["objective_pairwise_bpr", "objective_listwise_user_softmax"],
    "temporal_history": ["temporal_history_compact"],
    "multitask": ["multitask_single_auxiliary"],
    "duration_bias": ["duration_bias_censored_watch_time"],
    "features": ["temporal_drift_past_only"],
    "model": ["model_compact_ranker"],
    "ensemble": ["ensemble_diverse_residual_candidate", "ensemble_confirmed_members"],
    "evaluation": ["evaluation_random_exposure_robustness"]
  }
}
```

The JSON block above is the executable control surface. The harness validates
its rule identifiers and ordering before building `PlannerContext`; the prose
below explains the evidence semantics and research rationale for humans.

This file tells the Planner how to turn verified evaluation feedback into the
next research direction. It is seed knowledge, not dynamic memory. During a
run it is read-only: outcomes belong in `events.jsonl`, reusable conclusions
belong in `lesson.recorded`, and exact code belongs in Git.

The context builder should include the applicable section and referenced
method cards in `PlannerContext`. The Planner must still return one validated
`ExperimentSpec`; it must not write events, edit this file, or use hidden-final
feedback.

## 1. Inputs from the memory and evaluation schema

Use only verified fields already present in the current schemas:

- `output.checked.accepted`, checks, and violations;
- `evaluation.completed.population` and `fidelity`;
- `metric_set.metrics`, `primary_score`, `baseline_delta`, `parent_delta`, and
  `previous_best_delta`;
- `prediction_change.spearman_vs_parent` and `changed_row_fraction`;
- label-free `diagnostic_metrics`, especially FM correlation, residual spread,
  within-user rankability, and repeated-item personalization;
- `trust.verdict`, `trust.integrity`, `trust.stability`, and `trust.flags`;
- `experiment.decided.decision`, `parent_eligible`, and `best_eligible`;
- contract `epsilon`, patience, budgets, editable paths, and allowed data.

Never infer a direction from raw logs, an unverified patch, proxy-only reward,
hidden-final results, or a metric whose evaluator/contract hash does not match
the frozen run contract.

## 2. Mandatory decision order

Apply these rules from top to bottom. The first matching rule wins.

| Priority | Evidence condition | Required response | Research branch? |
| --- | --- | --- | --- |
| 0 | `output.checked.accepted = false` | Fix or abandon the output/contract failure through recovery. | No |
| 1 | Trust integrity is `compromised`, or verdict is `suspicious` | Quarantine the result; investigate data, evaluator, or leakage. | No |
| 2 | Verdict is `no_op`, or prediction change is below the contract's no-op threshold | Record a terminal null result. Let the legal-choice ranker select either one same-mechanism reimplementation from the trusted parent or an independent mechanism. | Trusted parent only |
| 3 | Stability is `unstable` | Confirm seeds or simplify/regularize the same mechanism. | No new family |
| 4 | Fidelity is `smoke` or `proxy` | Promote only if deterministic promotion rules pass. | No parent promotion |
| 5 | Full public result is trusted and improves the parent by more than `epsilon` | Accept; confirm once if stability is only `single_seed`, then deepen the same family. | Yes |
| 6 | Full public result changes predictions meaningfully but gain is within `[-epsilon, +epsilon]` | Treat as inconclusive/noise; record the result and move to the next independent mechanism. | Yes |
| 7 | Full public result is trusted and worse than `-epsilon` | Treat the tested mechanism as falsified under its stated conditions; do not tune it indefinitely. | Yes |

Only a full, verified, public-validation result with `trust.verdict = accepted`
and `trust.integrity = clean` may create a future parent. A positive proxy score
can justify more evaluation, but never a new trusted branch.

### Hard prune, soft prune, and portfolio retention

Canonical `parent_eligible` and `best_eligible` remain unchanged. Person 1 may
derive two narrower, non-checkpoint permissions from verified evidence:

- **hard prune:** rejected output, suspicious/compromised integrity, no-op,
  unstable result, invalid/retracted lineage, or primary regression worse than
  `max(5 * epsilon, 0.01)`. Never branch, refine, or ensemble the result node.
  After the first no-op only, the tree planner may select one reimplementation
  of the same mechanism from its last trusted parent; a second no-op retires it;
- **soft prune:** clean accepted output at proxy/full fidelity, meaningful
  prediction change, and either primary delta above that regression floor or a
  component-metric trade-off. Retain the node as evidence, not as a checkpoint;
- **bounded refinement:** a soft-pruned node with a documented metric trade-off
  may receive at most one child when a method card names the follow-up;
- **ensemble candidate:** a soft-pruned node may enter one fixed blend test only
  when its prediction Spearman magnitude versus the parent is below `0.98`.

These permissions never change validation-best selection. A soft node remains
`parent_eligible = false` and `best_eligible = false`; Person 1 must label the
proposal as a refinement or ensemble action, and the plan validator must
recompute eligibility from the same verified context.

## 3. Metric-shape diagnosis

Do not route on `primary_score` alone. Compare both component deltas against
the parent, using the contract's seed/noise tolerance.

| GAUC delta | nDCG@5 delta | Likely interpretation | Preferred next action |
| --- | --- | --- | --- |
| positive | positive | Broad pair ordering and top-5 placement both improved. | Confirm, then refine the same family before switching. |
| positive | negative | Broad within-user separation improved but top ranks worsened. | Try top-weighted/listwise or hybrid ranking loss; inspect top-5 errors. |
| negative | positive | Top-5 placement improved while general positive-negative ordering degraded. | Blend listwise/top-k emphasis with pairwise loss; avoid a pure top-k overfit. |
| near zero | near zero, predictions changed | Mechanism has little signal at current fidelity. | Move to the next independent family. |
| any | any, predictions barely changed | Terminal null result; it does not establish that the implementation is broken. | Let the tree planner rank one bounded same-mechanism reimplementation against independent mechanisms. |
| inconsistent across seeds | inconsistent across seeds | Variance dominates estimated gain. | Confirm or simplify; do not promote. |

Metric-specific interpretation is diagnostic, not proof. GAUC excludes users
with all-positive or all-negative impressions and weights eligible users by
positive count; nDCG@5 averages top-5 quality across all users, with all-negative
users contributing zero. Always inspect per-user cohorts before claiming a
mechanism.

Recommended cohorts:

- all-negative, all-positive, and mixed-label users;
- short versus long impression lists;
- low versus high history length;
- video-duration buckets;
- date/hour buckets;
- standard-policy versus random-exposure logs, when permitted.

## 4. Direction priority for this run

The default order is expected-value per unit cost, not an instruction to try
every item. Skip any direction whose prerequisites fail or whose estimated
cost does not fit the remaining budget.

### Direction 0 — baseline and evaluator parity

Before research, reproduce the frozen random, popularity, and FM checks. The
random model should be near the published lower-bound tolerance. The editable
candidate must also reproduce the official FM prediction bytes on smoke,
proxy, full validation, and final-inference views; a good evaluator score from
a separate file is not baseline parity. Confirm the setup-generated parity
receipt, row alignment, duplicate preservation, finite scores,
contract/evaluator hashes, and seed variance.

If parity fails, stop research and fix the harness. A broken evaluator can make
every later direction look productive.

### Direction 1 — objective alignment: pairwise first

**Why first:** the baseline optimizes pointwise binary log loss, while both
contract metrics depend only on within-user order. BPR-style pairwise logistic
loss directly trains a positive impression to score above a negative impression.

**First experiment:** keep the setup-verified official FM score and all
data/splits fixed. Train only a bounded additive residual with deterministic
within-user pairwise logistic loss:

```text
loss(u, i+, i-) = -log sigmoid(score(u, i+) - score(u, i-))
```

Use only observed impressions from the same user. Cap pairs per user and sample
deterministically so users with many interactions do not dominate. Users with
only one label provide no pairwise signal and must be skipped for this loss.
Initialize both latent factor sides with deterministic small non-zero values;
zero-initializing both sides makes every latent gradient zero. Before accepting
the implementation, require non-zero residual variance, meaningful
within-user score variation, and repeated-item user personalization.

**Do not:** pair across users, treat unexposed items as negatives, discard the
FM parent, change the evaluator, or combine a new model architecture in the
same experiment.

**Success:** a trusted full result exceeds the parent by more than contract
`epsilon`, with neither component metric showing a material regression.

**Falsifier:** predictions changed meaningfully, but a trusted full result fails
to improve beyond noise. Move on instead of doing an open-ended learning-rate
or embedding-size sweep.

**Second objective experiment, only when justified:** use a user-list softmax/
ListNet-style objective or a small pairwise-plus-listwise hybrid. Prefer this
when GAUC improves but nDCG@5 regresses, because it can put more emphasis on
the ordered list/top ranks. All-negative and all-positive lists contain no
binary ordering information and need explicit handling.

### Direction 2 — compact, leakage-safe user history

**Why second:** KuaiRand contains timestamps and repeated interactions, and
target-aware history models such as DIN can represent candidate-specific user
interest. However, KuaiRand-Pure has incomplete sequences relative to the full
27K/1K releases, so a large lifelong-sequence model is not the first test.

**First experiment:** add one compact history representation while keeping the
winning objective/model fixed. Start with recent-item/author embeddings,
recency-weighted pooling, history length, and deterministic truncation/padding.
Only interactions strictly earlier than the target row may enter its history.

For the first safe implementation, construct validation/test history from the
training window only. Rolling use of later observed feedback is allowed only if
the frozen contract says that feedback is available at inference time.

**Escalation:** if compact history improves cleanly, try target-conditioned DIN-
style attention. Use SIM/long-sequence retrieval only if sequence coverage and
runtime evidence show that the compact window is the bottleneck.

**Falsifier:** no gain over a no-history control, improvement disappears under
strict temporal cutoff, or any future-interaction leakage is detected.

### Direction 3 — one auxiliary task, then guarded multi-task sharing

**Why third:** KuaiRand provides multiple feedback signals, which can regularize
the `long_view` representation. Multi-task learning can also cause negative
transfer, so begin with one auxiliary target rather than a large MMoE/PLE stack.

**First experiment:** preserve `long_view` as the only decision-bearing head.
Add exactly one contract-permitted auxiliary label, a masked auxiliary loss,
and a fixed documented loss weight. Choose the auxiliary using training-only
coverage/correlation and UI-policy semantics; do not assume `is_click` is
meaningful in every interface.

Candidate order:

1. a dense engagement/completion signal with clear availability;
2. `is_click` when its UI semantics and missingness are valid;
3. sparse actions such as like/follow/comment/forward only after coverage checks.

Track primary-head GAUC/nDCG, gradient scale, label coverage, and per-task loss.
If the auxiliary helps itself but hurts `long_view`, reject it as negative
transfer. Try MMoE or PLE only after a shared-bottom single-auxiliary model shows
useful but conflicting task gradients.

### Direction 4 — watch-time and duration bias

**Why fourth:** observed watch time is truncated by video duration for completed
videos, so raw regression can confound interest with duration. CWM models a
counterfactual watch time and learns through a counterfactual likelihood.

**Prerequisite diagnostics:** verify availability and units of watch time and
duration; completion/censoring rate by duration bucket; relation between
`long_view`, watch time, duration, and explicit feedback; and contract legality.

**First experiment:** use a small CWM-inspired censored-duration auxiliary or
interest representation while retaining the original `long_view` primary head.
Compare against simpler controls such as duration buckets and completion-rate
normalization. A duration feature alone is not evidence of debiasing.

**Do not directly port the CWM repository as the baseline.** Its released code
uses an old PyTorch stack and its own reconstructed target/evaluation choices.
Reuse the mechanism, then implement it against this repository's frozen label,
split, evaluator, and resource limits.

**Falsifier:** improvement vanishes across duration cohorts, long-duration bias
increases, or the auxiliary improves watch-time fit while primary ranking falls.

### Direction 5 — temporal context and distribution drift

The split is chronological, so first measure drift rather than blindly adding
`date` and `hourmin`. Compare label rate, item/author frequency, unknown rate,
duration mix, and baseline residuals by time bucket.

Low-cost experiments, one at a time:

1. recency-decayed item/author statistics computed from past data only;
2. candidate-time × item/author interactions;
3. recent-window versus full-window training weights.

Pure global time offsets often do not change within-user order when all
impressions share the same time context. Require an interaction that can change
relative item scores. Never compute aggregate features with future rows.

### Direction 6 — model family change

Try DeepFM, DCN, or xDeepFM only after loss, history, or multi-task evidence
identifies useful nonlinear interactions. Existing local ablations show that
larger FM embedding capacity and broad static-feature expansion do not improve
the baseline; a model swap must test a mechanism, not merely add parameters.

Keep data, objective, training budget, and seed fixed. Start with the smallest
DCN or DeepFM that can express the hypothesized interaction. Prefer DCN when
explicit bounded-degree crosses are the hypothesis; prefer DeepFM when joint
low/high-order interactions are the hypothesis; use xDeepFM only after those
controls because it adds complexity.

**Falsifier:** no trusted gain at matched budget, higher variance, or apparent
gain caused only by extra training time/parameters.

### Direction 7 — random-exposure robustness evaluation

The random-exposure log is valuable because it weakens exposure-policy bias.
Use it as a frozen robustness population, not as a substitute for the contract
primary validation score.

Protocol:

1. freeze the candidate before evaluating the random log;
2. fit no preprocessing/statistics on this evaluation population;
3. verify label, feature, date, and row-alignment compatibility;
4. report metric direction, cohort changes, and uncertainty separately from
   standard public validation;
5. do not promote solely on the random-log score unless the frozen contract
   explicitly makes it decision-bearing.

A standard-log gain that reverses on random exposure is evidence of possible
policy overfitting, not automatic proof that the model is worse. Investigate
the population and propensity differences before changing training.

### Direction 8 — ensemble only confirmed complements

Consider rank averaging only after at least two full, trusted, reproducible
candidates improve through different mechanisms and have complementary errors.
Record exact member commits and weights. Reject the ensemble if it does not beat
the best member beyond noise or if one member is suspicious/unstable.

## 5. Concrete routing after each major direction

| Completed experiment | Trusted outcome | Next direction |
| --- | --- | --- |
| Pairwise objective | Both metrics improve beyond noise | Confirm once, then test one listwise/hybrid refinement or move to history. |
| Pairwise objective | GAUC up, nDCG@5 down | Listwise/top-weighted or hybrid objective on the same representation. |
| Pairwise objective | Meaningful predictions, no gain | Compact user history. |
| Compact history | Clean improvement | Target-conditioned attention; retain strict cutoff. |
| Compact history | No gain | One auxiliary task. |
| Single auxiliary | Primary improves | Test one additional auxiliary or guarded MMoE/PLE only if task conflict is observed. |
| Single auxiliary | Auxiliary improves, primary worsens | Reject negative transfer; try a different auxiliary or move to duration. |
| Duration correction | Improves across duration cohorts | Deepen the censored-watch mechanism. |
| Duration correction | Aggregate gain only in long videos | Treat as possible duration bias; do not promote until cohort-safe. |
| Temporal-drift feature | Clean cheap gain | Keep it as a component, then consider model interactions. |
| Model swap | No matched-budget gain | Stop architecture search; do not sweep capacity. |
| Any two confirmed candidates | Complementary residuals | Test a small rank-average ensemble. |

## 6. Planner proposal checklist

Every proposed experiment must state:

- one falsifiable hypothesis and one family;
- verified parent experiment and exact parent commit;
- the evaluation fields that triggered this direction;
- one method-card ID and supporting event IDs from `PlannerContext`;
- target files and stage, with no protected paths;
- smoke → proxy → full fidelity plan;
- expected mechanism and metric-shape prediction;
- cohort checks and no-op check;
- success threshold tied to contract `epsilon`;
- explicit falsification condition;
- bounded cost and deterministic seed/pair/history policy;
- a duplicate key based on normalized parent + family + change.

The Planner must choose `blocked` rather than inventing a direction when the
contract, parent, evidence, method prerequisite, or budget is unresolved.

## 7. Current first three experiments

Unless newer verified evidence in `PlannerContext` overrides this sequence:

1. **BPR-style within-user pairwise loss on the current FM.** No new features or
   architecture. This is the highest-value alignment test.
2. **Listwise/hybrid objective only if pairwise shows top-5 weakness; otherwise
   compact strict-cutoff history.** Let the metric shape decide.
3. **Compact history or one auxiliary task**, whichever remains untested after
   experiment 2. Do not combine them.

After each full evaluation, re-enter the mandatory decision order in section 2.

## 8. Sources and rationale

- Rendle et al., [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://mlanthology.org/uai/2009/rendle2009uai-bpr/).
- Cao et al., [Learning to Rank: From Pairwise Approach to Listwise Approach](https://mlanthology.org/icml/2007/cao2007icml-learning/).
- Zhou et al., [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978).
- Gao et al., [KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](https://arxiv.org/abs/2208.08696) and the [official dataset repository](https://github.com/chongminggao/KuaiRand).
- Ma et al., [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://doi.org/10.1145/3219819.3220007).
- Tang et al., [Progressive Layered Extraction](https://doi.org/10.1145/3383313.3412236).
- Zhao et al., [Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time](https://arxiv.org/abs/2406.07932) and the [official CWM implementation](https://github.com/hyz20/CWM).
- Wang et al., [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123).
- Guo et al., [DeepFM](https://arxiv.org/abs/1703.04247).
- Lian et al., [xDeepFM](https://arxiv.org/abs/1803.05170).

Local benchmark facts, metric definitions, baseline variance, oracle headroom,
and tested dead ends come from `docs/KUAIRAND_STARTER_KIT.md`,
`kuairand-starter-kit/evaluate.py`, and the frozen competition contract.
