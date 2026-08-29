# Person 5 — Complete Output Specification

**Against RankForge Memory Schema v1.0.**
Every artifact Person 5 produces, fully specified. This is the reference the orchestrator validates against and Person 1 codes against.

---

## 0. Ownership summary

Per §18, Person 5 owns `MetricSet`, `evaluation.completed`, trust verdicts, `experiment.decided`, lessons, and resource/intervention reports.

| # | Output | Kind | Canonical? | Primary consumer |
|---|---|---|---|---|
| 1 | `baseline.verified` | event | yes | run gate |
| 2 | `evaluation.completed` | event | yes | ledger, judges |
| 3 | `experiment.decided` | event | yes | state machine |
| 4 | `best.updated` | event | yes | trusted-best pointer |
| 5 | `lesson.recorded` | event | yes | planner context |
| 6 | `lesson.status_changed` | event | yes | planner context |
| 7 | `metrics` artifact | artifact | yes (hashed) | judges, charts |
| 8 | `delta_vector` artifact | artifact | yes (hashed) | orthogonality, redundancy |
| 9 | `submission` artifact | artifact | yes (hashed) | final deliverable |
| 10 | `report` artifacts | artifact | yes (hashed) | six charts |
| 11 | `EvaluationProjection` | derived | **no** | Person 1 |
| 12 | `LESSONS.md` | derived | **no** | humans |
| 13 | `SUMMARY.md` | derived | **no** | judges |

Person 5 **constructs and validates** payloads; the orchestrator is the sole appender (§3). Derived outputs are pure functions of the folded ledger and may be deleted and regenerated at any time (§2).

---

## 1. `baseline.verified`

Emitted once, before any experiment. This is the P0 gate: if it fails, the run does not start.

```json
{
  "evaluator_sha256": "<64 lowercase hex>",
  "contract_sha256": "<64 lowercase hex>",
  "data_manifest_sha256": "<64 lowercase hex>",
  "independent_metric_check": {
    "reimplemented": true,
    "max_abs_deviation": 3.7e-11,
    "tolerance": 1e-9,
    "passed": true
  },
  "reference_scores": [
    {"model": "random",         "population": "public_validation", "expected": 0.4834, "observed": 0.4834, "passed": true},
    {"model": "item_popularity","population": "public_validation", "expected": 0.5807, "observed": 0.5807, "passed": true},
    {"model": "fm_official",    "population": "public_validation", "expected": 0.6016, "observed": 0.6016, "passed": true},
    {"model": "random",         "population": "hidden_reference",  "expected": 0.4753, "observed": 0.4753, "passed": true},
    {"model": "item_popularity","population": "hidden_reference",  "expected": 0.5715, "observed": 0.5715, "passed": true},
    {"model": "fm_official",    "population": "hidden_reference",  "expected": 0.5946, "observed": 0.5946, "passed": true}
  ],
  "seed_independence_check": {
    "model": "fm_official",
    "seeds": [0, 1, 2],
    "observed_std": 0.00081,
    "expected_std": 0.0008,
    "tolerance_factor": 3.0,
    "passed": true
  },
  "population_manifest": {
    "internal_proxy":     {"rows": 100159, "users": 24262, "median_impressions": 5, "zero_positive_fraction": 0.262},
    "public_validation":  {"rows": 124909, "users": 22377, "median_impressions": 4, "zero_positive_fraction": 0.303},
    "unbiased_audit":     {"rows": 489612, "users": 21118, "date_range": "20220422-20220428"}
  },
  "all_passed": true
}
```

| Field | Type | Constraint |
|---|---|---|
| `independent_metric_check.max_abs_deviation` | float | must be `< tolerance` |
| `reference_scores[].observed` | float | must equal `expected` within ±0.0001 |
| `seed_independence_check.observed_std` | float | must be within `tolerance_factor` of `expected_std`, **and not near zero** |
| `all_passed` | bool | false blocks `run.status = running` |

**Why the seed check is a gate.** If a random source is not wired to the seed (data shuffle, feature subsample, model init, bagging), the observed std collapses toward 0. The Ladder's `eta = max(2·stderr, 0.0016)` then falls to the floor, the gate silently weakens, and every score looks reassuringly stable. Observed std below `0.0002` fails the check.

---

## 2. `evaluation.completed`

**One event per (experiment, population, fidelity, seed).** Three seeds at full public validation produce three events.

### 2.1 Full payload

```json
{
  "output_checked_event_id": "evt_000032",
  "prediction_artifact_id": "art_8f31a9c2",
  "metrics_artifact_id": "art_c40b17de",
  "delta_vector_artifact_id": "art_71e9a3f0",

  "population": "public_validation",
  "population_variant": "val_a",
  "fidelity": "full",
  "seed": 0,
  "public_query_index": 4,

  "evaluator_sha256": "<64 hex>",
  "contract_sha256": "<64 hex>",
  "data_manifest_sha256": "<64 hex>",

  "metric_set": { "...": "see 2.2" },

  "baseline_delta": 0.0088,
  "parent_delta": 0.0071,
  "previous_best_delta": 0.0071,

  "prediction_change": {
    "spearman_vs_parent": 0.71,
    "changed_row_fraction": 0.94,
    "parent_prediction_artifact_id": "art_5a2c9911"
  },

  "cross_population": {
    "internal_proxy_delta": 0.0063,
    "unbiased_audit_delta": 0.0058,
    "val_b_delta": null
  },

  "trust": { "...": "see 2.3" },

  "resource_delta": { "...": "ResourceDelta per §7.2" }
}
```

### 2.2 `MetricSet` — complete metric registry

Per §7.3, metric names are frozen by `COMPETITION.md`. Extra metrics are legal **only** if declared there as diagnostic metrics.

```json
{
  "metrics": {
    "gauc": 0.6702,
    "ndcg_at_5": 0.5506,

    "slice_primary_impr_le2": 0.5981,
    "slice_primary_impr_3_5": 0.6120,
    "slice_primary_impr_6_12": 0.6244,
    "slice_primary_impr_gt12": 0.6398,
    "slice_n_impr_le2": 7250,
    "slice_n_impr_3_5": 6015,
    "slice_n_impr_6_12": 3980,
    "slice_n_impr_gt12": 1032,

    "slice_primary_pos1": 0.6041,
    "slice_primary_pos2": 0.6183,
    "slice_primary_pos3plus": 0.6355,
    "slice_n_pos1": 5120,
    "slice_n_pos2": 3844,
    "slice_n_pos3plus": 3968,

    "slice_rank_tau_short": 0.4712,
    "slice_rank_tau_mid": 0.5238,
    "slice_rank_tau_flat18": 0.5601,
    "slice_n_tau_short": 4620,
    "slice_n_tau_mid": 25731,
    "slice_n_tau_flat18": 94558,

    "slice_rank_pop_cold": 0.4890,
    "slice_rank_pop_warm": 0.5401,
    "slice_rank_pop_hot": 0.5722,
    "slice_n_pop_cold": 41302,
    "slice_n_pop_warm": 41680,
    "slice_n_pop_hot": 41927,

    "drift_primary_slope": -0.00021,

    "unreachable_user_fraction": 0.303,
    "gauc_user_coverage": 0.578,
    "score_unique_fraction": 0.9714,
    "gain_concentration_top10pct": 0.41
  },
  "primary_metric_name": "primary",
  "primary_score": 0.6104
}
```

**Naming convention carries meaning.** `slice_primary_*` are **exact decompositions** — GAUC and nDCG are means over users, so grouping users partitions the metric exactly. `slice_rank_*` are **diagnostics, not decompositions** — a video appears in many users' lists, so its "contribution" has no unique definition. Those use the normalised within-group rank of each positive instead. The prefix makes the distinction unmissable in the ledger and in the README.

| Metric group | Type | Range | Constraint |
|---|---|---|---|
| `gauc`, `ndcg_at_5` | float | [0,1] | required, exactly once |
| `slice_primary_*` | float | [0,1] | support-weighted mean must reconstruct `primary_score` within 1e-9 |
| `slice_rank_*` | float | [0,1] | diagnostic only, no reconstruction constraint |
| `slice_n_*` | int | ≥0 | user counts for `impr`/`pos`, row counts for `tau`/`pop` |
| `drift_primary_slope` | float | any | OLS slope of per-day primary across the 7 validation days |
| `unreachable_user_fraction` | float | [0,1] | zero-positive users; nDCG structurally 0 |
| `gauc_user_coverage` | float | [0,1] | fraction with `0 < positives < impressions` |
| `score_unique_fraction` | float | (0,1] | `unique_scores / rows`; tie-collapse detector |
| `gain_concentration_top10pct` | float | [0,1] | share of total Δ from the top decile of users by \|δ\| |

`primary_score` must reproduce the frozen aggregation `mean(gauc, ndcg_at_5)` within tolerance (§7.3).

### 2.3 `trust` object

```json
{
  "verdict": "accepted",
  "stability": "confirmed",
  "integrity": "clean",
  "flags": [],
  "eta_applied": 0.0016,
  "seed_mean": 0.6104,
  "seed_stderr": 0.00035,
  "seed_count": 3
}
```

| Field | Values |
|---|---|
| `verdict` | `accepted` · `inconclusive` · `negative` · `no_op` · `suspicious` · **`redundant`** (extension) |
| `stability` | `single_seed` · `confirmed` · `unstable` · `not_applicable` |
| `integrity` | `clean` · `compromised` · `inconclusive` |
| `flags` | controlled vocabulary, §10.4 |
| `eta_applied` | the Ladder threshold used, `max(2·stderr, 0.0016)` |
| `seed_mean` / `seed_stderr` / `seed_count` | folded across sibling events at the same fidelity and population |

**`verdict = accepted` means the measurement is trustworthy, not that it improved.** Per §9.5, `parent_eligible` requires `accepted` + `clean`; `best_eligible` *additionally* applies the contract comparison rule. The Ladder therefore lives in `best_eligible`, not in the verdict.

### 2.4 Verdict decision function

First match wins.

```python
def compute_verdict(ev, parent, accepted_nodes, contract) -> Trust:
    if ev.prediction_change.spearman_vs_parent > 0.99:
        return Trust("no_op", "clean", ["spearman_ge_0.99"])

    if ev.contract_violations:
        return Trust("suspicious", "compromised", ev.contract_violations)

    if ev.cross_population.unbiased_audit_delta is not None \
       and ev.parent_delta > 0 and ev.cross_population.unbiased_audit_delta <= 0:
        return Trust("suspicious", "inconclusive", ["unbiased_disagrees"])

    if ev.cross_population.internal_proxy_delta is not None \
       and ev.parent_delta > 0 and ev.cross_population.internal_proxy_delta < 0:
        return Trust("suspicious", "inconclusive", ["proxy_sign_disagrees"])

    if ev.parent_delta > 0.05:
        return Trust("suspicious", "inconclusive", ["delta_gt_0.05_check_leak"])

    if ev.metrics["score_unique_fraction"] < 0.01:
        return Trust("suspicious", "compromised", ["score_ties_excessive"])

    corr, ref = max_delta_corr(ev, accepted_nodes)
    if corr > 0.7:
        return Trust("redundant", "clean", [f"delta_corr_{corr:.2f}_vs_{ref}"])

    mu, se, n = fold_seeds(ev)
    eta = max(2 * se, 0.0016)

    if n < 3 and ev.fidelity == "full" and ev.population == "public_validation":
        return Trust("accepted", "clean", [], stability="single_seed")   # not yet decidable

    if se > 3 * 0.0008:
        return Trust("inconclusive", "clean", ["seed_variance_high"], stability="unstable")

    if abs(mu - parent.primary) <= eta:
        return Trust("inconclusive", "clean", ["within_noise"])

    if mu < parent.primary - eta:
        return Trust("negative", "clean", [])

    flags = []
    if ev.metrics["gain_concentration_top10pct"] > 0.70:
        flags.append("fragile_concentrated_gain")
    if sign(ev.d_gauc) != sign(ev.d_ndcg):
        flags.append("lopsided_half_credit")
    if abs(ev.metrics["drift_primary_slope"]) > 0.002:
        flags.append("drift_detected")
    return Trust("accepted", "clean", flags, stability="confirmed")
```

### 2.5 Emission rules

| Rule | Source |
|---|---|
| No `evaluation.completed` unless `output.checked.accepted = true` | invariant 10 |
| `population = hidden_final` legal only after verified `run.stopped` | §9.5 |
| `public_query_index` increments **only** for `population = public_validation`, `population_variant = val_a` | §2.1 extension |
| Proxy and smoke never update trusted best or convergence | invariant 15 |
| `evaluator_sha256` and `contract_sha256` verified before every call | invariant 6 |
| Every score linked to prediction, evaluator, contract, commit, seed, data manifest | invariant 9 |

---

## 3. `experiment.decided`

Emitted once all seeds for the deciding fidelity are in.

```json
{
  "evaluation_event_id": "evt_000035",
  "decision": "accept",
  "reason_code": "TRUSTED_IMPROVEMENT",
  "fidelity_completed": "full",
  "parent_eligible": true,
  "best_eligible": true,
  "next_fidelity": null,
  "seed_evidence_event_ids": ["evt_000033", "evt_000034", "evt_000035"],
  "comparison": {
    "rule": "ladder_v1",
    "trusted_best_before": 0.6404,
    "candidate_seed_mean": 0.6571,
    "eta_applied": 0.0016,
    "supersedes": true
  },
  "supporting_event_ids": ["evt_000025", "evt_000032", "evt_000035"]
}
```

### 3.1 Complete reason-code table

| `reason_code` | `decision` | `trust.verdict` | `parent_eligible` | `best_eligible` |
|---|---|---|---|---|
| `TRUSTED_IMPROVEMENT` | accept | accepted | true | true |
| `TRUSTED_NO_IMPROVEMENT` | reject | accepted | true | false |
| `WITHIN_NOISE` | reject | inconclusive | false | false |
| `SEED_VARIANCE_HIGH` | reject | inconclusive | false | false |
| `NO_PREDICTION_CHANGE` | reject | no_op | false | false |
| `SIGNAL_ALREADY_CAPTURED` | reject | redundant | false | false |
| `INTEGRITY_UNVERIFIED` | reject | suspicious | false | false |
| `CLEAR_REGRESSION` | reject | negative | false | false |
| `PROXY_PASSED` | promote | accepted | false | false |
| `PROXY_FAILED` | prune | negative | false | false |
| `SMOKE_PASSED` | promote | not_applicable | false | false |
| `OUTPUT_INVALID` | invalid | not_applicable | false | false |

`promote` requires non-null `next_fidelity`. `accept`, `reject`, `prune`, `invalid` are terminal (§9.5).

### 3.2 The comparison rule

`comparison.rule = "ladder_v1"` refers to a block frozen in `COMPETITION.md`:

```md
## Comparison rule (ladder_v1)
A candidate primary score supersedes the trusted best only when

    mu > trusted_best + eta,   eta = max(2 * stderr(seeds), 0.0016)

mu and stderr are computed over all completed full public-validation seeds for
that experiment. 0.0016 is twice the official baseline's 5-seed standard
deviation of 0.0008. Rejected candidates leave best_primary_score unchanged.

Reference: Blum & Hardt, "The Ladder: A Reliable Leaderboard for Machine
Learning Competitions", ICML 2015 (arXiv:1502.04585).
```

Putting it in the contract makes it human-frozen, hash-protected and auditable, and requires no new machinery — §9.5 already routes `best_eligible` through "the contract's comparison rule."

---

## 4. `best.updated`

Emitted only when `best_eligible = true`.

```json
{
  "previous_best_experiment_id": "exp_0034",
  "previous_best_commit_sha": "<commit>",
  "previous_best_primary_score": 0.6404,
  "new_best_experiment_id": "exp_0038",
  "new_best_commit_sha": "<commit>",
  "new_best_primary_score": 0.6571,
  "evaluation_event_id": "evt_000035",
  "improvement": 0.0167,
  "eta_applied": 0.0016,
  "headroom_captured_pct": 23.15
}
```

`headroom_captured_pct = (new_best − 0.5946) / 0.2699 × 100`, where 0.5946 is the official baseline and 0.8645 the oracle ceiling on hidden test.

The Git ref `best/<run_id>` is the derived pointer; the event is canonical (§9.5).

---

## 5. `lesson.recorded`

```json
{
  "lesson_id": "lesson_0004",
  "category": "research_result",
  "status": "active",
  "tags": ["objective", "listwise", "within_user"],
  "summary": "Grouped-softmax over each user's impression list raised GAUC 1.8x more than nDCG@5 against the pointwise parent.",
  "applicability": "Use when training groups are constructed per user and contain both classes.",
  "avoid_when": "Do not construct groups across users, or from users with a single class.",
  "confidence": 0.9,
  "measured_under_frame_experiment_id": "exp_0031",
  "source_event_ids": ["evt_000035", "evt_000036"],
  "source_commit_shas": ["<commit>"]
}
```

| Field | Constraint |
|---|---|
| `category` | `research_result` · `resource_constraint` · `implementation_constraint` · `integrity_warning` · `process_rule` |
| `confidence` | [0,1] |
| `measured_under_frame_experiment_id` | **extension.** The frame node active when measured. Drives the staleness sweep |
| dedupe key | `sha256(normalize(category + sorted(tags) + applicability + avoid_when))` (§14) |

### 5.1 Emission rules

A lesson is permitted only when (§9.6): the source is a verified positive or negative result; recovery is exhausted and exposes a reusable constraint; a suspicious result exposes a reusable integrity rule; or a prior lesson needs superseding.

**Person 5 additions:**

| Verdict | Lesson? |
|---|---|
| `accepted` (superseded or not) | yes — `research_result` |
| `negative` | yes — `research_result` |
| `redundant` | yes — `research_result`, tag `saturated` |
| `suspicious` | yes — `integrity_warning` |
| `inconclusive` | **no** — nothing was learned |
| **`no_op`** | **never** — the implementation failed, the hypothesis is untouched |

The `no_op` rule is the important one. Writing a lesson there records a false belief about a direction that was never actually tested.

---

## 6. `lesson.status_changed`

```json
{
  "lesson_id": "lesson_0012",
  "new_status": "stale",
  "reason": "Measured under objective frame exp_0022, superseded by accepted frame move exp_0038.",
  "source_event_ids": ["evt_000035"]
}
```

`new_status` ∈ `active` · `stale` · `superseded` · `retracted`. Old events are never edited; retrieval uses the latest status event (§9.6).

### 6.1 Frame staleness sweep

```python
def sweep_staleness(accepted_decision, fold, contract):
    fam = fold.experiment(accepted_decision.experiment_id).family
    if contract.family_kind(fam) != "frame":       # frame = {objective, grouping}
        return []
    out = []
    for lesson in fold.active_lessons(category="research_result"):
        if contract.family_kind(lesson.tags) != "content":
            continue
        if lesson.measured_under_frame_experiment_id != accepted_decision.experiment_id:
            out.append(emit_status_changed(lesson.lesson_id, "stale",
                f"Measured under objective frame {lesson.measured_under_frame_experiment_id}, "
                f"superseded by accepted frame move {accepted_decision.experiment_id}."))
    return out
```

A frame move changes what "better" means, so every prior content result was measured against a superseded objective. It is **stale, not falsified** — without this the agent permanently writes off directions that were only ever tested under the wrong loss function.

---

## 7. Artifacts

| kind | path | content | when |
|---|---|---|---|
| `metrics` | `artifacts/<run>/<exp>/metrics_<pop>_<fid>_s<seed>.json` | full slice report, per-day breakdown, per-bucket supports | every evaluation |
| `delta_vector` | `artifacts/<run>/<exp>/delta_<pop>.npy` | float32[n_users], per-user Δ vs parent | full public validation only |
| `submission` | `artifacts/<run>/final/submission.csv` | `row_id,user_id,video_id,score`, 170,588 rows | finalization |
| `report` | `artifacts/<run>/reports/fig_<n>.png` | the six charts (§9.3) | finalization |

All carry `ArtifactRef` with `sha256` and `size_bytes` computed from stored bytes (§7.1). `delta_vector` requires the kind-enum extension.

Delta vector definition:

```
δ_e[u] = primary_contribution_e(u) − primary_contribution_parent(u)
```

22,377 float32 ≈ 90 KB per accepted node. Used for redundancy detection, orthogonality, combination-gain prediction, ensemble selection, and frontier tracking.

---

## 8. `EvaluationProjection` — the derived output Person 1 reads

Pure function of the folded ledger. Never persisted (§20). Recomputed on restart.

```python
from rankforge.evaluation.projection import evaluation_projection
proj = evaluation_projection(fold, experiment_id)
```

```python
@dataclass(frozen=True)
class EvaluationProjection:
    # ---- identity ----
    experiment_id: str
    parent_experiment_id: str | None
    family: str
    kind: Literal["frame", "content", "capacity"]
    fidelity_completed: Literal["smoke", "proxy", "full"]

    # ---- REWARD channel → UCB arithmetic ----
    trusted_best_primary: float          # GATED; unchanged when this node did not supersede
    trusted_best_experiment_id: str
    node_moved_best: bool
    headroom_captured_pct: float

    # ---- VERDICT channel → branch / prune logic ----
    trust_verdict: Literal["accepted","inconclusive","negative","no_op","suspicious","redundant"]
    trust_integrity: Literal["clean","compromised","inconclusive"]
    trust_stability: Literal["single_seed","confirmed","unstable","not_applicable"]
    decision: Literal["promote","accept","reject","prune","invalid"]
    reason_code: str
    parent_eligible: bool
    best_eligible: bool
    next_fidelity: Literal["proxy","full"] | None

    # ---- NARRATIVE channel → Manager prompt ----
    delta_band: Literal["much_worse","worse","within_noise","better"]
    metric_split: Literal["both_up","gauc_only","ndcg_only","both_down","mixed"]
    slice_digest: str                    # <=120 chars

    # ---- SELECTION-SUPPORT channel → exploration / diversity ----
    orthogonality: float                 # 1 - max corr(delta_this, delta_accepted); 1.0 if none
    holdout_signal: Literal["normal","widening","alert"]
    convergence_pressure: int
    scheduling_hint: Literal["free","force_high_variance"]

    # ---- COST ----
    cost: ResourceDelta
    public_query_index: int
    full_evaluations_remaining: int

    # ---- MEMORY MAINTENANCE ----
    stale_lesson_ids: list[str]
```

### 8.1 Derivation rules

| Field | Derivation |
|---|---|
| `trusted_best_primary` | latest `best.updated.new_best_primary_score`, or baseline if none |
| `node_moved_best` | a `best.updated` event exists with `new_best_experiment_id == experiment_id` |
| `delta_band` | `better` if Δ > η; `within_noise` if \|Δ\| ≤ η; `worse` if −0.01 < Δ < −η; `much_worse` if Δ ≤ −0.01 |
| `metric_split` | signs of `d_gauc` and `d_ndcg` |
| `orthogonality` | `1 − max` over accepted δ vectors of Pearson correlation |
| `holdout_signal` | `normal` if val_A−val_B gap < 0.006; `widening` if < 0.012; else `alert` |
| `convergence_pressure` | `consecutive_non_improving_full_evaluations` (§10) |
| `scheduling_hint` | `force_high_variance` when `convergence_pressure == 2` |
| `slice_digest` | template-rendered from the two largest-magnitude slice deltas |

### 8.2 Timing

Person 1 must not read a projection before `experiment.decided` exists for that experiment. A partially-seeded evaluation has an undefined verdict.

---

## 9. Derived Markdown

### 9.1 `LESSONS.md`

Generated from active `lesson.recorded` after applying all `lesson.status_changed` (§14).

```md
- [lesson_0004][objective][active][confidence=0.90] Grouped-softmax over each
  user's impression list raised GAUC 1.8x more than nDCG@5. Applies: per-user
  groups with both classes. Avoid: cross-user groups. Evidence: evt_000035,
  evt_000036; commit: 4ac19e2.
```

### 9.2 `SUMMARY.md`

Judge-facing final projection, generated at finalization.

```md
# Run Summary — run_20260829_a
## Result
primary 0.6371 on hidden test (baseline 0.5946, oracle ceiling 0.8645)
delta   +0.0425   headroom captured 45.5% -> 15.7% of remaining
gauc 0.6981 (+0.0371) | ndcg_at_5 0.5761 (+0.0479)
## Autonomy
manual interventions: 1 (watchdog restart at 03:41, logged evt_000212)
## Resource
LLM tokens (provider): 1,204,331 in / 218,904 out
agent wall-clock: 4h 51m | iterations used: 38 of 50 | GPU-hours: 0.0
## Verdict census
accepted 7 | negative 9 | no_op 6 | redundant 4 | suspicious 4 | inconclusive 8
full runs avoided by no_op + redundant detection: 10
## Convergence
converged at iteration 38 under epsilon=0.002, N=3
## Limitations
[...]
```

### 9.3 Charts

| # | Chart |
|---|---|
| 1 | val_A vs val_B divergence across full evaluations |
| 2 | reported (`trusted_best`) vs raw `primary_score` scatter |
| 3 | predicted vs measured combination gain |
| 4 | verdict census with cost avoided |
| 5 | headroom progress per metric |
| 6 | resource by role, provider and estimated tokens separated |

---

## 10. Complete enum reference

### 10.1 `population`
`internal_proxy` · `public_validation` · **`unbiased_audit`** (extension) · `hidden_final`

### 10.2 `population_variant` (extension)
`val_a` · `val_b` · `null`

### 10.3 `family` and derived `kind`

| family | kind |
|---|---|
| `objective` | frame |
| `grouping` | frame |
| `feature` | content |
| `sequence` | content |
| `multitask` | content |
| `censored_regression` | content |
| `temporal` | content |
| `model_class` | capacity |

Mapping lives in `COMPETITION.md`. **No new event field.**

### 10.4 `trust.flags` controlled vocabulary

```
spearman_ge_0.99
within_noise
seed_variance_high
unbiased_disagrees
proxy_sign_disagrees
delta_gt_0.05_check_leak
delta_corr_<0.00>_vs_<exp_id>
fragile_concentrated_gain
lopsided_half_credit
score_ties_excessive
drift_detected
forbidden_column_<name>
```

---

## 11. Emission sequence for one experiment

```
                                          [Person 4]  execution.finished
                                          [Person 3]  output.checked         <- Gate B
--- Person 5 begins ---
  evaluation.completed  proxy   / internal_proxy / seed 0
  experiment.decided    promote / PROXY_PASSED / next_fidelity=full
                                          [Person 4]  execution.finished x3
                                          [Person 3]  output.checked x3
  evaluation.completed  full / public_validation val_a / seed 0
  evaluation.completed  full / public_validation val_a / seed 1
  evaluation.completed  full / public_validation val_a / seed 2
  evaluation.completed  full / unbiased_audit / seed 0        <- free, no query index
  experiment.decided    accept / TRUSTED_IMPROVEMENT
  best.updated                                                <- only if best_eligible
  lesson.recorded                                             <- only if reusable
  lesson.status_changed x N                                   <- only if frame move
```

`internal_proxy` and `unbiased_audit` events never increment `public_query_index`, never affect convergence, never update trusted best.

---

## 12. Invariants Person 5 enforces

| # | Invariant |
|---|---|
| 1 | Zero LLM calls anywhere in the evaluation path |
| 2 | `evaluate.py` is called from a pinned copy outside every editable root; never wrapped, cached, or reimplemented as a "faster version" |
| 3 | `evaluator_sha256` and `contract_sha256` verified before every evaluation (invariant 6) |
| 4 | No evaluation before `output.checked.accepted = true` (invariant 10) |
| 5 | `metric_set` records raw truth; gating happens only in `best_eligible` |
| 6 | Proxy and smoke never update trusted best or convergence (invariant 15) |
| 7 | `hidden_final` never enters a planner, coder or recovery context (invariant 8) |
| 8 | A verified low score is not a runtime failure (invariant 13) |
| 9 | Resource totals are sums of event deltas, never mutable counters (invariant 16) |
| 10 | Provider-reported and estimated tokens are never silently combined (§7.2) |
| 11 | `no_op` never emits a lesson |
| 12 | `inconclusive` never reaches a planner context |

---

## 13. What Person 5 never emits

| Event | Owner |
|---|---|
| `run.started`, `run.stopped`, `contract.verified` | orchestrator |
| `context.created` | context builder |
| `experiment.proposed` | Person 1 |
| `patch.created` | Person 2 |
| `patch.checked` | Person 3 |
| `execution.started`, `execution.finished`, `recovery.decided` | Person 4 |
| `output.checked` | Person 3 |
| `manual.intervention` | orchestrator, on any human action |

Person 5 **reads** `output.checked` and `execution.finished`; it does not produce them.

---

## 14. Schema extensions required

| # | Extension | Where | Blocks |
|---|---|---|---|
| 1 | `unbiased_audit` in the population enum | §9.5 | bias-exploitation detector |
| 2 | `population_variant` field | §9.5 | val_A / val_B separation, query accounting |
| 3 | `redundant` in `trust.verdict` | §9.5 | δ-fingerprint redundancy |
| 4 | `delta_vector` in the `ArtifactRef` kind enum | §7.1 | orthogonality, redundancy |
| 5 | diagnostic metrics declared | `COMPETITION.md` | §7.3 forbids undeclared extras |
| 6 | `family → kind` map | `COMPETITION.md` | staleness sweep; no new field |
| 7 | `comparison rule (ladder_v1)` block | `COMPETITION.md` | `best_eligible` semantics |
| 8 | `measured_under_frame_experiment_id` on `lesson.recorded` | §9.6 | staleness sweep |
| 9 | `cross_population` object on `evaluation.completed` | §9.5 | proxy / unbiased sign checks |
| 10 | `orthogonality` in the planner context | §13.1 | diversity term cannot reach Person 1 |
| 11 | "inconclusive" added to the §13.1 exclusion list | §13.1 | gate leaks "almost worked" |

All eleven are cheap before the fold, replay and UCB code exist, and expensive after.
