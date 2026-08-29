# Person 5 — Evaluation, Trust, Diagnosis and Evidence

**TikTok TechJam 2026 · Problem Statement 2 · KuaiRand-Pure**
Implementation plan. Owner: Person 5.

---

## 0. Job statement

> Decide whether an experiment actually worked, explain why, remember it, and tell the planner what to do next — without ever letting noise into the loop.

Four questions answered after every experiment:

1. Did it really improve?
2. Why did it improve or fail?
3. What should the agent remember?
4. What feedback goes to the planner?

Plus one answered once, at the end: **when do we submit, and which checkpoint?**

---

## 1. Ground truth: the numbers this module is built around

From `baseline_scores.json` and my own measurements on the raw logs. Every threshold below derives from these.

### 1.1 Reference scores

| Model | valid primary | test primary |
|---|---|---|
| random | 0.4834 | 0.4753 |
| item popularity | 0.5807 | 0.5715 |
| **FM (official baseline)** | **0.6016** | **0.5946** |
| oracle (true labels as scores) | 0.8484 | 0.8645 |

FM 5-seed std on test = **0.0008** for GAUC, nDCG@5 and primary.

### 1.2 Headroom, per metric

| | baseline (test) | ceiling | headroom | value of +0.01 |
|---|---|---|---|---|
| GAUC | 0.6610 | 1.0000 | 0.3390 | 2.95% of headroom |
| nDCG@5 | 0.5282 | 0.7289 | **0.2007** | **4.98% of headroom** |
| primary | 0.5946 | 0.8645 | 0.2699 | 3.71% |

**A point of nDCG is worth 1.7× a point of GAUC in headroom terms.** Always report normalised deltas alongside raw ones.

### 1.3 Evaluation geometry (measured from the logs)

| | train | valid | test |
|---|---|---|---|
| rows | 1,141,112 | 124,909 | 170,588 |
| users | 26,210 | 22,377 | 23,875 |
| **impressions/user (median)** | **31** | **4** | **5** |
| zero-positive users | 5.1% | 30.3% | 27.1% |
| all-positive users | 2.3% | 11.9% | 9.2% |

- GAUC covers only **57.8% of valid users** (88.4% of positives).
- **63.7%** of valid users have ≤5 impressions → nDCG@5 is effectively full-list nDCG.
- **32.4%** have ≤2 impressions → nDCG is nearly binary for them.
- Train lists are ~8× longer than eval lists. **This is a frame problem, not a detail.**

### 1.4 Label mechanism (measured, not in the README)

```
τ(d) = 6000                       if duration_ms <  7000
     = min(0.97 · duration_ms, 18000)  otherwise

long_view = 1  ⟺  play_time_ms ≥ τ(duration_ms)
```

Accuracy: **99.47%** train, **99.52%** valid/test, **99.69%** random log — fitted on train only. Long-video branch alone: 99.88%. **75.4% of rows sit on the flat 18-second threshold.**

`duration_ms` is visible at eval time, so τ is known per row. Use it as a slice dimension.

### 1.5 Auxiliary signal strength (measured on train)

| signal | rate | φ vs `long_view` |
|---|---|---|
| `is_click` | 46.3% | **0.761** |
| `log(play_time_ms)` | — | **0.596** |
| `is_profile_enter` | 2.5% | 0.146 |
| `is_like` | 1.9% | 0.099 |
| `is_comment` | 0.3% | 0.059 |
| `is_follow` / `is_forward` | 0.10% | ~0.024 |
| `is_hate` | 0.04% | −0.004 |

Only the top two are worth auxiliary heads. Record this in the method cards so the agent doesn't waste iterations on the sparse ones.

### 1.6 Closed hypotheses (do not re-test)

| Hypothesis | Evidence | Source |
|---|---|---|
| Coarse categorical features in FM | 0.5940 vs 0.5950 | organizer ablation |
| Embedding capacity k=8/16/32 | 0.5895 / 0.5902 / 0.5887 | organizer ablation |
| Logged row order leaks the original ranking | GAUC 0.4992 / 0.5008 (both random) | my measurement |
| Time order within user | GAUC 0.516 — weak but real | my measurement |
| `video_features_statistic` leaks test period | stat_rate tracks train (r=0.878) better than test (r=0.818); 31× the scale of these logs → platform-wide side info | my measurement |

**Structural rule to pin everywhere:** any term constant within a user contributes *exactly zero*, because ranking happens inside each user's list. User-side features only act through crosses with item-side features. This is why the organizer feature ablation failed.

### 1.7 Budget

- **50 iterations** hard cap per run
- **6 h** wall-clock ceiling
- **Convergence: ε = 0.002 over N = 3 consecutive iterations** → run terminates

Three low-yield experiments in a row ends the run. Experiment ordering is a first-class design variable.

---

## 2. Architecture: six layers

```
RunResult (predictions + resource usage)
   │
   ├─ L1  MEASURE     pristine evaluate.py · alignment · 4 populations
   ├─ L2  TRUST       3 seeds · Ladder gate · holdout sign · suspicion
   ├─ L3  DIAGNOSE    slice attribution · delta fingerprint · verdict
   ├─ L4  REMEMBER    reflection · method card · ledger row
   ├─ L5  FEED BACK   split into two different objects
   └─ L6  DECIDE      submission gate (runs once, at convergence)
                │
       ┌────────┴────────┐
  EvaluationReport   SearchFeedback
  (full truth →      (gated →
   ledger, judges)    planner only)
```

**Nothing in this module calls an LLM.** Reasoning: a March 2026 measurement (RewardHackingAgents, arXiv:2603.11337) found natural agents attempt evaluator tampering in ~50% of episodes without being instructed to, and that locking evaluation integrity drives observed compromise to zero. METR measured o3 reward-hacking in 30.4% of RE-Bench runs unprompted. An LLM asked "is +0.0003 a real improvement?" will say "the direction looks right." Adjudication is arithmetic.

---

## 3. L1 — Measure

### 3.1 Four populations

| Population | Source | Cost | Used for |
|---|---|---|---|
| `internal_holdout` | train dates 20220415–21, subsampled to ~5 impressions/user | free | every `smoke` and `proxy` run |
| `val_A` | 80% of valid users, hashed | 1 query | all iterative selection |
| `val_B` | 20% of valid users, hashed | rare | audit + final pick |
| `random_log` | the **20220422–28 slice only** of `log_random` | free | unbiased bias check |

Route `smoke` and `proxy` fidelity to `internal_holdout` **only**. If 50 experiments run but 20 reach full fidelity, adaptive queries against official validation drop 60%. This is the cheapest possible protection against holdout overfitting (Dwork et al., *The reusable holdout*, Science 2015 — query count is the quantity that destroys validity).

`random_log` spans 4/22–5/08; the later half overlaps the test window. **Restrict to 4/22–4/28** so the temporal boundary stays clean. README item 7 explicitly sanctions this file as an extra unbiased validation set.

Subsample `internal_holdout` to ~5 impressions/user so its geometry matches eval (median 4–5), not train (median 31).

### 3.2 Split code

```python
def split_val(user_ids, ratio=0.8, salt="ladder-2026"):
    """Split by USER, not by row. GAUC/nDCG aggregate per user;
    splitting rows would contaminate both estimates.
    Hash, not shuffle, so the split survives process restarts."""
    uniq = np.unique(user_ids)
    h = np.array([int(hashlib.md5(f"{salt}{u}".encode()).hexdigest()[:8], 16) for u in uniq])
    a_users = set(uniq[h % 100 < ratio * 100])
    return np.isin(user_ids, list(a_users))
```

### 3.3 Evaluator adapter contract

```python
# benchmarks/kuairand_pure/evaluator_adapter.py
PRISTINE = "/opt/pinned/evaluate.py"   # outside the agent's workspace
EVAL_SHA = "…"                          # verified before every call

def score(population, user_ids, labels, scores) -> dict:
    assert sha256(PRISTINE) == EVAL_SHA
    validate_alignment(scores)          # length, finite, row_id contiguity
    return pristine_evaluate(user_ids, labels, scores)   # never wrapped, never cached
```

Rules:
- `evaluate.py` is **called, never modified, never wrapped, never reimplemented as a "faster version."**
- Slice attribution (L3) computes a *second* per-user decomposition outside `evaluate.py`, not inside it.
- Log `evaluator_sha256` on every `EvaluationPayload`.

### 3.4 P0 gate — blocks the whole project

Reproduce all six reference numbers:

```
valid:  random 0.4834 | pop 0.5807 | fm 0.6016
test:   random 0.4753 | pop 0.5715 | fm 0.5946
```

Also independently reimplement GAUC and nDCG@5 from the spec and assert agreement with `evaluate.py` to 1e-9. **If any of the six is off, stop and fix.** Every downstream number is measured on that ruler.

---

## 4. L2 — Trust: is the effect real?

### 4.1 The Ladder gate

Blum & Hardt, *The Ladder: A Reliable Leaderboard for Machine Learning Competitions*, ICML 2015 (arXiv:1502.04585). The mechanism updates the reported best **only** when a submission improves significantly; otherwise it reports the previous value. It has leaderboard-accuracy guarantees under fully adaptive querying and a parameter-free variant that sets the step from a significance test.

An overnight agent is the worst case of that setting: hundreds of adaptive queries with none of the friction that slows a human. **No existing MLE-agent scaffold (AIDE, AI-Scientist-v2, MLE-bench baselines, autoresearch) applies adaptive-data-analysis protection to its own search loop.** They feed the raw number straight back.

A 2026 empirical study of release mechanisms against an adaptive scalar-feedback attacker measured: naive reusable holdout → overfit gap 3.24, attacker wins 20/20; **Ladder → gap 0.00, attacker wins 0/20.**

```python
class Ladder:
    FLOOR = 0.0016   # organizers' 2σ (5-seed std = 0.0008)

    def __init__(self, n_val_a):
        self.best_true     = -math.inf   # internal, harness only
        self.best_reported = -math.inf   # what the planner sees
        self.prec = 1.0 / math.sqrt(n_val_a)
        self.queries = 0

    def submit(self, seed_scores):
        self.queries += 1
        mu  = statistics.mean(seed_scores)
        se  = statistics.stdev(seed_scores) / math.sqrt(len(seed_scores))
        eta = max(2 * se, self.FLOOR)
        meta = {"mu": mu, "se": se, "eta": eta,
                "delta": mu - self.best_true, "queries": self.queries}
        if mu > self.best_true + eta:
            self.best_true = mu
            self.best_reported = round(mu / self.prec) * self.prec
            return self.best_reported, "ACCEPT", meta
        return self.best_reported, "REJECT", meta   # ← returns the OLD value
```

Three design notes:

- **`best_true` and `best_reported` must be separate.** Compare on the true value; report the quantised one. Comparing on quantised values accumulates error.
- **Quantisation to `1/√n`** (≈0.0032 for ~100k val_A rows) further limits how much information the agent can extract per query. This is from the original algorithm.
- **`meta` is logged in full but never returned to the planner.** You need `delta` and `eta` for the charts; the agent gets one number.

**Why `max(2·se, FLOOR)`:** the measured term handles models with genuinely higher variance than FM (LambdaRank may be 3× noisier). The floor handles three samples estimating a standard deviation badly — a lucky-tight triple gives se ≈ 0.0001 and collapses the gate.

### 4.2 Mandatory synthetic test (write before any real data)

```python
ladder = Ladder(n_val_a=100_000)
ladder.submit([0.6000, 0.6008, 0.5996])              # ACCEPT (first)
for _ in range(50):
    s = [0.6000 + random.gauss(0, 0.0008) for _ in range(3)]
    assert ladder.submit(s)[1] == "REJECT"           # 50 pure-noise draws, zero accepts
assert ladder.submit([0.6100, 0.6105, 0.6098])[1] == "ACCEPT"
before = ladder.best_reported
ladder.submit([0.6101, 0.6099, 0.6103])              # tiny gain
assert ladder.best_reported == before                # rejects leave the value untouched
```

This test is also README evidence.

### 4.3 Seed independence check

Before trusting `se`, confirm all randomness actually varies with the seed: data shuffling, feature subsampling, model init, bagging. Run the FM baseline at 3 seeds — std should land near 0.0008. **If you get 0.00001, a seed isn't wired through**, `eta` collapses to the floor, and the gate silently weakens.

### 4.4 Suspicion detectors

Three-way label, never a boolean: `clean` / `compromised` / `inconclusive`. Over-attributing drift to compromise makes the tooling untrustworthy.

| Detector | Rule | Catches |
|---|---|---|
| Sign disagreement | val_A up, `internal_holdout` down | adaptive overfitting |
| **Bias exploitation** | val_A up, `random_log` flat or down | fitting the logging policy, not preference |
| Too-good | Δprimary > 0.05 in one step | leakage — hunt it |
| Metric split | ΔGAUC and ΔnDCG opposite signs | half credit only; flag |
| Tie collapse | unique scores < 1% of rows | degenerate; order falls to file order |
| Slice concentration | >70% of gain from <10% of users | fragile, won't survive combination |
| Forbidden columns | `play_time_ms`, `profile_stay_time`, `comment_stay_time`, any `is_*` used as input | target leakage (shared with Person 3) |

The bias-exploitation detector is the one only this dataset makes possible. The Ladder controls **variance**; `random_log` controls **bias**. They are orthogonal and neither substitutes for the other.

---

## 5. L3 — Diagnose

### 5.1 Slice attribution

GAUC and nDCG are means over users, so the user-level decomposition is **exact and free**. Item- and duration-level slices are *not* exact decompositions — use the normalised within-group rank of each positive as the diagnostic quantity there, and say so in the README.

```
primary 0.6104   (Δ vs parent +0.0071, Ladder: ACCEPT, verdict CONFIRMED)
  GAUC   0.6702  (+0.0092  →  2.7% of GAUC headroom)
  nDCG@5 0.5506  (+0.0050  →  2.5% of nDCG headroom)

  by impressions:   2 (32.4%) −0.004 | 3-5 +0.006 | 6-12 +0.012 | >12 +0.021
  by τ band:        short<7s −0.006 | mid +0.008 | flat-18s (75.4%) +0.011
  by item pop:      cold −0.007 | warm +0.009 | hot +0.014
  by positive cnt:  1 pos +0.004 | 2 pos +0.009 | 3+ pos +0.015
  by val day:       d1..d7, no drift detected
  random_log:       +0.0058  ← agrees, not bias exploitation
  internal_holdout: +0.0063  ← sign agrees
  unreachable:      30.3% zero-positive users, nDCG structurally 0
```

Slice rationale:
- **impressions** — the 2-impression group is 32.4% of users and behaves differently
- **τ band** — the label's own mechanism; surfaces duration bias from the data rather than from having read CWM
- **positive count** — GAUC weights by positive count, so gains there are worth more
- **val day** — temporal drift check
- **item popularity** — cold-start regression detection

### 5.2 Delta-vector fingerprints

For each experiment store the per-user delta vector

```
δ_e[u] = primary_contribution_e(u) − primary_contribution_parent(u)     δ_e ∈ R^22377
```

Five uses, all free:

1. **Redundancy.** `corr(δ_new, δ_accepted) > 0.7` → same signal in different clothing. Likely to fire between multi-task-with-`is_click` and watch-time regression, since `is_click` has φ=0.76 with the label and watch time determines it.
2. **Combination-gain prediction**, before spending an iteration:
   `predicted ≈ Δ₁ + Δ₂ · (1 − ρ(δ₁, δ₂))`
   The quantitative form of the 1+1<2 effect. Track predicted vs measured across the run → a chart no other team will have.
3. **Ensemble selection.** Rank-averaging helps in proportion to *low* δ correlation. Pick members analytically instead of by search.
4. **Frontier tracking.** The union of users improved by any accepted node; the complement is the un-improved frontier, and it can be characterised ("still losing on 2-impression lists and cold items"). A far stronger directive than a scalar.
5. **Orthogonality-seeking exploration.** Add a diversity term to the planner's UCB: prefer families whose historical δ vectors least correlate with the accepted set.

This transplants AlphaEvolve's diversity principle (island models over *program* space) into *per-example improvement* space, which is where it matters for a fixed metric.

### 5.3 The verdict classifier

**This is the highest-value component in the module.** AIDE parses a scalar from stdout; AI-Scientist-v2 inherits AIDE's tree and adds `debug_prob`; AlphaEvolve cascades but still selects on score. All of them collapse Δ≈0 into "try something else."

Six causes look identical from the score alone:

| Verdict | Detector | Next action |
|---|---|---|
| **NO_OP** | `spearman(new_scores, parent_scores) > 0.99` | The diff didn't change predictions. **Re-implement, don't re-think.** Do NOT falsify the hypothesis |
| **BROKEN** | R3 tiny-overfit fails, or train loss flat, or unique scores < 1% of rows | Route to Person 4's repair path |
| **BLOCKED** | method-card prerequisites unmet | DEFER; re-queue after the prerequisite lands |
| **REDUNDANT** | `corr(δ_new, δ_accepted) > 0.7` | Signal already captured; don't retry variants |
| **INCONCLUSIVE** | `0 < Δ < η` | Escalate to 5 seeds before deciding |
| **FALSIFIED** | none of the above and Δ ≤ 0 | Write a reflection, quarantine the family |

> **The NO_OP detector alone justifies building this layer.** The most common silent failure in an overnight run is a diff that looks plausible, executes cleanly, and changes nothing — a config flag never read, a feature list overwritten downstream, a loss swapped while the wrong optimiser is still called. The score returns parent ± noise, the agent concludes "grouped softmax doesn't help on this data," writes it into memory, and **permanently poisons the most promising direction on the roadmap.** One Spearman correlation prevents it.

Success splits too:

| Verdict | Condition | Feedback |
|---|---|---|
| **CONFIRMED** | Δ > η, holdout agrees, `random_log` agrees | Exploit: branch from here |
| **SUSPECT** | Δ > η but `random_log` flat | Logging-policy artefact. Do **not** branch |
| **FRAGILE** | Δ > η but >70% of gain from <10% of users | Accept but flag; unlikely to survive combination |
| **LOPSIDED** | ΔGAUC and ΔnDCG opposite signs | Half credit. Next hypothesis targets the lagging metric |

---

## 6. L4 — Remember

### 6.1 Three stores

**`method_cards/*.yaml`** — seeded before the run, updated after. Each carries: problem addressed, prerequisites, estimated cost, expected effect, prior results, and when *not* to try it.

```yaml
# method_cards/static_categorical_features.yaml
name: static categorical feature expansion
status: FALSIFIED_BY_ORGANIZER
evidence: "CWM 13 fields → 0.5940 vs 5 fields 0.5950 (within noise)"
reason: >
  user_id × video_id crosses already absorb the signal, and user-side
  first-order terms are exactly zero under within-user ranking.
retry_only_if: "numeric statistic fields inside a GBDT, not categoricals inside FM"
```

**`reflections.jsonl`** — written only when a *research direction* fails, never for a code crash. Person 4 owns crash recovery; Person 5 owns "this idea didn't work, and here is the generalisable reason."

**`ledger.jsonl`** — one row per iteration. Judge evidence, not logging.

### 6.2 The staleness rule

Experiments come in kinds:

| Kind | Examples | Property |
|---|---|---|
| **frame** | loss function, group construction | Changes what "better" means. **Invalidates prior content measurements** |
| **content** | sequences, time features, multi-task, censored regression | Adds information or supervision within a frame |
| **capacity** | model class | Gated on an information move landing first |

When a frame move is CONFIRMED, every prior `content` result is marked **STALE, not FALSIFIED**:

```python
if accepted.kind == "frame":
    for r in memory.where(kind="content", status="falsified"):
        r.status = "stale"
        r.note = f"measured under frame {accepted.parent_frame}, superseded by {accepted.id}"
```

Without this, the agent permanently writes off directions that were only ever measured under the wrong objective. **Requires an `experiment_kind` field on `ExperimentSpec` — negotiate with Person 1 before they write the UCB.**

---

## 7. L5 — Feed back: two objects, deliberately different

```python
# → ledger, judges, dashboard. FULL TRUTH.
EvaluationReport(
    seed_mean=0.61042, seed_std=0.00061,
    gauc=0.6702, ndcg5=0.5506,
    gauc_headroom_pct=2.71, ndcg_headroom_pct=2.49,
    baseline_delta=+0.0088, parent_delta=+0.0071,
    stability_passed=True, verdict="CONFIRMED",
    integrity_label="clean", evaluator_sha256="…",
    slices={...}, delta_vector_ref="deltas/exp_034.npy",
)

# → Person 1's UCB only. GATED.
SearchFeedback(
    official_score=0.6104,          # gated; unchanged when rejected
    gauc=0.6702, ndcg5=0.5506,      # both, not collapsed
    uncertainty=0.00035,
    verdict="CONFIRMED",
    slice_digest="dense users +, 2-impression lists −",
    holdout_signal="normal",         # three-level, NEVER the number
    orthogonality=0.34,              # 1 − max corr with accepted δ vectors
    convergence_pressure=1,          # consecutive sub-ε count
    cost=ResourceUsage(...),
    recommendation=DecisionAction.PROMOTE,
)
```

Reporting **both metrics separately** rather than a collapsed `primary` follows AlphaEvolve's finding that multi-metric feedback tends to improve the single target metric, likely through diversity.

### What must never reach the planner

- Hidden test data or labels — never enter the loop at all
- Raw `seed_mean` when the Ladder rejected — that is the gate
- `val_B` or `internal_holdout` **numbers** — three-level signal only, or they become a second optimisation target
- Any LLM inside `adjudicator.py`

### 7.1 Convergence-aware scheduling

```python
if consecutive_sub_epsilon == 2:
    recommendation = FORCE_HIGH_VARIANCE   # planner must pick an untried family,
                                           # not a variant of the current node
```

The stopping rule is fixed; choosing experiment order to avoid a premature plateau is legitimate experimental design. Document it.

**Track convergence on the true seed-mean best, not the gated value.** Otherwise the Ladder's own conservatism declares convergence on a run that was still improving. This is the single most important interface note in the project.

---

## 8. The probe pass — don't queue the directions

Before committing iterations, run six deliberately crude probes on `internal_holdout` only. Each ~30 lines, ~1 minute, **zero validation queries**.

| Direction | Crude probe |
|---|---|
| Loss function | LightGBM `lambdarank`, default groups |
| Group construction | same model, groups by user-day vs user |
| Sequences | last-5 `video_id`s as raw categoricals |
| Time | `hourmin` bucket + within-user recency rank |
| Multi-task | `is_click` as second target, fixed weight 0.3 |
| Censored regression | predict `log(play_time)`, threshold at known τ |

Set UCB priors from the results, **with a deliberate asymmetry:**

> A probe that **succeeds** is strong evidence — a bad implementation already found signal → raise the prior substantially.
> A probe that **fails** is weak evidence — the crudeness may be the cause → lower the prior slightly, never eliminate.

This is the **inverse** of the R5 pilot rule ("a cheap rung can reject but never declare victory"). The difference: a pilot is a faithful-but-small version of the *same code*; a probe is an unfaithful version of the same *idea*. State this explicitly in the write-up — getting it backwards is a natural mistake.

Ten minutes buys informed priors over the remaining 44 iterations.

---

## 9. L6 — Submission decision

### 9.1 Stopping

Whichever comes first:
- true validation best not improved by >0.002 over 3 consecutive iterations
- 50 iterations
- 6 h wall-clock

**Do not stop early to look cheap.** Feasibility is 15%, graded in three coarse tiers, and scored *only among submissions that beat the baseline*. Solid dominates frugal.

### 9.2 Selection — where good runs get thrown away

```
❌ argmax over val_A            ← by construction the most overfitted point
✅ among Ladder-ACCEPTED nodes:
     filter verdict == CONFIRMED
     require internal_holdout sign agreement
     require random_log agreement
     rank by val_B
     rank-average scores across 3–5 seeds
```

Rank-average, not score-average: only relative order is read, and seed score scales need not be comparable.

### 9.3 Pre-flight checklist

```
[ ] evaluate.py sha256 unchanged since P0
[ ] submit.py --check passes on the test split
[ ] row_id 0-based, contiguous, strictly increasing
[ ] never joined on (user_id, video_id) — 3.06% duplicates, up to 12×
[ ] no NaN / Inf; unique scores > 1% of rows
[ ] chosen node CONFIRMED, integrity clean, holdout sign agrees
[ ] results table: GAUC, nDCG@5, deltas vs 0.6610 / 0.5282
[ ] resource report: tokens in/out by role, wall-clock, iterations used of 50
[ ] intervention count from interventions.jsonl (auto-appended, not self-reported)
```

### 9.4 Result tiers — "is this good?"

| primary (test) | Δ | verdict |
|---|---|---|
| 0.5946 | 0 | baseline |
| < 0.600 | < +0.005 | **noise, not a result** — below 2.5σ |
| 0.605–0.615 | +0.01–0.02 | solid; the objective change worked |
| 0.615–0.635 | +0.02–0.04 | **strong**; multiple directions landed |
| 0.635–0.645 | +0.04–0.05 | excellent, likely top of field |
| > 0.65 | > +0.055 | **suspicious — hunt for a leak** |
| 0.8645 | +0.270 | oracle ceiling |

Realistic target given features and capacity are measured dead ends: **0.62–0.64**. Treat +0.03 as a strong outcome.

Progress framing for the report: `(score − 0.5946) / 0.2699`. The baseline already captures 30.7% of the attainable range.

---

## 10. Build order

Ordered by risk, not by layer number.

| Phase | Deliverable | Blocked by |
|---|---|---|
| **H0–H2** | `evaluator_adapter.py`; independent metric reimplementation agreeing to 1e-9; all six reference scores reproduced | nothing |
| **H2–H4** | **NO_OP detector** (1 line); `submission_adapter.py` with `--check` wired | nothing |
| **H4–H8** | Ladder gate + 50-noise synthetic test; seed-independence check | nothing — synthetic |
| **H8–H12** | 4-population split; `slices.py`; suspicion detectors | real predictions |
| **H12–H14** | Verdict classifier (six-way); delta-vector store | L2 + L3 |
| **H14–H16** | `SearchFeedback` assembly; ledger; method cards seeded; probe harness | Person 1 & 4 contracts |
| **run** | reflections, retrieval, convergence counter | live |
| **H40+** | charts, results table, final selection | everything |

Items in H2–H14 are the difference between an agent that learns and one that accumulates confident wrong beliefs overnight.

---

## 11. Interfaces to negotiate at H0

| With | Item | Why it can't wait |
|---|---|---|
| Person 1 | `official_score` is **gated**, frequently unchanged | Their UCB assumes a raw metric; exploration must handle a flat reward |
| Person 1 | `ExperimentSpec.experiment_kind` ∈ {frame, content, capacity} | Staleness rule and phase schedule both key off it |
| Person 1 | `stability_passed=False` must feed prune logic | Otherwise noise-driven nodes get branched from |
| Person 1 | `orthogonality` term available for UCB diversity | Needs a slot in the selection formula |
| Person 2 | The Coder is **blind to scores**, sees tracebacks only | Otherwise repair becomes a second selection channel routing around the gate |
| Person 3 | `play_time_ms`, `profile_stay_time`, `comment_stay_time`, all `is_*` are targets, not inputs | The agent will try this and validation will look spectacular |
| Person 4 | `RunResult.artifacts` must carry raw per-row scores, not just a metric | Needed for δ vectors and slice attribution |

---

## 12. What we add over existing frameworks

| Capability | AIDE | AI-Scientist-v2 | AlphaEvolve | **Ours** |
|---|---|---|---|---|
| Tree / branching | ✅ | ✅ | ✅ evolutionary | ✅ |
| Cheap→expensive cascade | — | — | ✅ | ✅ |
| Multi-metric feedback | — | — | ✅ | ✅ |
| Multi-seed evaluation | — | ✅ `num_seeds=3` | — | ✅ |
| **Significance gating of accepts** | — | — | — | ✅ |
| **Adaptive-overfitting protection** | — | — | — | ✅ |
| **NO_OP vs FALSIFIED separation** | — | — | — | ✅ |
| **Redundancy via δ fingerprints** | — | — | — | ✅ |
| **Frame/content staleness** | — | — | — | ✅ |
| **Convergence-aware scheduling** | — | — | — | ✅ |
| **Unbiased-log audit channel** | — | — | — | ✅ |

Seven rows nobody else has, all cheap, all inside this module.

---

## 13. Charts to produce (H40+)

1. **val_A vs val_B divergence** over iterations — adaptive overfitting made visible
2. **Reported vs true score scatter** — Ladder rejections visible as points off the diagonal
3. **Predicted vs measured combination gain** — validates the δ-correlation model
4. **Verdict distribution** — how many NO_OP / BROKEN / REDUNDANT / FALSIFIED, and cost avoided
5. **Headroom progress** — fraction of the 0.2699 captured, per metric
6. **Cost by role** — tokens split across hypothesis generation, code diffs, repair

Charts 1, 2 and 4 are the ones that say "we know what we are doing."

---

## 14. Scope warning

Items 1–8 of the original brief are close to two people's work. If nobody can be reallocated, cut in this order:

1. Method-card ownership → Person 1 (they consume it for hypothesis generation)
2. Dashboard → a static matplotlib notebook

**Do not cut:** evaluator adapter, Ladder gate, multi-seed stability, NO_OP detector, slice attribution. Those five carry both the primary metric and the Innovation argument.

---

## 15. References

| Source | What is taken |
|---|---|
| Blum & Hardt, *The Ladder*, ICML 2015 (arXiv:1502.04585) | The accept/reject rule; parameter-free variant |
| Dwork et al., *The reusable holdout*, Science 2015 | Thresholdout; query count as the quantity to minimise |
| RewardHackingAgents, arXiv:2603.11337 (Mar 2026) | Trust regimes; reported-vs-true pairing; three-way integrity label |
| METR, RE-Bench reward hacking (Jun 2025) | 30.4% unprompted reward-hack baseline rate |
| AlphaEvolve, arXiv:2506.13131 (DeepMind) | Evaluation cascade; multi-metric feedback; diversity |
| AIDE, arXiv:2502.13138 | Solution tree; Σ(T) summarisation; the scalar-feedback failure mode |
| AI-Scientist-v2, arXiv:2504.08066 | `num_seeds=3`, `num_drafts` independent roots, `debug_prob` |
| MLE-bench, arXiv:2410.07095 | Documented agent failure modes, incl. validation overfitting |
| Dacrema et al., RecSys 2019 | Why recsys gains routinely fail to replicate under proper evaluation |
| CWM, KDD 2024 (hyz20/CWM) | Censored watch-time regression; duration bias on KuaiRand-Pure |
| Skalse et al., NeurIPS 2022 | Proxy-metric optimisation can worsen the true objective |
