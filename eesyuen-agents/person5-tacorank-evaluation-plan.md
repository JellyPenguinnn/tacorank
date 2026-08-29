# Person 5 — Evaluation Implementation Plan

**Against TacoRank Memory Schema v1.0.**
Scope: everything between a verified prediction file and a trusted decision, plus the evidence that proves it.

Per schema §18, Person 5 owns: `MetricSet`, `evaluation.completed`, trust verdicts, `experiment.decided`, lessons, and resource/intervention reports.

---

## 1. What the schema already provides — do not rebuild

Several mechanisms I had planned to design are already first-class in v1.0. Building parallel versions would violate §18 ("no teammate may define a private alternative meaning").

| Mechanism | Already in schema | My earlier draft |
|---|---|---|
| NO_OP detection | `evaluation.completed.prediction_change.spearman_vs_parent` | proposed as new |
| Verdict enum | `trust.verdict` ∈ {accepted, inconclusive, negative, no_op, suspicious} | proposed 6-way |
| Integrity three-way | `trust.integrity` ∈ {clean, compromised, inconclusive} | proposed identically |
| Multi-seed stability | `trust.stability` ∈ {single_seed, confirmed, unstable, not_applicable} | proposed as new field |
| Selection-budget meter | `evaluation.completed.public_query_index` | proposed as new counter |
| Internal holdout population | `population = internal_proxy` | proposed as new |
| Convergence projection | §10.3 — only `full` + `public_validation` counts | proposed as new |
| Alignment / diversity gate | `output.checked` (Gate B) | proposed inside my adapter |
| Frame staleness | `lesson.status_changed` with `new_status = stale`; §9.6 example is literally "measured under an objective frame superseded by exp_0011" | proposed as new rule |
| Proxy cannot update best | Invariant 15 | proposed as a convention |
| Low score ≠ failure | Invariant 13 | proposed as a convention |

**Consequence:** my job is to *populate* these fields correctly, not to add a second trust layer beside them. Three genuine gaps remain (§2).

---

## 2. Six schema extensions to request from Person 1

Each is minimal and justified. Ask for all six at H0; retrofitting after the fold/replay code exists is expensive.

### 2.1 Add `unbiased_audit` to the population enum

Currently `internal_proxy | public_validation | hidden_final`. The randomized-exposure log (`log_random_4_22_to_5_08_pure.csv`, restricted to 20220422–28) is a missing-at-random sample and is explicitly sanctioned by the starter-kit README as an extra unbiased validation set.

It is **not** `public_validation` (must not increment `public_query_index`, must not affect convergence) and **not** `internal_proxy` (different provenance, different meaning). It needs its own value.

```json
"population": "unbiased_audit"
```

Rules: never affects convergence, never updates trusted best, never becomes a parent. Its only consumer is the `suspicious` verdict.

### 2.2 Add `redundant` to `trust.verdict`

A result can be statistically significant *and* capture a signal an accepted node already captured. Mapping this to `negative` destroys the distinction between "does not work" and "already have it," which drive opposite planner actions.

```json
"trust": { "verdict": "redundant", "flags": ["delta_corr_0.83_vs_exp_0004"] }
```

### 2.3 Declare diagnostic metrics in `COMPETITION.md`

§7.3 permits extra metrics "only if declared as diagnostic metrics in the contract." Slice attribution therefore has a legal home. Declare:

```
gauc, ndcg_at_5                                   # required
slice_impr_le2, slice_impr_3_5, slice_impr_6_12, slice_impr_gt12
slice_tau_short, slice_tau_mid, slice_tau_flat18
slice_pop_cold, slice_pop_warm, slice_pop_hot
slice_pos1, slice_pos2, slice_pos3plus
unreachable_user_fraction
```

The full per-slice report goes to an `ArtifactRef` of kind `metrics`; only the digest sits in `MetricSet`.

### 2.4 Add `delta_vector` to the ArtifactRef kind enum

Per-user delta vectors (22,377 float32 ≈ 90 KB) are decision-bearing evidence, hash-addressed like any other artifact. `metrics` would work but conflates two different consumers.

### 2.5 Add `orthogonality` to the planner context (§13.1)

One float: `1 − max corr(δ_candidate_family, δ_accepted)`. Feeds a diversity term in node selection. Without a slot in the context contract it cannot reach Person 1.

### 2.6 Define `family → kind` in `COMPETITION.md`

§11 has `family` but no frame/content/capacity distinction, which the staleness rule needs. This needs **no new field** — derive it:

```
frame:    objective, grouping
content:  feature, sequence, multitask, censored_regression, temporal
capacity: model_class
```

Frame moves change what "better" means and invalidate prior content measurements. Capacity is gated on a content move landing first (the organizers measured k=8/16/32 as flat for a 5-field FM).

---

## 3. The key architectural decision: where the Ladder lives

The Blum & Hardt Ladder mechanism (ICML 2015) must gate what the planner perceives as reward, or the agent chases seed noise. The organizers hand us the noise floor: FM 5-seed std = **0.0008**, and contract ε = 0.002 ≈ 2.5σ.

The schema forbids a second authoritative state file (§20) and makes ledger lines immutable (invariant 2). So the Ladder cannot be a mutable side-table. Two candidate homes:

| Option | Verdict |
|---|---|
| New field on `evaluation.completed` | ✗ The event records measurement, not policy. Policy belongs in the contract |
| **`COMPETITION.md` comparison rule** | ✓ §9.5 already says `best_eligible` requires "a primary score greater than the current trusted best **under the contract's comparison rule**" |

**Decision: the Ladder *is* the contract's comparison rule.** Add to `COMPETITION.md § Convergence and resource limits`:

```md
## Comparison rule
A candidate primary score supersedes the trusted best only when

    mu > trusted_best + eta,   eta = max(2 * stderr(seeds), 0.0016)

where mu and stderr are computed over all completed full public-validation
seeds for that experiment, and 0.0016 is 2x the official baseline's 5-seed
standard deviation of 0.0008.

Rejected candidates leave `best_primary_score` unchanged. Planner contexts
report the trusted best, never the rejected candidate's raw score.
```

This gives us, for free:

- `best_eligible` in `experiment.decided` becomes the Ladder decision — no new machinery
- The contract is human-frozen and hash-protected, so the rule is auditable
- `events.jsonl` still records the **raw** `metric_set` and all three deltas: the ledger stays honest for judges
- The gate lives in the projection layer, exactly where §13.1 already excludes suspicious scores from being read as positive rewards

**One separation to hold firm:** contract ε = 0.002 governs *convergence* (§10.3, organizer rule, not ours to change). Ladder η governs *trusted-best updates*. They are different thresholds with different jobs. Conflating them would let our own conservatism declare convergence on a run still improving.

---

## 4. Module inventory

```
src/tacorank/evaluation/
  metrics.py          MetricSet construction; wraps pristine evaluate.py
  populations.py      4-population manifest + deterministic splits
  slices.py           per-user decomposition; diagnostic metrics; delta vectors
  trust.py            verdict, stability, integrity computation
  seeds.py            multi-seed escalation policy
  ladder.py           the contract comparison rule (pure function)
  decide.py           builds experiment.decided payloads
src/tacorank/reflection/
  lessons.py          lesson.recorded / lesson.status_changed emission
  staleness.py        frame-move invalidation sweep
src/tacorank/reporting/
  resources.py        ResourceDelta folding, intervention count
  charts.py           six deliverable figures
  summary.py          SUMMARY.md projection
benchmarks/kuairand_pure/
  evaluator_adapter.py   pristine-copy caller, hash-pinned
  submission_adapter.py  final CSV writer + --check
  contract_metrics.py    diagnostic metric registry
```

**Zero LLM calls in any of these.** RewardHackingAgents (arXiv:2603.11337, Mar 2026) measured evaluator-tamper attempts in ~50% of episodes where evaluation was mutable and no instruction to tamper was given; locking integrity drove observed compromise to zero. METR measured o3 reward-hacking in 30.4% of RE-Bench runs unprompted. An LLM asked "is +0.0003 real?" answers "the direction looks right."

---

## 5. Populations

| Population | Source | Rows | Cost | Affects convergence | Can update best |
|---|---|---|---|---|---|
| `internal_proxy` | train 20220415–21, first 5 impressions/user | 100,159 | free | no | no |
| `public_validation` | official valid, 80% of users by hash (val-A) | ~100k | 1 query | **yes** | **yes** |
| `public_validation` (audit) | official valid, 20% of users (val-B) | ~25k | rare | no | tie-break only |
| `unbiased_audit` | `log_random`, 20220422–28 only | ~490k | free | no | no |
| `hidden_final` | organizer-held | 170,588 | once | — | — |

### 5.1 Internal proxy construction

Naive 04-15→04-21 has median 7 impressions/user and 18.4% zero-positive users. Official validation has median 4 and 30.3%. Different geometry means the proxy lies. Keeping the first 5 impressions per user fixes it:

| | inner train | raw holdout | **subsampled** | official valid |
|---|---|---|---|---|
| rows | 891,418 | 249,694 | **100,159** | 124,909 |
| users | 25,151 | 24,262 | **24,262** | 22,377 |
| imp/user median | 24 | 7 | **5** | 4 |
| zero-positive | 6.9% | 18.4% | **26.2%** | 30.3% |
| GAUC covers | — | — | **64.3%** | 57.8% |

```python
def build_internal_proxy(train_rows):
    hold = train_rows[train_rows.date >= 20220415].copy()
    hold["k"] = hold.groupby("user_id").cumcount()
    return hold[hold.k < 5]          # deterministic, no RNG, survives restart
```

**Trap:** a model evaluated on `internal_proxy` trained on 891,418 rows, not 1,141,112 — 22% less. Its absolute score is not comparable to a `public_validation` score or to 0.6016. Valid uses: comparing two experiments *within* `internal_proxy`, and sign agreement with `public_validation`. Never absolute comparison.

### 5.2 val-A / val-B split

```python
def split_val(user_ids, ratio=0.8, salt="tacorank-2026"):
    """By USER, not row — GAUC/nDCG aggregate per user.
    Hash, not shuffle, so the split is identical after restart (§15)."""
    uniq = np.unique(user_ids)
    h = np.array([int(hashlib.md5(f"{salt}{u}".encode()).hexdigest()[:8], 16) for u in uniq])
    return np.isin(user_ids, list(uniq[h % 100 < ratio * 100]))
```

Only val-A increments `public_query_index`. val-B is queried at branch checkpoints and at final selection, never to choose between siblings.

---

## 6. Evaluator adapter

```python
PRISTINE_EVAL = "/opt/pinned/evaluate.py"   # outside every editable root
CONTRACT_SHA  = "<from COMPETITION.md>"

def build_metric_set(population, user_ids, labels, scores) -> MetricSet:
    assert sha256_file(PRISTINE_EVAL) == EVALUATOR_SHA   # invariant 6
    r = pristine_evaluate(user_ids, labels, scores)      # called, never wrapped
    return MetricSet(
        metrics={"gauc": r["GAUC"], "ndcg_at_5": r["nDCG@5"], **slice_metrics(...)},
        primary_metric_name="primary",
        primary_score=r["primary"],
    )
```

Rules:
- `evaluate.py` is **called, never modified, never wrapped, never reimplemented as a faster version.**
- Slice attribution computes a second per-user decomposition *outside* `evaluate.py`.
- `evaluator_sha256` and `contract_sha256` on every `evaluation.completed` (invariant 9).
- No evaluation before `output.checked.accepted = true` (invariant 10).

### 6.1 P0 gate — blocks the project

Reproduce all six references before anything else:

```
valid:  random 0.4834 | pop 0.5807 | fm 0.6016
test:   random 0.4753 | pop 0.5715 | fm 0.5946
```

Also independently reimplement GAUC and nDCG@5 from the `evaluate.py` docstring and assert agreement to 1e-9. This forces internalising the pinned conventions: zero-positive users score nDCG 0 **and are included**; GAUC counts only `0 < positives < impressions`, weighted by positive count; gain is `2^rel − 1`.

Any mismatch → stop. Every downstream number is measured on that ruler.

---

## 7. Multi-seed protocol

The schema stores one seed per `evaluation.completed`. Stability is derived across events for the same experiment at the same fidelity and population.

| Situation | Seeds | Rationale |
|---|---|---|
| `internal_proxy`, any fidelity | 1 | screening only, never updates best |
| `public_validation` full, first pass | 3 | matches AI-Scientist-v2's `num_seeds` default |
| raw Δ within `[0, η)` | escalate to 5 | resolves `inconclusive` internally |
| final candidate | 5 | selection quality |

```
stability = single_seed   if n == 1
          = unstable      if stdev(seeds) > 3 * 0.0008
          = confirmed     if n >= 3 and stdev within expectation
```

**Seed-independence check, before trusting any standard error:** run the FM baseline at 3 seeds and confirm std lands near 0.0008. If it returns 0.00001, a random source isn't wired to the seed (data shuffle, feature subsample, model init, bagging) — η collapses to the floor and the gate silently weakens while every score looks reassuringly stable.

---

## 8. Slice attribution

GAUC and nDCG are means over users, so the **user-level decomposition is exact and free**. Item- and duration-level slices are *not* exact decompositions — use the normalised within-group rank of each positive as the diagnostic quantity, and say so in the README.

```
primary 0.6104   (parent_delta +0.0071, best_eligible true, verdict accepted)
  gauc      0.6702  (+0.0092  →  2.71% of GAUC headroom)
  ndcg_at_5 0.5506  (+0.0050  →  2.49% of nDCG headroom)

  by impressions:  <=2 (32.4%) -0.004 | 3-5 +0.006 | 6-12 +0.012 | >12 +0.021
  by tau band:     short<7s -0.006 | mid +0.008 | flat-18s (75.4%) +0.011
  by item pop:     cold -0.007 | warm +0.009 | hot +0.014
  by positives:    1 +0.004 | 2 +0.009 | 3+ +0.015
  unbiased_audit:  +0.0058   (agrees)
  internal_proxy:  +0.0063   (sign agrees)
  unreachable:     30.3% zero-positive users, nDCG structurally 0
```

Slice rationale, each tied to a measured property of this dataset:

- **impressions** — 32.4% of validation users have ≤2 impressions; nDCG@5 is near-binary for them
- **τ band** — `long_view` is a threshold on watch time: `τ(d) = 6000` if `d < 7s` else `min(0.97d, 18000)`, reconstructing the label at 99.47% train / 99.52% valid / 99.69% random log. 75.4% of rows sit on the flat 18s threshold. This slice surfaces duration bias **from the agent's own data** rather than from having read CWM
- **positives** — GAUC weights by positive count, so gains there are worth more
- **item popularity** — cold-start regression detection

### 8.1 Headroom normalisation

Absolute delta is a poor yardstick because the metrics have very different room:

| | baseline (test) | ceiling | headroom | value of +0.01 |
|---|---|---|---|---|
| gauc | 0.6610 | 1.0000 | 0.3390 | 2.95% |
| ndcg_at_5 | 0.5282 | 0.7289 | **0.2007** | **4.98%** |
| primary | 0.5946 | 0.8645 | 0.2699 | 3.71% |

A point of nDCG is worth 1.7× a point of GAUC in headroom terms. Report both normalised.

### 8.2 Delta vectors

```
δ_e[u] = primary_contribution_e(u) − primary_contribution_parent(u)     δ_e ∈ R^22377
```

Free — the array already exists from slice computation. Stored as an `ArtifactRef` of kind `delta_vector`. Five uses:

1. **Redundancy** — `corr(δ_new, δ_accepted) > 0.7` → `trust.verdict = redundant`. Likely to fire between multitask-with-`is_click` and censored watch-time regression, since `is_click` has φ = 0.761 with the label and watch time determines it outright
2. **Combination-gain prediction** — `predicted ≈ Δ₁ + Δ₂·(1 − ρ)`, the quantitative form of the 1+1<2 effect. Track predicted vs measured across the run
3. **Ensemble selection** — rank-averaging helps in proportion to *low* δ correlation; pick members analytically
4. **Frontier** — complement of the union of improved users, characterised in one line for the planner
5. **Orthogonality** — the diversity term in §2.5

---

## 9. Trust verdict computation

The decision function, in order. First match wins.

```python
def compute_trust(ev, parent, accepted_nodes, contract) -> Trust:
    # 1. did the change do anything at all?
    if ev.prediction_change.spearman_vs_parent > 0.99:
        return Trust("no_op", flags=["spearman_0.99"])

    # 2. rules violation — Person 3's guards fired
    if ev.contract_violations:
        return Trust("suspicious", integrity="compromised", flags=ev.contract_violations)

    # 3. bias exploitation: public up, unbiased flat or down
    if ev.public_delta > 0 and ev.unbiased_delta <= 0:
        return Trust("suspicious", integrity="inconclusive", flags=["unbiased_disagrees"])

    # 4. adaptive overfitting: public up, internal proxy down
    if ev.public_delta > 0 and ev.proxy_delta < 0:
        return Trust("suspicious", integrity="inconclusive", flags=["proxy_sign_disagrees"])

    # 5. implausible jump
    if ev.parent_delta > 0.05:
        return Trust("suspicious", integrity="inconclusive", flags=["delta_gt_0.05_check_leak"])

    # 6. already captured
    if max_delta_corr(ev, accepted_nodes) > 0.7:
        return Trust("redundant", flags=[f"delta_corr_{...}"])

    # 7. the contract comparison rule (the Ladder)
    if ladder_supersedes(ev.seed_scores, contract.trusted_best):
        flags = []
        if slice_concentration(ev) > 0.70:  flags.append("fragile_concentrated_gain")
        if sign(ev.d_gauc) != sign(ev.d_ndcg): flags.append("lopsided_half_credit")
        return Trust("accepted", integrity="clean", flags=flags)

    # 8. positive but under the floor — escalate seeds, do not report yet
    if 0 < ev.parent_delta:
        return Trust("inconclusive", flags=["escalate_to_5_seeds"])

    return Trust("negative")
```

### 9.1 Why `inconclusive` must not reach the planner

`inconclusive` means `0 < Δ < η` — positive but below the noise floor. Surfacing it leaks "it almost worked," and the natural response is to try variants pushing it over. **That is precisely the noise-chasing the Ladder exists to prevent.**

Resolve it internally: escalate to 5 seeds, then emit whatever it resolves to. §13.1's exclusion list should gain one line: *provisional and inconclusive evaluations are excluded from planner contexts.*

### 9.2 The NO_OP check earns the whole layer

The most common silent failure in an overnight run is a diff that looks plausible, executes cleanly, and changes nothing — a config flag never read, a feature list overwritten downstream, a loss swapped while the old optimiser is still called. The score returns parent ± noise, the agent concludes "grouped softmax doesn't help on this data," writes it to memory, and **permanently poisons the highest-prior direction on the roadmap.**

`spearman_vs_parent` is already a schema field. Populate it, branch on it, and **do not emit a lesson** on `no_op` — the hypothesis stays open, only the implementation failed.

---

## 10. `experiment.decided`

| verdict | decision | reason_code | parent_eligible | best_eligible |
|---|---|---|---|---|
| accepted | `accept` | `TRUSTED_IMPROVEMENT` | true | ladder result |
| accepted (proxy) | `promote` | `PROXY_PASSED` | false | false |
| redundant | `reject` | `SIGNAL_ALREADY_CAPTURED` | false | false |
| suspicious | `reject` | `INTEGRITY_UNVERIFIED` | **false** | false |
| no_op | `reject` | `NO_PREDICTION_CHANGE` | false | false |
| inconclusive | *(not terminal)* | escalate seeds first | — | — |
| negative | `reject` | `NO_IMPROVEMENT` | false | false |

Invariant 14 already forces `parent_eligible = true` to require a verified full public-validation result with verdict `accepted` and integrity `clean`, so `suspicious` nodes are excluded from parenthood by the schema itself. Good.

---

## 11. Lessons and staleness

Per §9.6, a lesson is permitted only from a verified positive or negative result, exhausted recovery exposing a reusable constraint, a suspicious result exposing an integrity rule, or supersession. **One-off syntax mistakes are not lessons — and neither is `no_op`.**

Seed `research/methods/*.md` before the run with the organizers' measured dead ends, so the agent doesn't spend iterations rediscovering them:

```md
# static_categorical_features
## Falsification condition
Already falsified by the organizer ablation: CWM's 13 fields scored 0.5940
against 0.5950 for the 5-field baseline — inside noise, marginally worse.
Embedding capacity k=8/16/32 gave 0.5895/0.5902/0.5887.
## Do not use when
The model is an FM over categorical crosses. user_id x video_id already
absorbs the signal, and under within-user ranking any term constant within a
user contributes exactly zero, so user-side first-order features cannot move
the score at all.
## Minimal implementation
Retry only as numeric statistic fields inside a GBDT, never as coarse
categoricals inside FM.
```

### 11.1 Frame staleness sweep

When an experiment with `family ∈ {objective, grouping}` is accepted, every prior `content`-family negative result was measured under a superseded objective. It is **stale, not falsified**:

```python
def sweep_staleness(accepted, ledger):
    if family_kind(accepted.family) != "frame":
        return
    for lesson in active_lessons(ledger, category="research_result"):
        if family_kind(lesson.tags) == "content" and lesson.measured_before(accepted.seq):
            emit("lesson.status_changed", {
                "lesson_id": lesson.id, "new_status": "stale",
                "reason": f"Measured under an objective frame superseded by {accepted.experiment_id}.",
                "source_event_ids": [accepted.event_id]})
```

Without this the agent permanently writes off directions that were only ever measured under the wrong objective. The §9.6 schema example anticipates exactly this case.

---

## 12. Convergence and submission

### 12.1 Convergence

Per §10.3 — untouched by me. Only verified `evaluation.completed` with `population = public_validation` and `fidelity = full` count. Contract ε = 0.002, patience N = 3. Proxy, smoke, invalid, suspicious and hidden-final never affect it.

I own one derived signal fed to the planner:

```python
if consecutive_non_improving_full_evaluations == 2:
    planner_hint = "FORCE_HIGH_VARIANCE"   # untried family, not a variant
```

The stopping rule is fixed; choosing experiment order to avoid a premature plateau is legitimate experimental design. Document it in `SUMMARY.md`.

**Do not stop early to look cheap.** Feasibility is 15%, graded in three coarse tiers, and scored *only among submissions that beat the baseline*. Solid dominates frugal.

### 12.2 Final selection

Invariant 20: only after deterministic stop and clean reproduction.

```
❌ argmax over public_validation      ← by construction the most overfitted point
✅ filter trust.verdict == accepted AND integrity == clean AND fidelity == full
   → require internal_proxy sign agreement
   → require unbiased_audit agreement
   → rank by val-B
   → rank-average across 5 seeds
   → submission_adapter --check
```

Rank-average, not score-average: only relative order is read, and seed score scales need not be comparable.

### 12.3 Pre-flight

```
[ ] evaluator_sha256 unchanged since baseline.verified
[ ] contract_sha256 unchanged
[ ] output.checked accepted on the test split
[ ] row_id 0-based, contiguous; never joined on (user_id, video_id) — 3.06% duplicates, up to 12x
[ ] finite scores; unique_scores > 1% of rows
[ ] chosen node: accepted + clean + full + proxy sign agrees + unbiased agrees
[ ] results table: gauc, ndcg_at_5, deltas vs 0.6610 / 0.5282
[ ] resource totals folded from event deltas, provider and estimated tokens reported separately
[ ] manual_interventions summed from manual.intervention events, never self-reported
```

### 12.4 Result tiers

| primary (test) | Δ | verdict |
|---|---|---|
| 0.5946 | 0 | baseline |
| < 0.600 | < +0.005 | **noise, not a result** — under 2.5σ |
| 0.605–0.615 | +0.01–0.02 | solid; the objective change worked |
| 0.615–0.635 | +0.02–0.04 | **strong**; multiple directions landed |
| 0.635–0.645 | +0.04–0.05 | excellent |
| > 0.65 | > +0.055 | **suspicious — hunt for a leak** |
| 0.8645 | +0.270 | oracle ceiling |

Given features and capacity are measured dead ends, realistic target is **0.62–0.64**. Progress framing: `(score − 0.5946) / 0.2699`; the baseline already captures 30.7% of the attainable range.

---

## 13. Build order and tests

Aligned with schema §19 — deterministic fakes first, real evaluator last.

| Phase | Deliverable | Test |
|---|---|---|
| **E0** | `evaluator_adapter.py`; independent metric reimplementation | agreement to 1e-9; all six references reproduced |
| **E1** | `prediction_change.spearman_vs_parent` populated | inject an identical-predictions patch → verdict `no_op`, no lesson emitted |
| **E2** | `ladder.py` as a pure function of (seeds, trusted_best, contract) | **50 pure-noise draws at σ=0.0008 → zero supersessions**; a +0.01 draw supersedes; rejected draws leave `best_primary_score` unchanged |
| **E3** | `populations.py`; four-population manifest | split is byte-identical after restart (§15); `internal_proxy` geometry matches the table in §5.1 |
| **E4** | `slices.py`; diagnostic metrics; delta vectors | per-user decomposition sums to the official `primary` within 1e-9 |
| **E5** | `trust.py` full decision function | golden fixtures for all seven verdict paths |
| **E6** | `decide.py`; lessons; staleness sweep | illegal-transition tests per §12; frame move marks content lessons stale |
| **E7** | `resources.py`; charts; `SUMMARY.md` | totals equal the sum of event deltas, never a mutable counter (invariant 16) |

The E2 test, written before any real data:

```python
ladder = LadderRule(floor=0.0016)
best = 0.6000
for _ in range(50):
    seeds = [0.6000 + random.gauss(0, 0.0008) for _ in range(3)]
    assert not ladder.supersedes(seeds, best)      # 50 noise draws, zero accepts
assert ladder.supersedes([0.6100, 0.6105, 0.6098], best)
```

Fifty pure-noise draws, zero accepts. This is README evidence as much as a unit test.

---

## 14. Charts (finalization phase)

1. **val-A vs val-B divergence** over full evaluations — adaptive overfitting made visible
2. **Reported vs raw primary scatter** — Ladder rejections visible off the diagonal
3. **Predicted vs measured combination gain** — validates the δ-correlation model
4. **Verdict census** — counts of no_op / redundant / suspicious / negative, and full-run cost avoided
5. **Headroom progress** — fraction of 0.2699 captured, per metric
6. **Resource by role** — provider and estimated tokens reported separately (§7.2)

Charts 1, 2 and 4 are the ones that say "we know what we are doing."

The single most persuasive number, if the A/B arms run:

```
raw-feedback path  → hidden test:  |val_best − test|
laddered path      → hidden test:  |trusted_best − test|
```

The leaderboard-accuracy gap. If the gated path transfers better, that is a measured result, not a claim.

---

## 15. Interface asks, in priority order

| With | Ask | Cost of delay |
|---|---|---|
| Person 1 | The six schema extensions in §2 | Retrofitting after fold/replay exists is expensive |
| Person 1 | Ladder as the contract comparison rule (§3) | Changes `best_eligible` semantics; must precede selection code |
| Person 1 | Planner reward may be flat for long stretches | Their UCB must tolerate an unchanged `best_primary_score` |
| Person 1 | `inconclusive` excluded from planner contexts (§13.1) | Otherwise the gate leaks "almost worked" |
| Person 2 | Coder context excludes all scores; tracebacks only | Otherwise repair becomes a second selection channel around the gate |
| Person 3 | `play_time_ms`, `profile_stay_time`, `comment_stay_time` and every `is_*` column are targets, not inputs | The agent will try this and validation will look spectacular |
| Person 4 | `predictions` artifact carries raw per-row scores, not a summary | δ vectors and slices are impossible without it |

---

## 16. References

| Source | Taken |
|---|---|
| Blum & Hardt, *The Ladder*, ICML 2015 (arXiv:1502.04585) | The comparison rule; parameter-free variant |
| Dwork et al., *The reusable holdout*, Science 2015 | Query count as the quantity to minimise |
| RewardHackingAgents, arXiv:2603.11337 (Mar 2026) | ~50% unprompted evaluator-tamper rate; three-way integrity label |
| METR RE-Bench (Jun 2025) | 30.4% unprompted reward-hacking baseline |
| AlphaEvolve, arXiv:2506.13131 | Evaluation cascade; multi-metric feedback improves the single target |
| AIDE, arXiv:2502.13138 | Solution tree; the scalar-feedback failure mode |
| AI-Scientist-v2, arXiv:2504.08066 | `num_seeds=3`; independent root drafts |
| MLE-bench, arXiv:2410.07095 | Validation overfitting as a documented agent failure |
| CWM, KDD 2024 (hyz20/CWM) | Censored watch-time regression; duration bias on KuaiRand-Pure |
| Dacrema et al., RecSys 2019 | Recsys gains routinely fail to replicate under proper evaluation |
