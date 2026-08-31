```json
{"schema_version":"1.0","method_id":"ensemble_zblend_diverse","family":"ensemble","status":"candidate","tags":["ensemble","zscore","diversity","blend"],"cost_tier":"low","prerequisites":["baseline_parity","two_confirmed_clean_members"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":[]}
```

## Mechanism

Per-user z-score each accepted member's predictions, then take a fixed
equal-weight sum. Z-scoring inside each user removes scale differences
between members so only within-user ordering information combines; equal
weights avoid fitting blend weights to evaluation feedback. In the sibling
lab study the weight grid was flat (plus or minus 0.0004), and the
equal-weight blend of the LightGBM ensemble, an xendcg member, and the
CatBoost YetiRank member reached 0.6172 valid versus 0.6120 for the best
single member.

## Preconditions

At least two accepted, mechanism-diverse members exist in the run lineage
(for example the LightGBM causal-history model and the CatBoost YetiRank
model).

## Allowed data

Only the member models' own outputs recomputed deterministically inside
the candidate, plus contract-permitted columns.

## Expected effect

Offline measurement on this exact contract (2026-08-31): a per-user z-sum
of three diverse members — the accepted causal-replacement model, the
compact-rank variant, and a reference causal lambdarank — scored 0.60411
full-fidelity versus the 0.60147 parent and 0.60351 best single blend.
Diversity of independently designed feature frames is what pays; members
from one shared frame measured only 0.60216.


Sibling measurement: about +0.005 over the best single member when the
members are architecturally diverse; near zero when they are seeds of the
same model (+0.001).

## Falsification condition

No trusted improvement over the best member alone.

## Do not use when

Members are not diverse (same architecture and features), or a member was
never accepted at full fidelity.

## Minimal implementation

Recompute each member's scores deterministically with its accepted recipe
and fixed seeds, z-score per user (population ddof 0; users with zero
spread contribute zeros), sum with fixed equal weights declared in the
spec, and emit the summed score. Do not tune weights against evaluation
feedback inside the experiment; a deliberate ablation is a separate
proposal.

## Memory discipline

Train members SEQUENTIALLY inside the candidate: build one member's frame,
train, predict, then `del` the frame/dataset and `gc.collect()` before the
next member. Holding two full feature frames simultaneously exceeds the
container memory limit and the experiment dies on an OOM kill (this killed
a prior ensemble attempt). Peak memory must stay under a single member's
footprint plus predictions.

## Sources

Sibling lab study measurements (lab/PLAYBOOK.md).
