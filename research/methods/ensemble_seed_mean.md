```json
{"schema_version":"1.0","method_id":"ensemble_seed_mean","family":"ensemble","status":"candidate","tags":["ensemble","seeds","variance","zscore"],"cost_tier":"low","prerequisites":["baseline_parity","standard_public_evaluation_complete"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":[]}
```

## Mechanism

Refit the single accepted best model under a small set of fixed seeds and
serve the per-user z-scored rowwise mean. Seed averaging removes
seed-specific ranking noise; the sibling lab study measured about +0.001
from a five-seed mean of one LightGBM ranker, and the earlier harness run
run_017_global_repro50 confirmed the same construction end to end
(exp_002, +0.0002 over its single-seed parent).

## Preconditions

Exactly one accepted full-fidelity model parent exists (this card needs no
second architecture, unlike ensemble_zblend_diverse).

## Allowed data

Only the parent's own recipe recomputed under fixed seeds; the train-split-
only feature rule applies unchanged.

## Expected effect

Small but nearly free: roughly +0.001, with unchanged mechanism risk.

## Falsification condition

No trusted improvement over the single-seed parent.

## Do not use when

Two diverse accepted members already exist — prefer ensemble_zblend_diverse
then, which measured about +0.005 in the sibling study.

## Minimal implementation

Recompute the accepted parent's pipeline under fixed declared seeds (for
example invocation.seed plus fixed offsets 100 and 200), z-score each
member's predictions within each user (population ddof 0), average, and
emit. No weight tuning, no member selection, deterministic and finite.

## Sources

Sibling lab study (lab/PLAYBOOK.md) and run_017_global_repro50 exp_002.
