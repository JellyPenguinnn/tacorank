# TacoRank Evaluation Implementation

## Purpose and Status

This document records the implemented Person 5 evaluation, trust, final-selection,
reflection, and reporting design. It explains the intended end-to-end flow and the
rationale for the major implementation decisions. The executable authorities are
`src/tacorank/schemas.py`, `src/tacorank/evaluation/`, the production bridge in
`src/tacorank/orchestrator/live.py`, and the KuaiRand adapters under
`benchmarks/kuairand_pure/`.

The deterministic evaluation components and their integration tests are implemented.
Production composition uses the real Person 3 Docker runner, Gate B, and protected
evaluation bridge. Deterministic fake adapters remain test fixtures only. Protected
metric computation runs in a separate isolated Python process.

## Design Goals

The implementation follows five rules:

1. An evaluation is valid only when it can be traced to a verified Gate-B event and
   an immutable prediction artifact.
2. Official metrics are computed only by the hash-pinned evaluator; independent
   metrics are parity checks and diagnostics, not a replacement authority.
3. Stability is derived from verified evaluation events, never caller-supplied score
   arrays.
4. Integrity and no-op checks precede performance reward.
5. Only clean, confirmed public-validation evidence can update the trusted best.

## Module Map

| Area | Main files | Responsibility |
| --- | --- | --- |
| Shared transport | `src/tacorank/schemas.py` | Pydantic contracts persisted in the event ledger |
| Evaluation domain | `src/tacorank/evaluation/types.py` | Immutable evaluation-specific dataclasses and canonical conversion |
| Protected scoring | `src/tacorank/evaluation/adapter.py` | Gate-B binding, hash checks, route checks, scoring, seed resolution |
| Worker isolation | `src/tacorank/evaluation/_isolated_worker.py` | Clean-process loading and invocation of the official evaluator |
| Metrics | `src/tacorank/evaluation/metrics.py` | Strict labels, independent GAUC/nDCG parity, metric validation |
| Trust and stability | `trust.py`, `stability.py`, `no_op.py` | Ordered adjudication, seed aggregation, no-op detection |
| Decisions | `src/tacorank/evaluation/decisions.py` | Promotion, confirmation, rejection, parent and best eligibility |
| Diagnostics | `src/tacorank/evaluation/slices.py` | User slices, delta vectors, concentration, drift, rank diagnostics |
| Final selection | `src/tacorank/evaluation/final_selection.py` | Eligibility filtering and optional rank averaging |
| Benchmark binding | `benchmarks/kuairand_pure/` | KuaiRand metric contract, evaluator construction, submission checks |
| Reflection | `src/tacorank/reflection/` | Evidence-backed lessons and frame staleness recommendations |
| Reporting | `src/tacorank/reporting/` | Ledger-derived views and evaluation-specific judge summaries |

## End-to-End Evaluation Flow

### 1. Freeze authorities

Before a run, the contract, official evaluator, protected paths, and data manifest are
hashed. `create_evaluator_adapter()` binds the KuaiRand evaluator path, expected
evaluator hash, expected contract hash, metric registry, and optional population
manifests. Hashes supplied later in an evaluation request must match these frozen
values and the current file bytes.

This double comparison is intentional. Checking only the request would accept a
mutated local evaluator, while checking only local bytes would allow a caller to
silently change the requested contract identity.

### 2. Produce and check predictions

Candidate execution produces a prediction artifact. Gate B validates the submission
schema, finite scores, exact row count, zero-based contiguous `row_id`, and ordered
user/video identity. Duplicate `(user_id, video_id)` pairs are preserved because row
identity includes `row_id` and order.

`SubmissionCheck` creates two linked objects:

- `PredictionBatch`: artifact ID/hash plus ordered row IDs, user IDs, item IDs, and
  exact scores.
- `OutputGateEvidence`: the verified `output.checked` event identity, acceptance,
  population, artifact identity, row digest, and prediction digest.

The orchestrator persists the same two ordered digests in `OutputCheckResult`.

### 3. Resolve Gate-B evidence

`EvaluationService` requires an `output_gate_resolver`. It resolves the event by ID
and compares the resolved value with the request evidence. Evaluation fails closed if
the resolver is absent, fails, returns the wrong type, reports rejection, or returns a
different event value.

The service then verifies:

- requested population equals the Gate-B population;
- prediction artifact ID and SHA-256 equal the checked artifact;
- ordered row digest equals the checked row digest; and
- ordered prediction digest equals the checked score vector.

Passing a structurally valid evidence object is therefore insufficient. The object
must be backed by the event resolver.

### 4. Validate the route

Population and fidelity combinations are explicit:

| Population | Fidelity | Query index | Best eligibility |
| --- | --- | --- | --- |
| `internal_proxy` | `proxy` | forbidden | never |
| `public_validation` | `full` | required | possible after confirmation |
| `unbiased_audit` | `full` | forbidden | never; trust evidence only |
| `hidden_final` | `final` | forbidden | no experiment decision |

Smoke output does not call the official evaluator. Hidden-final evaluation additionally
requires verified `run.stopped` evidence. These restrictions prevent proxy, audit, and
hidden information from contaminating the public-validation search state.

### 5. Run protected scoring

The parent process validates labels, score finiteness, population row identity, and
the evaluator/contract hashes. It serializes only user IDs, binary labels, and scores
to JSON and launches:

```text
<python> -I _isolated_worker.py <evaluator_path> <expected_sha256>
```

The worker starts with isolated Python settings, a sanitized environment, a fixed
working directory, captured output, and a timeout. It hashes the evaluator again,
loads it from the explicit path, invokes `evaluate`, and returns JSON. The parent then
validates the returned metric names, ranges, finiteness, and primary aggregation.

The child-side hash check closes the time and process gap between parent verification
and import. Process isolation also prevents candidate monkeypatches or imported module
state from changing the official metric implementation.

### 6. Compare and diagnose

The official result is compared with the frozen baseline, parent, and previous best.
Comparisons require identical metric schemas and primary metric names. Prediction
change analysis records Spearman correlation, changed-row fraction, exact-score
identity fraction, and score diversity.

### 7. Resolve seed evidence

Seed confirmation accepts event IDs, not raw scores. Each ID must resolve to an
earlier `EvaluationResult` with:

- the same run and experiment;
- the same public-validation/full route;
- the same evaluator, contract, and data-manifest hashes;
- clean integrity;
- a distinct seed;
- strictly increasing attempts; and
- the same baseline, parent, and previous-best metric references.

Reference compatibility is checked for the primary score and every component metric.
The current evaluation is appended internally, so a caller cannot omit an unfavorable
current seed from the aggregate.

The aggregate metric set is the arithmetic mean of each metric over all verified seed
events plus the current event. Trust receives both the aggregate parent deltas and the
individual primary scores. The result persists only the prior event IDs; `seed_count`
must equal `len(seed_evidence_event_ids) + 1`.

### 8. Assess trust and decide

Trust adjudication is ordered deliberately:

1. Missing Gate-B evidence, hash mismatch, or forbidden inputs produces
   `suspicious/compromised`.
2. Suspected row alignment produces `suspicious/inconclusive`.
3. An unchanged prediction vector produces `no_op` before score reward.
4. Degenerate score diversity produces `suspicious/compromised`.
5. Implausible gains and cross-population sign conflicts produce suspicious evidence.
6. A highly correlated delta fingerprint produces `redundant`; stability is
   `not_applicable` because redundancy is not seed confirmation.
7. Metric-direction conflict, concentrated gain, and temporal drift add visible flags.
8. Proxy evidence can promote or prune but cannot become parent or best.
9. Full public-validation evidence enters seed-stability adjudication.

For seed scores `s_1 ... s_n`:

```text
mean = arithmetic_mean(scores)
stderr = sample_standard_deviation(scores) / sqrt(n)
eta = max(2 * stderr, 0.0016)
```

Three seeds are required for confirmation. Standard deviation above three times the
frozen baseline seed standard deviation is unstable. Aggregate movement within
`[-eta, +eta]` is inconclusive; movement below `-eta` is negative; movement above
`+eta` is accepted.

A single accepted full result requests confirmation but is not parent or best eligible.
After confirmation, the candidate can become a parent. It becomes best only when the
aggregate seed mean exceeds the previous best by more than `eta`. The decision layer
uses the aggregate mean, not the current seed score.

## Row and Prediction Digests

`ordered_row_identity_sha256()` encodes each row as canonical JSON:

```text
[integer_row_id, string_user_id, string_item_id]
```

Each record is prefixed by its eight-byte big-endian length before hashing. Length
prefixing prevents concatenation ambiguity.

`ordered_prediction_sha256()` uses the same identity plus `float.hex(score)`. Hex
encoding preserves the exact normalized binary float value and avoids decimal-format
differences. It prevents a caller from swapping scores between duplicate identities or
changing values after Gate B while retaining the same row digest.

Artifact SHA-256 and prediction digest serve different purposes. The artifact hash
binds file bytes; the prediction digest binds the parsed values that are actually sent
to the evaluator.

## Strict Label and Metric Handling

`normalize_binary_labels()` rejects booleans, strings, bytes, non-numeric objects,
non-finite values, fractions, and any numeric value other than exact zero or one.
Validation occurs before integer conversion, preventing values such as `0.9` from
silently becoming `0`.

The validator is shared by the official adapter, independent AUC/nDCG implementation,
per-user contributions, slices, and rank diagnostics. This centralization prevents
diagnostics from accepting data that protected scoring would reject.

The independent implementation reproduces the starter-kit semantics:

- GAUC uses positive-count weighting for users containing both classes;
- users without both classes do not contribute to the GAUC numerator or weight;
- nDCG@5 is averaged over users;
- zero-positive users receive zero nDCG; and
- primary is the equal-weight mean of GAUC and nDCG@5.

Independent metrics are used for parity verification and diagnostics only. Production
scores always come from the protected evaluator.

## Diagnostics and Research Evidence

User metrics support impression-count and positive-count slices. Reconstruction uses
the official metric denominators: positive weights for GAUC and user counts for nDCG.
This avoids the incorrect shortcut of averaging slice primary scores.

Per-user delta vectors decompose the exact primary delta into additive contributions.
They support:

- correlation-based redundancy detection;
- top-10-percent gain concentration checks;
- persisted float32 delta-vector artifacts with ordered-user hashes;
- row-level positive-rank diagnostics for duration or popularity buckets; and
- daily primary-score slope for temporal drift.

Rank diagnostics are explicitly non-additive. They explain where ordering changes but
must not be presented as official metric decompositions.

## Final Selection

`select_final()` filters before ranking. A candidate must be accepted, confirmed,
clean, supported by internal holdout and unbiased audit agreement, successfully
reproduced in a clean environment, and have a Val-B score. Selection then prioritizes
Val-B, public score, and deterministic experiment ID tie-breaking.

`rank_average()` is available for aligned seed or ensemble predictions. Average ranks
handle ties and normalize each vector before averaging, avoiding domination by score
scale. Rank averaging does not waive candidate eligibility requirements.

## Reflection and Reporting

Research lessons are emitted only from eligible evidence. No-op and inconclusive
results create no lessons. Negative lessons require full public-validation evidence;
accepted lessons require confirmation. Hidden-final evidence never creates planning
memory. Suspicious evidence creates integrity warnings rather than positive research
reward.

The protected decision bridge creates the canonical `LessonCandidate` from the
persisted evaluation event, its seed-evidence events, the evaluated commit, and the
original `ExperimentSpec`. The controller then appends `lesson.recorded`; it remains
the sole ledger writer. Replay exposes active lessons to later planner and coder
contexts, while reporting materializes the human-readable lesson files.

Reporting has two distinct APIs:

- `render_summary(events)` and `rebuild_views()` generate ledger-derived `STATUS.md`,
  `lessons/INDEX.md`, per-lesson Markdown files, and `reports/SUMMARY.md` projections.
- `render_evaluation_summary()` and `render_metric_table()` generate judge-facing
  evaluation detail from typed evaluation results.

The names are intentionally distinct. During integration, both branches had an
incompatible `render_summary`; preserving the event-derived name keeps CLI behavior
stable while the explicit evaluation name avoids argument-based dispatch.

## Shared Schema Integration Decisions

The incoming main branch supplied the event store, replay engine, orchestrator,
contexts, recovery flow, and nested event payloads. The evaluation branch supplied a
more detailed but incompatible flat event schema. The resolution uses the integrated
nested event transport as the backbone because every runtime caller and ledger test
depends on it.

Evaluation-specific capabilities were added to that backbone:

- `FINAL`, `UNBIASED_AUDIT`, `REDUNDANT`, and `DELTA_VECTOR` enum values;
- `PredictionChange` and enriched `TrustAssessment`;
- ordered Gate-B digests in `OutputCheckResult`;
- seed evidence and route validation in canonical `EvaluationResult`;
- artifact byte verification and metric-contract validation; and
- event hash and Markdown schema helper APIs retained from the evaluation branch.

The fake Gate-B adapter was upgraded to produce full row/user/video CSV identity and
both ordered digests. `ContextBuilder` normalizes structured `PredictionChange` to its
changed-row fraction because planner summaries intentionally consume a compact scalar.

Published baseline evidence is exempt from experiment seed-event validation. It is a
bootstrap authority verified by baseline parity rather than an experiment confirmation
chain. All non-baseline confirmed or unstable evaluations require seed aggregate
metadata and matching evidence-event counts.

## Decision Rationale Summary

| Decision | Rationale | Rejected alternative |
| --- | --- | --- |
| Resolve Gate-B events instead of trusting request objects | Prevent fabricated typed evidence | Trust caller-created evidence because it validates structurally |
| Hash ordered rows and exact predictions separately | Bind both population identity and parsed score values | Hash only user IDs or artifact bytes |
| Preserve `row_id` | Duplicate user/video pairs are legal and order-sensitive | Reduce identity to `(user_id, video_id)` |
| Use event IDs for seeds | Makes every score auditable and identity-compatible | Accept raw seed score arrays |
| Add current seed internally | Prevent selective omission | Let callers provide the complete score list |
| Aggregate every metric | Trust flags and decisions must use consistent evidence | Aggregate only the primary score |
| Compare best with seed mean | Best updates represent stable performance | Compare the latest seed with best |
| Validate labels before coercion | Avoid fractional or string truncation | Convert with `int(label)` first |
| Run evaluator in `python -I` worker | Separate candidate interpreter state | Import evaluator in the harness process and hash afterward |
| Keep official evaluator authoritative | Preserve competition contract | Replace it with an independent reimplementation |
| Treat redundancy stability as not applicable | Correlation is not seed confirmation | Mark a one-seed redundant result confirmed |
| Keep audit and hidden results out of decisions | Avoid adaptive leakage | Feed all measured scores back to planning |
| Filter final candidates before ranking | Never let a high unsafe score win | Select maximum score and inspect trust afterward |
| Use incoming nested event schema | Matches event store and orchestrator flow | Preserve the branch-local flat payload schema |

## Verification

The merged suite covers protected hash rejection, evaluator-process isolation, official
metric parity, fractional/string/boolean labels, duplicate rows, score swapping,
fabricated Gate-B evidence, population routes, incompatible seed events, duplicate
seeds, current-seed inclusion, aggregate best decisions, no-op handling, slices,
delta-vector reconstruction, final selection, reflection, event replay, reporting, and
the complete fake lifecycle.

Run the full suite with:

```text
PYTHONPYCACHEPREFIX=/tmp/tacorank-pycache .venv/bin/python -m pytest
```

Run syntax compilation with:

```text
python3 -m compileall -q src benchmarks
```

At the time of the latest wiring review, the complete suite passed with 488 tests;
11 platform-specific cases were skipped on macOS.

## Known Limitations and Next Work

1. Integrated lifecycle tests use deterministic fake adapters, while production uses
   `ProtectedEvaluationBridge`. The bridge resolves canonical `EvaluationRequest`,
   `output.checked`, artifact/digest identity, reference predictions, and seed events
   into `EvaluationInputs`.
2. The isolated evaluator worker is a process boundary, not a complete OS/container
   sandbox. Person 3 must enforce filesystem, network, resource, and candidate-process
   isolation in the real runner.
3. Full FM baseline reproduction requires the external KuaiRand data and is not part of
   the unit-test suite.
4. Production population manifests are frozen from the deployment data views. Unit
   tests may omit them only when no expected data-manifest hash is configured.
5. Candidate finalization remains deliberately strict: clean reproduction must match
   the trusted best score exactly before final inference and submission checking.

These limitations are explicit boundaries. They must not be bypassed by weakening hash,
route, trust, or final-selection validation.
