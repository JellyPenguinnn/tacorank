```json
{"schema_version":"1.0","method_id":"ensemble_parallel_round_synthesis","family":"ensemble","status":"candidate","tags":["ensemble","parallel_round","alignment","composition"],"cost_tier":"medium","prerequisites":["two_confirmed_clean_members"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","date","duration_ms","long_view","verified_predictions"],"prohibition_conditions":["provisional_or_unreproducible_member","adaptive_weight_sweep"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Start from the strongest independently accepted member of the current round.
Inspect every other accepted component patch supplied by the controller, preserve
compatible improvements cumulatively, and resolve overlapping score-path changes
explicitly. The result is a new candidate, never an in-place merge of evidence
branches. It must pass the normal semantic verifier, Gate A, execution ladder,
Gate B, and protected evaluation before it can become the next-round parent.

## Preconditions

At least two confirmed, clean, independently evaluated round members are available.

## Allowed data

Only the candidate-visible training fields already authorized for the accepted members,
plus controller-verified predictions and patches from their exact commits.

## Expected effect

Retain complementary gains in one aligned and reproducible candidate.

## Falsification condition

Any gate failure, interaction regression, or failure to improve the strongest member.

## Do not use when

Any component is provisional, suspicious, unverified, or not reproducible.

## Minimal implementation

Do not average or sweep validation-selected weights. Prefer a deterministic,
bounded composition whose behavior remains attributable to the supplied members.

## Sources

No external source is required for this controller-owned composition step.
