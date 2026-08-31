```json
{"schema_version":"1.0","method_id":"static_feature_expansion_known_negative","family":"features","status":"known_negative","tags":["static","ablation","known-negative"],"cost_tier":"low","prerequisites":[],"allowed_data":["train_interactions","user_id","video_id"],"prohibition_conditions":["static_feature_expansion"],"sources":["kuairand-starter-kit/README.md"]}
```

## Mechanism

Add raw music, video type, upload type, or coarse static user fields directly
to the existing FM, or change only embedding capacity.

## Preconditions

None. This card records a completed negative ablation rather than a proposal.

## Allowed data

The original five-field FM input and the previously tested static side fields.

## Expected effect

No trusted improvement; the local three-seed ablation was within noise and
slightly below the five-field baseline.

## Falsification condition

Not applicable because this method is excluded from autonomous selection.

## Do not use when

Always prohibit as a standalone direction unless a distinct candidate-dependent
interaction mechanism is explicitly identified and isolated.

## Minimal implementation

Do not implement or tune this method.

## Sources

The starter-kit's documented static-feature and embedding-capacity ablations.
