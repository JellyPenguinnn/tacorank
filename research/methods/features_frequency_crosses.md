```json
{"schema_version":"1.0","method_id":"features_frequency_crosses","family":"features","status":"candidate","tags":["frequency","feature_interaction","cold_start","residual"],"cost_tier":"low","prerequisites":["baseline_parity"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","long_view","verified_predictions"],"prohibition_conditions":["validation_fitted_feature"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/inference.py"],"sources":["https://www.kdd.org/kdd2020/accepted-papers/view/autofis-automatic-feature-interaction-selection-in-factorization-models-for.html"]}
```

## Mechanism

Add smoothed train-frequency and fixed crossed-frequency residual features for
user, item, author, and tab exposure regimes.

## Preconditions

Counts are fitted exclusively on training interactions.

## Allowed data

Training entity identities, context, labels, and verified predictions.

## Expected effect

Correct systematic FM errors across head, torso, and cold entities.

## Falsification condition

Frequency features only reproduce popularity or do not alter within-user ranks.

## Do not use when

Counts require validation rows or adaptive bucket search.

## Minimal implementation

Use fixed log-count transforms, smoothing, a small fixed cross set, bounded
residuals, and deterministic unseen-entity fallback.

## Sources

AutoFIS: Automatic Feature Interaction Selection in Factorization Models.
