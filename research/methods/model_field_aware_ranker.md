```json
{"schema_version":"1.0","method_id":"model_field_aware_ranker","family":"model","status":"candidate","tags":["field_aware","feature_interaction","compact","parent_replacement"],"cost_tier":"high","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","duration_ms","long_view"],"prohibition_conditions":["unbounded_model_growth"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2302.01115"]}
```

## Mechanism

Replace FM with a compact field-aware ranker that gives user-item,
user-author, and user-tab interactions separate low-rank parameters.

## Preconditions

The objective/data frame is verified and the model fits CPU limits.

## Allowed data

Only contract-permitted categorical fields, duration, and training labels.

## Expected effect

Represent personalized field interactions that shared FM factors blur together.

## Falsification condition

The model exceeds resource bounds, collapses, or fails to beat simpler rankers.

## Do not use when

It requires unbounded embeddings, external features, or GPU execution.

## Minimal implementation

Use small fixed dimensions, deterministic initialization, bounded sampling, and
emit direct finite scores without an FM blend.

## Sources

PEPNet: Parameter and Embedding Personalized Network.
