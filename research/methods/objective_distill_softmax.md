```json
{"schema_version":"1.0","method_id":"objective_distill_softmax","family":"objective","status":"candidate","tags":["distillation","softmax","teacher_student"],"cost_tier":"medium","prerequisites":["baseline_parity","verified_best_prediction"],"allowed_data":["train_interactions","user_id","video_id","long_view","verified_predictions"],"prohibition_conditions":["teacher_prediction_unverified"],"implementation_targets":["solution/candidate.py","solution/features.py","solution/model.py","solution/train.py","solution/inference.py"],"sources":[]}
```

## Mechanism

Distill a verified parent score distribution into a compact student residual
without replacing the parent score scale.

## Preconditions

The teacher predictions are verified and available under the run contract.

## Allowed data

Only training interactions, legal labels, and verified parent predictions.

## Expected effect

Transfer stable ranking structure while reducing variance in a smaller learner.

## Falsification condition

The distilled residual regresses trusted full validation or reproduces no change.

## Do not use when

The teacher is not verified or student training would use validation labels.

## Minimal implementation

Use a fixed temperature and loss weight, train only on permitted rows, and add
only a bounded residual to the unchanged parent score.

## Sources

No external source required for the bounded trial.
