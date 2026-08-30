```json
{"schema_version":"1.0","method_id":"evaluation_random_exposure_robustness","family":"evaluation","status":"candidate","tags":["random_exposure","unbiased_evaluation","robustness"],"cost_tier":"low","prerequisites":["random_exposure_log","standard_public_evaluation_complete"],"allowed_data":["random_exposure_log","verified_predictions"],"prohibition_conditions":["adaptive_tuning_on_audit_labels"],"implementation_targets":["solution/candidate.py","solution/inference.py"],"sources":["https://arxiv.org/abs/2208.08696","https://github.com/chongminggao/KuaiRand"]}
```

## Mechanism

Evaluate a frozen candidate on the randomly exposed KuaiRand population to
diagnose dependence on the standard logging policy.

## Preconditions

The candidate is frozen, the random log is contract-permitted, schemas align,
and the standard public evaluation is already complete.

## Allowed data

The random-exposure log as an evaluation-only population; no fitting,
preprocessing estimation, or candidate selection on its labels unless the
frozen contract explicitly permits it.

## Expected effect

Expose policy-specific gains or reversals and provide a robustness diagnostic
separate from the primary score.

## Falsification condition

The populations or labels are not comparable, uncertainty is too large, or the
result cannot be reproduced without adapting to the evaluation population.

## Do not use when

It would consume hidden-final feedback, change the official primary metric, or
turn a diagnostic population into an adaptive tuning loop.

## Minimal implementation

Freeze the candidate, validate alignment, compute the contract metrics and
cohorts separately, and record uncertainty and population differences.

## Sources

Gao et al., KuaiRand: An Unbiased Sequential Recommendation Dataset with
Randomly Exposed Videos, plus the official dataset repository.
