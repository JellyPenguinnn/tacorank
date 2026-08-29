```json
{"schema_version":"1.0","method_id":"multitask_single_auxiliary","family":"multitask","status":"candidate","tags":["multitask","auxiliary"],"cost_tier":"medium","sources":[]}
```

## Mechanism

Use related engagement supervision to regularize long-view prediction.

## Preconditions

One legal auxiliary label is available.

## Allowed data

The primary target plus one contract-permitted auxiliary signal.

## Expected effect

Improve generalization of the primary ranking head.

## Falsification condition

The auxiliary task degrades primary validation or violates the contract.

## Do not use when

The auxiliary label is not explicitly permitted.

## Minimal implementation

Begin with one auxiliary signal and a fixed documented loss weight.

## Sources

No external source required for the bounded trial.
