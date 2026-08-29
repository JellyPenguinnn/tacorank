```json
{"schema_version":"1.0","method_id":"ensemble_confirmed_members","family":"ensemble","status":"candidate","tags":["ensemble","rank_average"],"cost_tier":"low","prerequisites":["two_confirmed_clean_members"],"allowed_data":["verified_predictions"],"prohibition_conditions":["provisional_or_unreproducible_member"],"sources":[]}
```

## Mechanism

Reduce variance by combining complementary trusted rankers.

## Preconditions

Two or three confirmed, clean members are available.

## Allowed data

Only predictions from the exact confirmed member commits.

## Expected effect

Improve stability or small residual headroom.

## Falsification condition

No gain over the best member or incompatible score behavior.

## Do not use when

Any member is provisional, suspicious, or not reproducible.

## Minimal implementation

Rank-average the members while retaining exact member commit identities.

## Sources

No external source required for the bounded trial.
