```json
{"schema_version":"1.0","method_id":"model_stacked_cross_residual","family":"model","status":"known_negative","tags":["stacked","feature_interaction","cross_network","residual"],"cost_tier":"high","prerequisites":["baseline_parity","objective_data_frame_verified"],"allowed_data":["train_interactions","user_id","video_id","author_id","tab","duration_ms","date","long_view","verified_predictions"],"prohibition_conditions":["evaluator_or_split_change_required"],"sources":["https://arxiv.org/abs/2008.13535","https://dl.acm.org/doi/10.1145/3442381.3450078"]}
```

## Mechanism

Learn a second-stage residual with explicit bounded-degree feature crossing,
stacked on the frozen parent rather than replacing it.

The parent is a factorisation machine, so it represents second-order feature
interactions and nothing beyond. DCN V2 (Wang et al., Google, WWW 2021) exists
because that ceiling is real at ranking scale: its cross layers learn explicit
higher-degree interactions that a second-order model cannot express, and the
paper reports gains across Google's web-scale learning-to-rank systems.

The stacked arrangement is the one DCN V2 itself describes, cross layers
followed by a deep component, and it fits this contract without modification
because the parent stays and only the residual is learned.

## Preconditions

Executable FM parity is verified. This is the one card whose premise is that
the parent's *hypothesis class* is the limit, so it should follow evidence that
same-order mechanisms have stopped paying, not precede it.

## Allowed data

Contract-permitted fields only, with all aggregates computed strictly from
training rows preceding every scored date.

## Expected effect

Improve within-user ordering by representing interactions the parent cannot,
rather than re-fitting interactions it already has. This is the only card in
the portfolio whose stated source of gain is a larger hypothesis class.

## Falsification condition

No trusted full-fidelity improvement over the parent at matched budget, or the
residual improves training fit while within-user ordering is unchanged, which
indicates it is re-learning the parent's own signal with more parameters.

## Do not use when

Retired on measured evidence: sibling DCNv2 attempts failed in coding twice under this budget; capacity-class mechanisms measured below small trees on this data.

The budget cannot train it honestly at every fidelity. The fidelity views share
identical training data and differ only in scored population, so a model that
is only trained properly at full fidelity will be pruned at proxy for a
handicap it imposed on itself.

## Minimal implementation

Learn one bounded additive residual from a small stack of cross layers over the
permitted fields, add it to the frozen FM parent on the original score scale,
and keep the objective, evaluator, split, and population fixed. Prefer few
layers and a narrow width: the mechanism under test is explicit feature
crossing, not capacity.

## Sources

Wang et al., "DCN V2: Improved Deep & Cross Network and Practical Lessons for
Web-scale Learning to Rank Systems", WWW 2021.
