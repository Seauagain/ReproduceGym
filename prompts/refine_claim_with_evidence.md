You refine one paper claim into a verifiable reproduction-task contract.

Return strict JSON only: one claim object. Do not wrap in Markdown.

Separate the complex reproduction workflow from the narrow verifier:

- `reproduction_protocol`: natural-language/structured guidance for what the agent should run. It may mention training, API calls, inference, evaluation scripts, datasets, expected outputs, logs, and caveats.
- `verification_contract`: structured reward contract. It must fit one of:
  - `numeric_threshold`
  - `directional_comparison`
  - `table_or_curve_point`
  - `artifact_metric`

Required top-level fields:

- `statement`
- `claim_type`
- `reproduction_protocol`
- `verification_contract`
- `evidence_anchors`
- optional `likely_pool`

`verification_contract` must include these keys even when empty:

- `type`
- `conditions`: list of `{label, description, switches?}`
- `metrics`: list of `{name, formula, direction, window?}`
- `params`: list of parameter objects. Use `use:"target"` only for paper-grounded target values and include source/read_from/confidence when possible.
- `thresholds`: list of `{metric, pass_threshold, exposure, source?, rationale?, target_value?, tolerance_abs?, target_evidence?}` when a numeric paper target is explicit.
- `verdict_rules`: object with reproduced/failed/inconclusive/invalid rule lists when known.

For directional claims, prefer an executable cross-condition metric, e.g.
`mean(method_a.pass1) - mean(method_b.pass1)` or
`mean(method_a.length) / mean(method_b.length)`. Do not invent paper numeric
targets. If the evidence is insufficient to define a verifier, keep metrics or
thresholds empty and set `likely_pool:"exploration"`.

The generated verifier supports ONLY this formula grammar:

- Aggregations over CSV columns: `mean(col)`, `sum(col)`, `min(col)`, `max(col)`, `median(col)`, `std(col)`, `var(col)`, `last(col)`, `first(col)`, `count(col)`.
- Condition-specific columns: `mean(condition_label.column_name)`.
- Arithmetic over aggregated scalars: `+`, `-`, `*`, `/`, `%`, `**`, numeric constants, and `abs(...)`.

Never write prose formulas such as `num_correct / num_total * 100 on AIME 2024`,
`mean_response_length(late_training)`, or `spearman_correlation(...)`. Instead,
define output CSV columns that make the formula executable, for example:

- `mean(AIME_2024.pass1)`
- `mean(late.mean_response_length) / mean(early.mean_response_length)`
- `mean(PPO.truthful_informative_rate) / mean(GPT3.truthful_informative_rate)`
