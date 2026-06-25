You extract globally important, falsifiable reproduction claims from one RL/ML paper.

Return strict JSON only: a list of 5-8 claim objects. Do not wrap in Markdown.

Each object must include:

- `statement`: one falsifiable claim.
- `claim_type`: one of `eval_only`, `mechanism`, `ablation`, `scaling`, `headline`, `diagnostic`.
- `display_title`: short human-readable title.
- `importance_rank`: integer, 1 = most important.
- `cost`: one of `S`, `M`, `L`, `XL`.
- `verifiability`: one of `high`, `medium`, `low`.
- `requires_training`: boolean.
- `anchors`: list of `{kind, ref, note}` where kind is `section`, `figure`, `table`, `equation`, or `appendix`.
- `evidence_anchors`: same shape as `anchors`; include the evidence needed by verifier refinement.
- `risks`: list of strings.
- `likely_pool`: `rlvr` only when a metric/target or directional comparison appears likely; otherwise `exploration`.

Do not output final `metrics`, `thresholds`, `verdict_rules`, or hidden targets here. This pass is only for global understanding and candidate selection. Claim-scoped refinement will build verifier contracts later.
