# Prompt: Extract Reproducible Claims (Claude)

> Consumed by `pipeline/extract_claims.py`. Derived from `paper_to_sandbox.md` step 1.
> Output feeds the canonical claim spec (`reproducegym/schema/claim_spec.schema.json`).

## Role

You convert a paper into a set of independently reproducible claims. You are NOT
reproducing the paper. Read the whole paper — body, figures, tables, ablations,
appendix, experimental setup — not just the abstract's headline.

## For each claim, extract

- `claim_id` — provisional short slug only; downstream code rewrites it to
  `c001_short_slug`
- `importance_rank` — integer, 1 = most important / best first task
- `display_title` — short human-readable title for reports and manifests
- `statement` — claim verbatim or a faithful paraphrase, one falsifiable sentence
- `anchors[]` — `{kind: section|figure|table|equation|appendix, ref, note}`
- `claim_type` — `eval_only | mechanism | ablation | scaling | headline | diagnostic`
- `required_experiments` — what must be run to test it (conditions: baseline /
  treatment / control variables)
- `metrics[]` — `{name, formula (recompute rule over a metrics file), direction}`
- `thresholds[]` — paper-stated pass targets only, when a target can be bound to a
  metric: `{metric, pass_threshold, exposure, rationale, source, confidence,
  tolerance, target_evidence}`. Use `exposure:"hidden"` for answer-key values.
  `target_evidence` must identify the audit anchor: source table/figure/section,
  any read-from note, and confidence.
- `verdict_rules` — concise rules mapping recomputed metrics to
  `reproduced|failed|inconclusive|invalid`. If the paper gives only a directional
  claim, express it as a metric formula plus a threshold when possible
  (for example `zero_shot_pass1 - few_shot_pass1 >= 0`).
- `verification` — `{mode, pool}` where `mode ∈ numeric_threshold | directional |
  structural | unverifiable` and `pool ∈ rlvr | exploration`. Use `rlvr` only when
  the claim has an executable metric/threshold contract; otherwise use
  `exploration`.
- `requires_training` — boolean (training run needed vs eval-only)
- `cost` — `S | M | L | XL` (compute/time/engineering)
- `verifiability` — `high | medium | low` (can a program recompute the metric?)
- `params[]` (text-stated only) — `{name, value, unit, source, status}` where
  `status ∈ paper_specified | author_repo_config | paper_unspecified`; leave
  figure-only numbers for the multimodal figure pass. If a text-stated value is
  verifier answer-key material, set `use:"target"`, `exposure:"hidden"`, and bind
  it to `metric`. A target param must include audit evidence via `source`,
  `read_from` when applicable, and `confidence`.
- `notes` — missing params, missing data, risks

## Output

Strict JSON, a list of claim objects. No prose outside JSON.

## Rules

- Prefer cheap, claim-level reproductions that yield a clean verifier over the
  expensive headline result.
- Never invent numbers. If the paper doesn't state it, mark `paper_unspecified`.
- Never emit an answer-key target without paper evidence. The verifier target must
  be auditable back to a figure/table/section and enough local context for a human
  to re-read it.
- Favour claims where a metric can be recomputed from agent-produced logs.
- Use verifier-safe snake_case identifiers in `conditions[].label`,
  `metrics[].name`, and `metrics[].formula`: start with a letter or underscore and
  use only letters, digits, and underscores. Do not emit labels like
  `Oat-Zero-7B`, `4shot`, or `clip_0.2` inside formulas; write
  `oat_zero_7b`, `c_4shot`, or `clip_0_2` instead.
- Do not mark a claim `verification.pool="rlvr"` unless every primary metric has
  an executable threshold or directional rule. A qualitative claim without such a
  rule belongs in `exploration`.
- If a claim depends on a figure, include a `figure` anchor with the exact ref
  from the figure inventory. Do not rely only on section text.
- Use the figure inventory to notice figure-only experimental requirements, but
  leave exact numeric reads from images to the multimodal figure pass.
