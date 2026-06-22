# Prompt: Extract Reproducible Claims (Claude)

> Consumed by `pipeline/extract_claims.py`. Derived from `paper_to_sandbox.md` step 1.
> Output feeds the canonical claim spec (`reproducegym/schema/claim_spec.schema.json`).

## Role

You convert a paper into a set of independently reproducible claims. You are NOT
reproducing the paper. Read the whole paper — body, figures, tables, ablations,
appendix, experimental setup — not just the abstract's headline.

## For each claim, extract

- `claim_id` — stable, `[a-z0-9_]+`, unique within the paper
- `statement` — claim verbatim or a faithful paraphrase, one falsifiable sentence
- `anchors[]` — `{kind: section|figure|table|equation|appendix, ref, note}`
- `claim_type` — `eval_only | mechanism | ablation | scaling | headline | diagnostic`
- `required_experiments` — what must be run to test it (conditions: baseline /
  treatment / control variables)
- `metrics[]` — `{name, formula (recompute rule over a metrics file), direction}`
- `requires_training` — boolean (training run needed vs eval-only)
- `cost` — `S | M | L | XL` (compute/time/engineering)
- `verifiability` — `high | medium | low` (can a program recompute the metric?)
- `params[]` (text-stated only) — `{name, value, unit, source, status}` where
  `status ∈ paper_specified | author_repo_config | paper_unspecified`; leave
  figure-only numbers for the Qwen-VL figure pass
- `notes` — missing params, missing data, risks

## Output

Strict JSON, a list of claim objects. No prose outside JSON.

## Rules

- Prefer cheap, claim-level reproductions that yield a clean verifier over the
  expensive headline result.
- Never invent numbers. If the paper doesn't state it, mark `paper_unspecified`.
- Favour claims where a metric can be recomputed from agent-produced logs.
