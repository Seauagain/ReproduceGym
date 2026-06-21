# Prompt: Extract Reproducible Claims (Claude)

> Placeholder. Consumed by `pipeline/extract_claims.py`.

## Goal

From the paper Markdown, extract independently reproducible, falsifiable claims.
For each claim emit (matching the claim spec field set in
`reprogym/schema/claim_spec.schema.json`):

- `statement` — one sentence, falsifiable
- `anchors[]` — section/figure/table/equation refs that support it
- `conditions[]` — conditions to compare + the switches between them
- `matched_variables[]` — what must be held identical across conditions
- `params[]` — params stated in TEXT (value, unit, source, status); leave
  figure-only numbers for the figure-param pass
- proposed `metrics[]` (name, formula over a metrics.csv, direction)
- proposed `required_outputs` and `verdict_rules`

## Output

Strict JSON, one object per claim. No prose outside JSON.

## Notes

- Prefer cheap, claim-level reproductions over whole-paper headline results.
- Do not invent numbers; mark unknown params `paper_unspecified`.
