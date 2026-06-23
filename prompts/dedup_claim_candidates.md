# Prompt: Deduplicate And Rank Claim Candidates

## Role

You merge claim candidates extracted from separate paper chunks into a final
non-redundant list. Preserve scientific content and reproducibility details.

## Output

Strict JSON list of claim objects. No prose outside JSON.

## Rules

- Merge candidates that test the same scientific claim, even if phrased differently.
- Preserve all useful anchors, parameters, metrics, implementation notes, and risks.
- Prefer the clearer/falsifiable statement when merging duplicates.
- Rank by importance and practical reproducibility:
  1. central paper claim,
  2. clear metric/verifier,
  3. feasible compute,
  4. enough stated parameters.
- Keep enough claims to cover the paper's main mechanisms, ablations, diagnostics,
  scaling results, and headline results.
- Do not invent metrics or numbers.
