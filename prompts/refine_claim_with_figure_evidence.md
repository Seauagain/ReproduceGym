# Prompt: Refine One Claim With Figure Evidence

## Role

You receive one claim candidate and structured evidence read from only the
figures anchored by that claim. Fuse them into one stronger claim object.

## Output

Strict JSON object for the refined claim. No prose outside JSON.

## Rules

- Keep the same scientific claim; do not introduce unrelated claims.
- Use figure evidence to fill visible reproduction parameters, condition labels,
  axis ranges, target names, and verifier hints.
- Do not expose hidden target values as visible parameters.
- Preserve all original anchors and add missing figure/table anchors if the
  evidence clearly supports them.
- If figure evidence conflicts with text, keep both in `notes` and mark
  `verifiability` lower.
- Ensure `metrics[]`, `required_experiments`, `params[]`, and `notes` are useful
  for building a standalone reproduction task.
