# Prompt: Claim Triage (Claude)

> Consumed by `pipeline/triage.py`. Derived from `paper_to_sandbox.md` step 2.
> Decides which extracted claims become sandbox tasks and which is the v0.

## Input

The list of claims from `extract_claims` (+ figure params if available).

## Score each claim on

- scientific value — is it a core contribution?
- training value — does building it train the agent's reproduction ability?
- verifiability — can a program recompute the metric? (use claim.verifiability)
- cost — is the compute/time/data/engineering acceptable? (use claim.cost)
- information completeness — do paper/author repo give enough params?
- diversity — does it add a new task type vs. repeating one experiment?

## Output: `paper_triage.yaml`

- `build[]` — claim_ids to turn into sandbox tasks
- `defer[]` — claim_ids to postpone, each with a reason
- `v0` — the single recommended first sandbox (cheap + clean verifier)
- `rationale` — notably, why a costly headline claim is NOT v0

## Rules

- Prefer a cheap, cleanly-verifiable claim as v0 over the headline result.
- Favour a diverse `build[]` set over many variants of one experiment.
