---
name: build-task
description: Author the hidden verifier (reward/check.py) for a ReproGym sandbox task from a canonical claim spec. Use after render_task has produced the visible task files; this skill writes the one non-deterministic artifact and self-checks it against the claim spec.
---

# build-task

> Placeholder skill. The ONLY agentic step in the task factory.

Everything else (task.md, params.yaml, protocol.yaml, expected.json,
data_entry.json) is deterministically rendered from the claim spec by
`render_task.py`. This skill writes `reward/check.py` — the recompute + scoring
logic — because doing it correctly needs an agent loop, not a single prompt.

## Inputs

- `sandboxes/<paper>/claims/<claim_id>.yaml` (claim spec, single source of truth)
- the rendered `tasks/<claim_id>/` (for the constants to match)
- `reprogym/schema/task_contract.md` (the contract + consistency rules)
- `prompts/paper_to_sandbox.md` (master; this skill is its steps 5–6)

## Procedure (to implement)

1. Read the claim spec: metrics[].formula, thresholds, required_outputs,
   verdict_rules, reward.base_by_verdict.
2. Write `reward/check.py` that:
   - validates required files exist and are well-formed,
   - recomputes each metric from output/metrics.csv via its formula,
   - checks reported result.json is consistent with recomputed values,
   - assigns a verdict and a scalar reward in [0, 1].
3. Write `reward/reward.sh` (entry; last stdout line = the float).
4. Self-check: run check.py on a synthetic pass submission and a synthetic fail
   submission; confirm thresholds/verdicts behave.
5. Hand back to `validate_task.py` for the cross-file consistency gate.

## Hard rules

- Hidden targets live under reward/ only (ClawGym copies just reward/ at scoring).
- Never read a value the agent could have copied; recompute from submitted logs.
