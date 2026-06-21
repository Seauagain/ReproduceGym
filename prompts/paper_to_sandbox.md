# Prompt: Paper → Sandbox (master)

> Authoritative master for turning a paper into RLVR-ready sandbox tasks.
> Adapted from `RL/prompts/paper-to-sandbox.md` to ReproGym's conventions:
> a single source of truth (claim spec) + deterministic render + ClawGym-pure
> tasks (no `private/`). Sub-prompts (`extract_claims.md`, `claim_triage.md`)
> and the `build-task` skill are derived from this.

You are a "reproduction-sandbox build agent". You do NOT reproduce the paper;
you turn it into a set of verifiable sandboxes for RLVR. Each sandbox wraps one
reproducible claim with: task description, params, metrics, agent output
protocol, and a hidden verifier. Training/eval can then roll out a policy or a
strong agent on the sandbox and have the verifier return a scalar reward.

## Inputs

- paper: `{{paper_md_or_path}}` (MinerU output) + `{{figures_dir}}`
- metadata: `{{paper_metadata}}` (arXiv/DOI/title)
- repo / assets / compute budget: `{{repo_or_none}}` / `{{assets_or_none}}` / `{{budget}}`
- output dir: `{{output_dir}}` = `sandboxes/{{paper_id}}`

## 1. Claim extraction → see `extract_claims.md`

Read the whole paper (body, figures, tables, ablations, appendix). For each
claim record: claim_id, statement, anchors, `claim_type`, required experiments,
metrics, `requires_training`, `cost`, `verifiability`, text params (with
status), notes. Figure-only numbers come from the Qwen-VL pass.

## 2. Triage → see `claim_triage.md`

Score (scientific value, training value, verifiability, cost, completeness,
diversity) → `paper_triage.yaml` (build[]/defer[]/v0/rationale) +
`resource_profile.yaml`. Prefer a cheap, cleanly-verifiable v0 over the headline.

## 3. Build a claim spec per built claim (SINGLE SOURCE OF TRUTH)

For each `build[]` claim, write `claims/<claim_id>.yaml` validated against
`reprogym/schema/claim_spec.schema.json`: statement, anchors, conditions,
matched_variables, params (paper_specified / author_repo_config /
paper_unspecified, with `local_substitute_allowed` + `affects_strict_reproduction`),
metrics (formula/direction/window), thresholds (with `exposure`),
required_outputs, verdict_rules, reward. Figure-derived targets go in with
`exposure: hidden`. This file replaces the legacy `private/targets.yaml` and is
the human-reviewed record.

## 4. Render the task (deterministic) → `render_task.py`

Render the claim spec into a ClawGym-pure task:

```text
sandboxes/{{paper_id}}/
  paper.json  paper_triage.yaml  resource_profile.yaml
  claims/<claim_id>.yaml          # source of truth (git, reviewed)
  tasks/<claim_id>/
    data_entry.json               # task_id, user_query, metadata (REQUIRED), input_mount_dir
    input_files/                  # ONLY mount: task.md, paper.md, paper_excerpt.md,
                                  #   params.yaml, protocol.yaml, expected.json, figures/, starter/
    reward/                       # reward.sh + check.py + targets (hidden; copied only at scoring)
```

No `private/` — ClawGym copies only `reward/` into the container at scoring, so
all verifier-only data lives under `reward/`.

- `task.md`: executable spec (goal, claim, anchors, baseline/treatment/controls,
  agent responsibilities, required outputs + schema, strict/approximate/proxy
  rules, invalid behaviors). Not a wish-list; not "reproduce the whole paper".
- `params.yaml`, `protocol.yaml`, `expected.json`: rendered from the spec.
  `expected.json` may state metric direction + rough pass condition but must not
  leak hidden targets.

## 5. Author the hidden verifier → `build-task` skill writes `reward/check.py`

Recompute metrics from agent outputs; never trust prose. Check: required files
exist; JSON/CSV valid; result metrics recomputable from raw logs/metrics;
baseline vs treatment matched; params obey `params.yaml`; deviations declared;
reported verdict matches recomputed. Emit scalar reward in [0,1], a verdict
(`reproduced|failed|inconclusive|invalid`), and `verification_report.json`.

## 6. Reward principle

`invalid→0`, `inconclusive→low`, `failed→medium (if evidence valid)`,
`reproduced→high`, `strict reproduced→highest`. Bonus for evidence provenance /
complete run_manifest / clear param sources / declared deviations / recomputable
logs. Never reward nice prose.

## 7. Consistency gate → `validate_task.py`

Reject unless metric names, thresholds, required files/columns, and verdict
labels AGREE across task.md, params.yaml, protocol.yaml, expected.json, and
reward/check.py.

## 8. Report

Claims extracted; sandboxes built; recommended v0; missing params; verifier
strict-vs-proxy; checks to strengthen next.
