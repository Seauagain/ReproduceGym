# Task Render Contract

How a canonical **claim spec** (`sandboxes/<paper>/claims/<claim_id>.yaml`,
validated against `claim_spec.schema.json`) becomes a **ClawGym-compatible
sandbox task** (`sandboxes/<paper>/tasks/<claim_id>/`).

Rule of thumb: **one source of truth (the claim spec), deterministic render,
one hand-authored file (`reward/check.py`), one consistency gate.**

## 1. Output layout (must satisfy the ClawGym rollout contract)

```
tasks/<claim_id>/
├── data_entry.json     # task_id, user_query, metadata (REQUIRED by ClawGym), input_mount_dir
├── input_files/        # the ONLY thing mounted into the agent workspace
│   ├── task.md
│   ├── params.yaml
│   ├── protocol.yaml
│   ├── expected.json
│   ├── paper.md / paper_excerpt.md   # if exposure_policy exposes them
│   ├── figures/        # public figures (optional)
│   └── starter/
└── reward/             # verifier-only; copied into the container ONLY at scoring time
    ├── reward.sh       # entry: prints a single float on the last stdout line
    ├── check.py        # hand/agent-authored recompute + scoring
    └── targets.*       # hidden thresholds/targets (when exposure: hidden)
```

ClawGym hard constraints (see `clawgym_rl_rollout.py`):
- task dir MUST contain `data_entry.json`, and `data_entry.json` MUST have `metadata`.
- if `input_mount_dir` is set, `input_files/` MUST exist.
- `reward/reward.sh` MUST exist; its last stdout line MUST be a float.
- `_discover_task_entries` scans only ONE level → the rollout `source_path`
  must be a flat dir of task dirs. Nested `<paper>/tasks/<claim>` is flattened
  into `datasets/<name>/` (symlinks) by `dataset.py`.
- Scoring copies ONLY `reward/`. Anything the verifier needs at scoring time
  (hidden targets) MUST live under `reward/`, NOT a sibling `private/`.

## 2. claim spec field → rendered artifact (single source of truth)

| claim spec field      | renders into                                                        |
|-----------------------|---------------------------------------------------------------------|
| statement, anchors    | task.md §1–2                                                         |
| conditions, matched_variables | task.md §3, params.yaml `algorithm_switches`                |
| params[]              | params.yaml (`paper_specified`/`author_repo_config`/`paper_unspecified`), task.md §4 |
| metrics[]             | task.md §7/§11, protocol `metric_computation`, expected.json, check.py |
| thresholds[]          | expected.json (if `exposure: visible`) else reward/targets; check.py constants |
| required_outputs      | task.md §6/§8–10, protocol `agent_must_write`, check.py REQUIRED_*    |
| verdict_rules         | task.md §11, protocol `verdict_rules`, check.py                       |
| reward                | reward/check.py scoring (always hidden)                               |
| input_files           | extra agent-visible assets copied into input_files/                   |

## 3. Exposure routing

- Each leaf's `exposure` (with `exposure_policy` as default) decides
  visible (`input_files/`) vs hidden (`reward/`).
- Thresholds default to `hidden`. `v0_full_paper_public` may expose them via
  `expected.json`; stricter tiers keep them only in `reward/`.
- Never write a hidden value into any `input_files/` artifact.

## 4. Consistency gate (`validate_task.py`)

Render is rejected unless ALL of these agree across every produced file +
`reward/check.py`:
- metric names (task.md, protocol, expected.json, check.py)
- threshold values + which metric they bind to
- required output files and metrics.csv columns
- verdict label set

This is the gate that stops the 4-way duplication in the real dr-grpo task
from silently drifting.

## 5. What is NOT auto-rendered

`reward/check.py` is authored by the `build-task` skill, because writing a
correct recompute+scoring routine is genuinely agentic (generate → run on a
synthetic/sample submission → self-check thresholds). The renderer supplies
its constants (metric names, thresholds, required files) from the claim spec;
the skill fills in the recompute logic, then `validate_task.py` checks the two
stayed consistent.
