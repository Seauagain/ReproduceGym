# ReproduceGym

ReproduceGym turns RL/ML papers into sandbox reproduction tasks with verifiable,
metric-based rewards. It is designed for the workflow where an agent reads a
paper, builds claim-level reproduction tasks, runs those tasks in an isolated
sandbox, and scores the result with a hidden verifier.

The current pipeline is built around RLVR-style tasks:

```text
RLVR task = paper claim + recomputable metrics + paper-grounded targets + reward curves
```

Instead of treating a paper as one giant reproduction, ReproduceGym extracts the
paper's scientific claims, binds each claim to its supporting evidence, compiles
a verifier contract, renders ClawGym-compatible tasks, and runs an in-sandbox
reproduction agent on a chosen compute node.

## Why This Exists

Older task builds could produce runnable-looking tasks without explicit targets,
with directional thresholds such as `metric > 0`, or with rewards tied to verdict
labels. Those tasks were weak RLVR targets: a partial or failed reproduction
could still receive a misleading reward.

This version makes the contract stricter:

- Claims come from the paper text, not from reverse-engineering figures.
- Figures and tables are evidence for claims and sources for targets.
- Accepted RLVR tasks must have paper-grounded numeric `target_value` fields.
- Reward is computed from metric values and continuous reward curves, not from
  verdict strings.
- Metrics without grounded targets stay as diagnostics or go to `exploration`;
  they are not exposed as accepted RLVR tasks.

## Quick Start

```bash
# 1. Parse a paper into Markdown + local figures.
python parse_paper.py --url 2503.20783 --paper-id 2503-dr-grpo

# 2. Build claim-level RLVR tasks from the parse bundle.
python build_claim_tasks.py \
  --paper runs/2503-dr-grpo \
  --paper-id 2026-06-25-dr-grpo \
  --out runs \
  --parse-images auto \
  --non-strict-vl \
  --max-claims 5

# 3. Inspect the exact task dirs downstream agents should consume.
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("runs/2026-06-25-dr-grpo/task_manifest.json").read_text())
for task in manifest["tasks"]:
    print(task["claim_id"], task["spec_hash"], task["task_dir"])
PY

# 4. Run one rendered task on a compute node.
python run.py \
  --claim_id c001_dr_grpo_reduces_response_length \
  --spec-hash <spec_hash> \
  --server <node-alias>
```

## Architecture

Host-side code controls parsing, building, sandbox launch, secrets, verifier
rendering, and trajectory capture. GPU nodes are only accessed by the
in-sandbox reproduction agent when a task needs remote compute.

```text
paper source
  -> parse_paper.py
  -> runs/<paper_id>/00-parse/
  -> build_claim_tasks.py
  -> runs/<paper_id>/{01-extract,02-spec,03-task}
  -> run.py
  -> runs/<paper_id>/04-run/<claim>/<hash>/<attempt>/
```

Important directories:

```text
prompts/                 LLM/VL prompts for claim and evidence extraction.
reproducegym/pipeline/   Parse, build, render, and validation pipeline.
reproducegym/schema/     Canonical claim spec schema.
reproducegym/verifier/   Reward recomputation engine.
agent_trace/             API-level trajectory capture.
config/                  Compute inventory examples and local inventories.
runs/<paper_id>/         Generated parse/build/run artifacts.
docs/                    Design notes and known gaps.
```

## Setup

Create a local environment file:

```bash
cp .env.example .env
```

Fill in the providers you use:

- MinerU credentials for PDF parsing.
- Anthropic-compatible credentials for text claim extraction and refinement.
- Optional multimodal vision provider for figure target extraction.

For remote compute, create a local node inventory:

```bash
cp config/metax_nodes.example.yaml config/metax_nodes.yaml
```

`config/metax_nodes.yaml` is gitignored and is the default inventory for
`run.py --compute`.

Run tests:

```bash
python -m pytest -q
```

## Pipeline Stages

### Stage 0: Parse

`parse_paper.py` converts an arXiv id, PDF URL, local PDF, or local Markdown into
a parse bundle. The build stage consumes this bundle and reuses its local figure
index, so image references must resolve locally.

```text
runs/<paper_id>/00-parse/
  paper.md
  figures/
  figures.index.json
```

Examples:

```bash
python parse_paper.py --url 2503.20783
python parse_paper.py --url https://arxiv.org/abs/2503.20783
python parse_paper.py --pdf paper/dr-grpo.pdf --paper-id 2503-dr-grpo
python parse_paper.py --md paper/dr-grpo.md --paper-id 2503-dr-grpo
```

Use `--paper-id` when you want a stable run directory name. If omitted, the
pipeline derives one from the source.

### Stage 1: Build Claims And Tasks

`build_claim_tasks.py` consumes a parse bundle and produces claim-level tasks.
It does not launch a sandbox or GPU job. It may call text and vision model APIs
depending on cache state and `--parse-images`.

```bash
python build_claim_tasks.py \
  --paper runs/2503-dr-grpo \
  --paper-id 2026-06-25-dr-grpo \
  --out runs \
  --parse-images auto \
  --max-claims 5
```

Useful flags:

| flag | meaning |
|------|---------|
| `--paper` | parse bundle dir, `00-parse/` dir, or raw `paper.md` |
| `--paper-id` | output id under `runs/` |
| `--out` | runs root, default `runs/` |
| `--max-claims N` | render top N selected RLVR claims; `0` means all |
| `--claim-id ID` | build only a specific claim; repeatable |
| `--refresh-claims` | ignore cached extraction/refinement and call models again |
| `--parse-images auto` | use VL only if figures and provider are available |
| `--parse-images always` | require VL and local figures |
| `--parse-images never` | text/table-only build |
| `--non-strict-vl` | skip malformed VL JSON instead of failing the build |

Build output:

```text
runs/<paper_id>/
  01-extract/
    paper_evidence_index.json
    claim_candidates.json
    triaged_claims.json
    claim_evidence/<claim_uid>.json
    refined_claims.json
    claim_verification_report.json
    selected_claims_for_build.json
  02-spec/
    c001_slug.<spec_hash>.yaml
  03-task/
    c001_slug/<spec_hash>/
      task.md
      data_entry.json
      input_files/
      reward/check.py
  build_validation.json
  task_manifest.json
  token_usage.jsonl
  token_usage.summary.json
  CLAIMS.md
```

Downstream consumers should read `task_manifest.json`. Do not guess the latest
hash directory from `03-task/`.

### Stage 2: Run A Reproduction

`run.py` executes one already-rendered task. It never rebuilds tasks and never
reads the paper. It resolves a task, probes the selected node, launches the
in-sandbox reproduction agent, and records the resulting trajectory.

```bash
python run.py \
  --claim_id c001_dr_grpo_reduces_response_length \
  --spec-hash 75568a2222bb \
  --server verl-grpo-44487
```

Or bypass lookup with an exact task directory:

```bash
python run.py \
  --task-dir runs/2026-06-25-dr-grpo/03-task/c001_dr_grpo_reduces_response_length/75568a2222bb \
  --server verl-grpo-44487
```

Useful flags:

| flag | meaning |
|------|---------|
| `--task-dir` | exact rendered task directory |
| `--claim_id` | claim id to resolve under `runs/*/03-task` |
| `--spec-hash` | exact spec/task version; required for ambiguous claims |
| `--server` | compute node alias from the inventory |
| `--compute` | inventory path or scheme; default `config/metax_nodes.yaml` |
| `--timeout 0` | auto-estimate wall-clock budget; pass seconds to override |
| `--max-turns 0` | uncapped turns; wall-clock timeout is the real budget |
| `--run-dir` | explicit attempt directory for parallel dispatch |
| `--probe-only` | resolve task and node, then exit |
| `--no-capture` | skip API-level trajectory capture |

Run output:

```text
runs/<paper_id>/04-run/<claim>/<hash>/<attempt>/
  workspace/
    output/result.json
    output/metrics.csv
  trajectory/
    trajectory.jsonl
    trajectory.merged.json
    sft.jsonl
    captures/
```

## How RLVR Task Selection Works

The build pipeline is claim-first:

1. Extract candidate claims from the whole paper text.
2. Rank/triage claims by importance, quantifiability, reproducibility, and cost.
3. Build claim-scoped evidence bundles from relevant paper slices, tables,
   captions, and figures.
4. Refine each claim into metrics, params, thresholds, and reproduction protocol.
5. Run deterministic contract synthesis:
   - normalize verifier-safe identifiers;
   - bind paper targets to metrics;
   - synthesize thresholds and reward curves;
   - move ungrounded metrics to diagnostics;
   - route tasks to `rlvr` or `exploration`.
6. Validate accepted tasks with schema checks, formula checks, target/reward
   checks, leak scans, hash consistency, and synthetic reward selftests.

Only accepted `rlvr` tasks are written to `task_manifest.json`. Rejected or
partial claims are still preserved in `01-extract/claim_verification_report.json`
with the reason they were routed to `exploration`.

## Example: Build Dr-GRPO Tasks

Parse:

```bash
python parse_paper.py --url 2503.20783 --paper-id 2503-dr-grpo
```

Build up to five RLVR tasks:

```bash
python build_claim_tasks.py \
  --paper runs/2503-dr-grpo \
  --paper-id 2026-06-25-dr-grpo \
  --out runs \
  --parse-images auto \
  --non-strict-vl \
  --max-claims 5
```

Inspect selected tasks:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("runs/2026-06-25-dr-grpo/task_manifest.json").read_text())
for task in manifest["tasks"]:
    print(task["claim_id"], task["spec_hash"], task["task_dir"])
PY
```

Run one task:

```bash
python run.py \
  --claim_id c001_dr_grpo_reduces_response_length \
  --spec-hash <spec_hash> \
  --server <node-alias>
```

## Known Gaps

- Derived targets for ablation dominance and causal contrast are still limited.
  See `docs/derived-target-contract-gaps.md`.
- Visual curve targets depend on VL estimates and therefore use conservative
  tolerances.
- Some important claims remain in `exploration` when the paper lacks a numeric
  target, the figure read is ambiguous, or the reproduction parameters are
  incomplete.

## Development Notes

- The build stage can run without local GPUs; it uses API providers.
- The run stage may need remote GPUs depending on the task.
- `run.py` forces provider credentials from `.env` to avoid ambient shell
  variables pointing at the wrong relay.
- Generated `runs/` artifacts are gitignored.
- Keep scratch scripts out of commits unless they are promoted into tests or
  documented utilities.
