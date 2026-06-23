# ReproduceGym

Automated RL-literature reproduction as sandbox tasks. A paper becomes
ClawGym-compatible sandbox tasks; each task runs in two modes from the same dir,
unchanged:

- **Interactive reproduction** — host sandbox + reproduction agent, hidden
  verifier scores it, full trajectory recorded.
- **Training rollout** — consumed by `../RL/ClawGym-Agents/RL` rollout; produces
  trajectories + reward to update the policy.

> Status: the end-to-end path runs — build a claim into a task, launch a
> reproduction agent on a GPU node, capture the trajectory, score it. See
> **Run a reproduction** below.

## Architecture (one line)

Main control + sandbox + verifier + secrets all live on the **host**; compute
nodes are only reached by the in-sandbox agent (plain ssh) when it needs GPUs.

## Workflow (see `docs/background.md`)

1. parse PDF → Markdown + figures (MinerU)
2. extract claims (Claude) + figure evidence (multimodal vision model) → ordered `c001_slug`
   claim specs with `spec_hash`
3. render claim spec → sandbox task; `build-task` skill writes the hidden
   `reward/check.py`; consistency gate
4. launch host sandbox (Claude Code agent, key from `.env`)
5. run + record trajectory
6. retry on interruption
7. hidden verifier → scalar reward

## Layout

```
reproducegym/        host-side main control (cli, orchestrator, models, pipeline/, sandbox/, verify, trajectory, runlayout, dataset)
  schema/        claim_spec.schema.json + task_contract.md  ← the heart
agent_trace/     API-level trajectory capture (passthrough proxy + message-level builders + raw/SFT exporters)
prompts/         LLM prompts for deterministic steps (extract_claims, extract_figure_params)
skills/          build-task/ — the one agentic step (authors reward/check.py)
runs/<paper_id>/ per-paper pipeline output, numbered by stage (gitignored; self-describing via README.md + manifest.json):
                 00-parse/  01-extract/  02-spec/  03-task/<claim>/<hash>/  04-run/<claim>/<hash>/NNN/
datasets/        flat symlink views for rollout (build artifact, gitignored, auto-created)
train/           rollout adapter into the ClawGym RL stack
docs/            background/design notes (gitignored)
```

## Setup

```bash
cp .env.example .env   # fill MinerU / multimodal vision / Claude keys (local only; never committed or shipped)
```

Three cloud-API backends; the pipeline needs no local model or GPU.

For GPU nodes, copy the inventory and fill in real connection info:

```bash
cp config/metax_nodes.example.yaml config/metax_nodes.yaml   # gitignored
```

`nodes:` is the inventory; `notes` / `launch_template` / `remote_workdir` are
injected into the agent's `compute_access.md` (the env prelude, the verl launch
recipe, the polling guidance), which `task.md` only *references* — so the task
stays about the science and the agent decides where/how to run.

## Pipeline: three stages

The pipeline is explicitly split into **parse -> build -> run**. Each stage has a
thin entrypoint and writes into the same `runs/<paper_id>/` layout.

### Stage 0 - parse (source -> structured markdown + figures)

`parse_paper.py` turns a source (arXiv id/URL, any PDF URL, a local PDF, or a
local Markdown) into a clean `00-parse/` bundle via the MinerU cloud open API:
`paper.md` + local `figures/` + `figures.index.json`. This is what guarantees the
figures the build's multimodal step needs actually exist locally.

```bash
python parse_paper.py --url 2503.20783                 # arXiv id (or abs/pdf link)
python parse_paper.py --pdf paper/dr-grpo.pdf --paper-id dr-grpo
python parse_paper.py --md  paper/dr-grpo.md  --paper-id dr-grpo   # md must reference local images
```

### Stage 1 - build (parse bundle -> claim tasks)

`build_claim_tasks.py` does **not** launch a sandbox or GPU job. It consumes the
parse bundle, reuses `00-parse/figures.index.json`, optionally runs a multimodal
vision model over the figures, extracts/ranks claims, and renders hash-versioned
task bundles.

```bash
python build_claim_tasks.py --paper runs/dr-grpo --out runs --parse-images auto
```

`--paper` accepts a parse bundle dir (`runs/<id>` or its `00-parse/`) or a raw
`paper.md` (figures resolved from a sibling `figures/`). New claim IDs are ordered
by importance, e.g. `c001_length_bias`, and each task carries a `spec_hash`; if
task-affecting parameters change, the hash changes and the old task/run directory
is not reused.

`--parse-images` supports `auto`, `always`, and `never` (`--解析图片` is also
accepted). In `auto`, image-enhanced parsing runs only when local figures exist
and a multimodal model is configured via `MULTIMODAL_*` / `VISION_*` (or legacy
`QWEN_*`) env vars.

### Stage 2 - run (task -> reproduction attempt)

`run.py` is the run-stage entrypoint. It consumes an already-rendered task, picks
a GPU node, forces the `.env` provider for the agent, launches the in-sandbox
reproduction agent, and records the trajectory (raw stream + API-level capture).
It never reads the paper or rebuilds the task.

```bash
# reproduce one already-built claim on one node (auto-estimates runtime; captures trajectory)
python run.py --claim_id c001_length_bias --spec-hash <hash> --server verl-grpo-44487

# or run directly from a rendered task directory
python run.py --task-dir runs/dr-grpo/03-task/c001_length_bias/<hash> --server verl-grpo-44487
```

Key flags:

| flag | meaning |
|------|---------|
| `--task-dir <dir>` | exact rendered task directory; bypasses claim/hash lookup |
| `--server <alias>` | node alias from the compute inventory (required) |
| `--spec-hash <hash>` | exact task/spec version; required when a claim has multiple task hashes |
| `--compute <spec>` | inventory: a path (`.yaml`/`.json`/`.md`) or scheme (`servers-md:...`, `lbg:...`). Default `config/metax_nodes.yaml` |
| `--timeout 0` | `0` = auto-estimate wall-clock from the claim (`cost`+`requires_training`; training can be **>24h**); pass seconds to force a budget |
| `--max-turns 0` | `0` = uncapped — bounded only by `--timeout`. The agent is told to background training and poll sparsely, so a long job is not killed by a turn cap |
| `--run-dir <dir>` | explicit attempt dir; pre-assign when fanning out in parallel to avoid `NNN` collisions |
| `--no-capture` | skip the API-level trajectory proxy |
| `--probe-only` | resolve + estimate + check the node, then exit (no launch) |

Output lands in `runs/<paper>/04-run/<claim>/<hash>/NNN/`:

- `workspace/` — agent working dir, incl. `output/result.json` + `output/metrics.csv`
- `trajectory/` — raw `trajectory.jsonl`, merged `trajectory.merged.json`, `sft.jsonl`, and `captures/`

### Parallel / best-of-N

Fan out by launching one `run.py` per `(claim, node)` with a **pre-assigned**
`--run-dir` (so concurrent attempts don't race for the same `NNN`):

```bash
declare -A MAP=( [verl-1-44852]=004 [verl-6-44486]=005 [verl-7-42188]=006 )
for nd in "${!MAP[@]}"; do
  n=${MAP[$nd]}
  python run.py --claim_id c001_length_bias --spec-hash "$HASH" --server "$nd" \
    --run-dir runs/dr-grpo/04-run/c001_length_bias/$HASH/$n \
    > runs/dr-grpo/_dispatch/c001_length_bias@$nd.log 2>&1 &
done
```
