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
2. extract claims (Claude) + figure params (Qwen-VL) → merge into a **claim spec**
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
                 01-extract/  02-spec/  03-task/<claim>/  04-run/<claim>/NNN/
datasets/        flat symlink views for rollout (build artifact, gitignored, auto-created)
train/           rollout adapter into the ClawGym RL stack
docs/            background/design notes (gitignored)
```

## Setup

```bash
cp .env.example .env   # fill MinerU / Qwen-VL / Claude keys (local only; never committed or shipped)
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

## Run a reproduction

`run.py` is the single entrypoint. It resolves (or builds) a claim's task, picks a
GPU node, forces the `.env` provider for the agent, launches the in-sandbox
reproduction agent, and records the trajectory (raw stream + API-level capture).

```bash
# reproduce one already-built claim on one node (auto-estimates runtime; captures trajectory)
python run.py --claim_id dr_grpo_len --server verl-grpo-44487

# build the claim from a paper first, if it isn't rendered yet
python run.py --claim_id <id> --server <node> --paper path/to/paper.md
```

Key flags:

| flag | meaning |
|------|---------|
| `--server <alias>` | node alias from the compute inventory (required) |
| `--compute <spec>` | inventory: a path (`.yaml`/`.json`/`.md`) or scheme (`servers-md:...`, `lbg:...`). Default `config/metax_nodes.yaml` |
| `--timeout 0` | `0` = auto-estimate wall-clock from the claim (`cost`+`requires_training`; training can be **>24h**); pass seconds to force a budget |
| `--max-turns 0` | `0` = uncapped — bounded only by `--timeout`. The agent is told to background training and poll sparsely, so a long job is not killed by a turn cap |
| `--run-dir <dir>` | explicit attempt dir; pre-assign when fanning out in parallel to avoid `NNN` collisions |
| `--no-capture` | skip the API-level trajectory proxy |
| `--probe-only` | resolve + estimate + check the node, then exit (no launch) |

Output lands in `runs/<paper>/04-run/<claim>/NNN/`:

- `workspace/` — agent working dir, incl. `output/result.json` + `output/metrics.csv`
- `trajectory/` — raw `trajectory.jsonl`, merged `trajectory.merged.json`, `sft.jsonl`, and `captures/`

### Parallel / best-of-N

Fan out by launching one `run.py` per `(claim, node)` with a **pre-assigned**
`--run-dir` (so concurrent attempts don't race for the same `NNN`):

```bash
declare -A MAP=( [verl-1-44852]=004 [verl-6-44486]=005 [verl-7-42188]=006 )
for nd in "${!MAP[@]}"; do
  n=${MAP[$nd]}
  python run.py --claim_id dr_grpo_len --server "$nd" \
    --run-dir runs/dr-grpo/04-run/dr_grpo_len/$n \
    > runs/dr-grpo/_dispatch/dr_grpo_len@$nd.log 2>&1 &
done
```
