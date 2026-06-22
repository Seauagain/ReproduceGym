# ReproduceGym

Automated RL-literature reproduction as sandbox tasks. A paper becomes
ClawGym-compatible sandbox tasks; each task runs in two modes from the same dir,
unchanged:

- **Interactive reproduction** — host sandbox + reproduction agent, hidden
  verifier scores it, full trajectory recorded.
- **Training rollout** — consumed by `../RL/ClawGym-Agents/RL` rollout; produces
  trajectories + reward to update the policy.

> Scaffold stage: directories, contracts and stubs only — no logic yet.

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
