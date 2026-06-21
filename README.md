# ReproGym

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
reprogym/        host-side main control (cli, orchestrator, models, pipeline/, sandbox/, verify, trajectory, dataset)
  schema/        claim_spec.schema.json + task_contract.md  ← the heart
prompts/         LLM prompts for deterministic steps (extract_claims, extract_figure_params)
skills/          build-task/ — the one agentic step (authors reward/check.py)
sandboxes/       produced units: claims/ (source of truth) + tasks/ (rendered, ClawGym-compatible)
datasets/        flat symlink views for rollout (build artifact, gitignored)
runs/            interactive run trajectories + outputs (gitignored)
train/           rollout adapter into the ClawGym RL stack
docs/  scratch/  background/design + throwaway probes (gitignored)
```

## Setup

```bash
cp .env.example .env   # fill MinerU / Qwen-VL / Claude keys (local only; never committed or shipped)
```

Three cloud-API backends; the pipeline needs no local model or GPU.
