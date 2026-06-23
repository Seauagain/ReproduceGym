# AGENTS.md — operating manual for agents working in ReproduceGym

Read this first. It exists so you don't re-derive the run flow and the non-obvious
gotchas from source every session. Human-facing run docs live in `README.md`; this
file is the agent cheat-sheet (entrypoints, where things come from, what bites).

## What this repo does

Turns an RL paper into sandbox reproduction tasks, runs an in-sandbox agent against
them on GPU nodes, captures the trajectory, and scores it. Host holds main control +
secrets; GPU nodes are reached only by the in-sandbox agent over plain ssh.

## Three stages: parse -> build -> run

The pipeline is split into three explicit stages, each a thin entrypoint writing
into `runs/<paper_id>/`.

### Stage 0 - parse: `parse_paper.py`

Source (arXiv id/URL, PDF URL, local PDF, or local md) -> `00-parse/{paper.md,
figures/, figures.index.json}` via the MinerU cloud open API. This is what makes
the figures the build's multimodal step needs actually present locally (the old
flow silently skipped multimodal when md image refs pointed at missing files).

```bash
python parse_paper.py --url 2503.20783                 # arXiv id / abs / pdf link, or any PDF URL
python parse_paper.py --pdf paper/dr-grpo.pdf --paper-id dr-grpo
python parse_paper.py --md  paper/dr-grpo.md  --paper-id dr-grpo
```

MinerU creds come from `.env` (`MINERU_TOKEN` / `MINERU_API_KEY`); it is a cloud
service, not self-hosted.

### Stage 1 - build: `build_claim_tasks.py`

Consumes the parse bundle (reuses `00-parse/figures.index.json`), optionally runs
a multimodal vision model with captions/context, extracts/ranks claims, and writes
hash-versioned tasks. No sandbox/GPU.

```bash
python build_claim_tasks.py --paper runs/dr-grpo --out runs --parse-images auto
```

`--paper` accepts a parse bundle dir (`runs/<id>` or its `00-parse/`) or a raw
`paper.md` (figures from a sibling `figures/`). `--parse-images auto` runs image
parsing only when figures exist and `MULTIMODAL_*` / `VISION_*` / legacy `QWEN_*`
are configured; `always` fails if multimodal is unavailable; `never` is text-only.
In `auto`, if a paper has image refs but none resolve locally, build warns (run
parse first).

Outputs:
`runs/<paper>/01-extract/{figures.index.json,figure_evidence.yaml,claims.json}`,
`02-spec/<claim>.<hash>.yaml`, and `03-task/<claim>/<hash>/`.

## Run entrypoint: `run.py`

```bash
python run.py --claim_id <id> --spec-hash <hash> --server <node-alias>
python run.py --task-dir runs/<paper>/03-task/<claim>/<hash> --server <node-alias>
python run.py --claim_id <id> --spec-hash <hash> --server <node> --probe-only
```

It: forces `.env` creds → resolves the built task → estimates runtime → starts an
API-capture proxy → launches the in-sandbox `claude -p` agent on the node → writes
the trajectory. Flags are documented in `README.md` (`### Stage 2 - run`). Output:
`runs/<paper>/04-run/<claim>/<hash>/NNN/{workspace,trajectory}/`.

Do not pass a paper to `run.py`; the run stage never builds or mutates task
definitions.

Run tests with: `python3 -m pytest -q` (pure-python; no GPU/network needed).

## Compute inventory — DO NOT trust stale notes; probe live

- Source of truth for nodes is `../servers.md`. It documents **11 verl nodes** on one
  physical box `106.75.252.110`: 2 as ```yaml``` blocks (`verl-40855`,
  `verl-grpo-44487`) **plus 9 in the `## verl-pool` markdown table** (`verl-1..9`).
- `reproducegym.compute.servers_md` only parses the ```yaml``` blocks, so
  `load_inventory("servers-md:../servers.md")` sees just 2. The full idle set lives in
  **`config/metax_nodes.yaml`** (gitignored) — `run.py --compute` defaults to it.
- The occupancy/GPU notes in `servers.md` go stale fast (other users grab GPUs).
  **Always re-probe before launching:**
  ```bash
  for p in 44487 44852 41463 41398 43821 44486 42188 41238; do
    ssh -o BatchMode=yes -p $p root@106.75.252.110 \
      "mx-smi 2>/dev/null | grep -oE '[0-9]+/65536' | cut -d/ -f1 | awk '{n++;if(\$1>2000)b++}END{print (b+0)\"/\"n}'"
  done
  ```
- **Iron rule on these nodes: only read/write under `/mnt/public/code/hlk/xuzhiqin`.**
- verl/MACA setup (env prelude, metric keys, working GSM8K reference, poll cadence) is
  documented once in `config/metax_nodes.yaml` `notes`/`launch_template` and injected
  into each run's `compute_access.md` (referenced from `task.md`). Don't re-discover it.

## Auth gotcha (this WILL bite you)

The operator's ambient shell exports `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`
(pointing at a *different* relay), and `reproducegym.config.get_env` reads
`os.environ` first, so it shadows `.env` → the agent sends the wrong key to the
gpugeek endpoint → 401. `run.py` fixes this with `force_env_provider()` (loads
`.env` and overwrites `os.environ` for the provider keys). If you call the pipeline
directly, do the same.

## Runtime model — turns are NOT the budget

- The reproduction agent (`claude -p`) is bounded by **wall-clock `--timeout`**, not a
  turn cap. `run.py --max-turns 0` = uncapped (default).
- `--timeout 0` (default) auto-estimates from the claim via `reproducegym/estimate.py`
  (`cost` + `requires_training`): training L=36h, M=18h, S=8h; light tiers are short.
- **Why long runs used to die at the turn cap:** every `tail`/`mx-smi` poll = one turn.
  Babysitting a multi-hour job per turn exhausts turns before training ends. The agent
  is instructed to `setsid nohup` the training and poll sparsely (interval from the
  estimate) + write `output/` + `touch DONE`. Keep it that way.

## Building more claims — do not re-extract per run

Use `build_claim_tasks.py` once per paper, then fan out `run.py` over the built
`claim_id` + `spec_hash` pairs. New claim IDs are deterministic (`c001_slug`,
`c002_slug`, ordered by importance) and task-affecting changes create a new
`spec_hash`. If `run.py` says a claim has multiple versions, pass `--spec-hash`;
do not guess from a stale slug.

## Cleanup — leave nodes clean

verl runs can orphan vLLM workers that hold all 8 GPUs after the driver exits. After a
batch, sweep and kill leftovers (verl containers are single-tenant = yours):
```bash
ssh -p <port> root@106.75.252.110 'pkill -9 -f "VLLM::"; pkill -9 -f "main_ppo"; pkill -9 -f "ray::"'
```
Then confirm `mx-smi` shows 0/8 busy.

## Layout

```
run.py                 single launch entrypoint
config/metax_nodes.yaml  node inventory + injected verl/MACA notes (gitignored)
reproducegym/          host control: cli, orchestrator, models, estimate, metax,
                       runlayout, pipeline/, sandbox/, compute/, schema/
agent_trace/           API-level trajectory capture (proxy + builders + raw/SFT)
runs/<paper>/          01-extract/ 02-spec/ 03-task/<claim>/<hash>/ 04-run/<claim>/<hash>/NNN/  (gitignored)
```
