# AGENTS.md — operating manual for agents working in ReproduceGym

Read this first. It exists so you don't re-derive the run flow and the non-obvious
gotchas from source every session. Human-facing run docs live in `README.md`; this
file is the agent cheat-sheet (entrypoints, where things come from, what bites).

## What this repo does

Turns an RL paper into sandbox reproduction tasks, runs an in-sandbox agent against
them on GPU nodes, captures the trajectory, and scores it. Host holds main control +
secrets; GPU nodes are reached only by the in-sandbox agent over plain ssh.

## Single entrypoint: `run.py`

```bash
python run.py --claim_id <id> --server <node-alias>          # reproduce a built claim
python run.py --claim_id <id> --server <node> --paper p.md   # build from paper first
python run.py --claim_id <id> --server <node> --probe-only   # resolve + estimate + check node
```

It: forces `.env` creds → resolves/builds the task → estimates runtime → starts an
API-capture proxy → launches the in-sandbox `claude -p` agent on the node → writes
the trajectory. Flags are documented in `README.md` (`## Run a reproduction`). Output:
`runs/<paper>/04-run/<claim>/NNN/{workspace,trajectory}/`.

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

## Building more claims — pipeline is flaky, know this

`orchestrator.build_task` **re-extracts claims on every call and the claim_ids are
non-deterministic** (a later extraction yields different ids than `claims.json`), and
`merge_claim_spec` sometimes emits an invalid `direction` (e.g. `context_dependent`).
So building N specific claims by their old ids fails intermittently. To add claims
reliably: extract **once**, persist, then build each from the saved extraction (and
retry/repair invalid spec fields). The build env also lacks the `anthropic` package —
pass a custom `client` with a `.complete(prompt)->str` method (stdlib `urllib` against
`<base>/v1/messages` with `x-api-key`) instead of installing it.

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
runs/<paper>/          01-extract/ 02-spec/ 03-task/<claim>/ 04-run/<claim>/NNN/  (gitignored)
```
