#!/usr/bin/env python3
"""run.py - run-stage entrypoint: execute one already-rendered task.

    python run.py --claim_id c001_len --spec-hash <hash> --server verl-grpo-44487
    python run.py --task-dir runs/paper/03-task/c001_len/<hash> --server verl-grpo-44487

Phases: resolve task -> probe server -> launch sandbox (in-sandbox Claude Code
reproduction agent ssh's to the MetaX node and runs the experiment there) ->
persist trajectory (raw stream-json + API-level capture: merged + SFT).

.env is forced as the authoritative provider (the operator's ambient shell points
at a different relay). Local outputs go to /home because /data is full; the heavy
compute runs on the MetaX node, not locally.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from reproducegym.compute.sources import load_inventory
from reproducegym.config import parse_env_text
from reproducegym.estimate import RuntimeEstimate, estimate_runtime
from reproducegym.runlayout import PaperLayout, write_run_record
from reproducegym.sandbox.backends import ClaudeCodeBackend
from reproducegym.sandbox.launcher import launch
from reproducegym.sandbox.runner import run

_PROVIDER_KEYS = (
    "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY",
    "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "CLAUDE_CODE_MAX_TURNS",
)


def force_env_provider() -> None:
    envd = parse_env_text((REPO / ".env").read_text(encoding="utf-8"))
    for k in _PROVIDER_KEYS:
        if envd.get(k):
            os.environ[k] = envd[k]
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


def _candidate_tasks() -> list[Path]:
    tasks: list[Path] = []
    for claim_dir in sorted((REPO / "runs").glob("*/03-task/*")):
        if (claim_dir / "data_entry.json").is_file():
            tasks.append(claim_dir)
            continue
        if claim_dir.is_dir():
            for hash_dir in sorted(p for p in claim_dir.iterdir() if p.is_dir()):
                if (hash_dir / "data_entry.json").is_file():
                    tasks.append(hash_dir)
    return tasks


def _task_metadata(task: Path) -> dict:
    try:
        return json.loads((task / "data_entry.json").read_text(encoding="utf-8")).get("metadata", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _claim_id_for_task(task: Path, meta: dict | None = None) -> str:
    meta = meta or _task_metadata(task)
    if meta.get("claim_id"):
        return str(meta["claim_id"])
    if task.parent.parent.name == "03-task":
        return task.parent.name
    return task.name


def resolve_task(claim_id: str, spec_hash: str | None = None) -> Path | None:
    matches: list[Path] = []
    for task in _candidate_tasks():
        meta = _task_metadata(task)
        task_claim = _claim_id_for_task(task, meta)
        if task_claim != claim_id and task.name != claim_id and task.parent.name != claim_id:
            continue
        if spec_hash and meta.get("spec_hash") != spec_hash and task.name != spec_hash:
            continue
        matches.append(task)
    if not matches:
        return None
    if spec_hash:
        return matches[0]
    hashes = {(_task_metadata(t).get("spec_hash") or "") for t in matches}
    if len(matches) > 1 and len(hashes) > 1:
        choices = [
            f"{_task_metadata(t).get('claim_id', t.parent.name)}@{_task_metadata(t).get('spec_hash', t.name)}"
            for t in matches
        ]
        raise SystemExit(
            f"[resolve] claim {claim_id!r} has multiple task versions: {choices}; pass --spec-hash"
        )
    return matches[0]


def resolve_existing_task(
    *,
    task_dir: str | None,
    claim_id: str | None,
    spec_hash: str | None,
) -> Path:
    if task_dir:
        task = Path(task_dir)
        if not (task / "data_entry.json").is_file():
            raise SystemExit(f"[resolve] not a rendered task dir (missing data_entry.json): {task}")
        meta = _task_metadata(task)
        if claim_id and _claim_id_for_task(task, meta) != claim_id:
            raise SystemExit(f"[resolve] task claim_id {_claim_id_for_task(task, meta)!r} != {claim_id!r}")
        if spec_hash and meta.get("spec_hash") != spec_hash and task.name != spec_hash:
            raise SystemExit(f"[resolve] task spec_hash {meta.get('spec_hash')!r} != {spec_hash!r}")
        return task
    if not claim_id:
        raise SystemExit("[resolve] pass --task-dir, or --claim_id with optional --spec-hash")
    task = resolve_task(claim_id, spec_hash)
    if task is None:
        hint = " Run build_claim_tasks.py first." if spec_hash is None else " Check --spec-hash or rebuild the task."
        raise SystemExit(f"[resolve] built task not found for claim {claim_id!r}.{hint}")
    return task


def probe_server(server: str, compute: str) -> None:
    inv = load_inventory(compute)
    if server not in inv:
        raise SystemExit(f"[probe] server {server!r} not in inventory; have {sorted(inv)}")
    node = inv[server]
    print(f"[probe] {server}: host={getattr(node, 'host', '?')} port={getattr(node, 'port', '?')} "
          f"user={getattr(node, 'user', '?')}", flush=True)


def _inject_runtime_hint(workspace: Path, est: RuntimeEstimate, timeout: float) -> None:
    """Tell the in-sandbox agent its time budget + the poll cadence for THIS claim.
    Turns are uncapped; the only stop is this wall-clock budget, so the agent must
    background long training and poll sparsely rather than babysit it."""
    tm = workspace / "task.md"
    if not tm.is_file():
        return
    hint = (
        "\n\n## Runtime budget\n\n"
        f"Estimated wall-clock for this claim: {est.label}. The session is killed only "
        f"after ~{timeout / 3600:.1f}h (there is NO turn limit), so you may run long.\n"
        f"Background training fully (`setsid nohup <cmd> > train.log 2>&1 &`) and poll about "
        f"every {est.poll_s // 60} min (`sleep {est.poll_s} && tail -n 30 train.log`), NOT every "
        "turn. Best: have the job write `output/result.json` + `output/metrics.csv` and `touch DONE` "
        "when finished, then just wait for DONE.\n"
    )
    tm.write_text(tm.read_text(encoding="utf-8") + hint, encoding="utf-8")


def resolve_run_dir(task: Path, out_fallback: str) -> Path:
    """Standard layout: runs/<paper>/04-run/<claim>/<hash>/NNN. Falls back to --out only
    if the task isn't part of a paper layout."""
    layout = PaperLayout.from_task_dir(task)
    meta = _task_metadata(task)
    claim_id = _claim_id_for_task(task, meta)
    spec_hash = meta.get("spec_hash")
    if layout is not None:
        return layout.next_run_dir(claim_id, spec_hash)
    base = Path(out_fallback) / claim_id
    if spec_hash:
        base = base / spec_hash
    base.mkdir(parents=True, exist_ok=True)
    n = max([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()] or [0]) + 1
    return base / f"{n:03d}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reproduce one claim on a chosen server.")
    ap.add_argument("--task-dir", help="rendered task dir: runs/<paper>/03-task/<claim>/<hash>")
    ap.add_argument("--claim_id")
    ap.add_argument("--spec-hash", help="exact task/spec version; required when a claim has multiple hashes")
    ap.add_argument("--server", required=True, help="node alias from servers.md")
    ap.add_argument("--paper", help=argparse.SUPPRESS)
    ap.add_argument("--out", default=str(REPO / "runs"), help="fallback output root (used only if task is not in a paper layout)")
    ap.add_argument("--compute", default=str(REPO / "config" / "metax_nodes.yaml"),
                    help="compute inventory spec: a path (.yaml/.json/.md) or scheme:rest (servers-md:..., lbg:...)")
    ap.add_argument("--model", default="opus[1m]")
    ap.add_argument("--max-turns", type=int, default=0,
                    help="reproduction-agent (claude -p) turn cap; 0 = UNCAPPED. A long training "
                         "polled turn-by-turn always hits a finite cap, so default to 0 and bound the "
                         "run by --timeout instead (the agent is told to background training + poll sparsely)")
    ap.add_argument("--timeout", type=float, default=0,
                    help="wall-clock seconds for the reproduction-agent subprocess; 0 = auto-estimate "
                         "from the claim's cost/requires_training (training can be >24h). Override to force a budget.")
    ap.add_argument("--run-dir", help="explicit attempt dir (pre-assigned by a parallel dispatcher to avoid NNN races)")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--no-capture", action="store_true")
    args = ap.parse_args(argv)

    force_env_provider()

    if args.paper:
        raise SystemExit(
            "[resolve] run.py no longer builds from paper. "
            "First run build_claim_tasks.py, then run by --task-dir or --claim_id/--spec-hash."
        )
    task = resolve_existing_task(task_dir=args.task_dir, claim_id=args.claim_id, spec_hash=args.spec_hash)
    print(f"[resolve] task={task}", flush=True)

    meta = json.loads((task / "data_entry.json").read_text(encoding="utf-8")).get("metadata", {})
    spec_hash = meta.get("spec_hash")
    est = estimate_runtime(requires_training=bool(meta.get("requires_training")), cost=meta.get("cost"))
    timeout = args.timeout if args.timeout and args.timeout > 0 else est.timeout_s
    print(f"[estimate] {est.label}  -> timeout={timeout / 3600:.1f}h", flush=True)

    probe_server(args.server, args.compute)
    if args.probe_only:
        print("[probe-only] OK", flush=True)
        return 0

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = resolve_run_dir(task, args.out)
    caps = run_dir / "trajectory" / "captures"
    session = meta.get("claim_id") or args.claim_id or task.name

    capture = not args.no_capture
    if capture:
        from agent_trace.proxy.capture_writer import CaptureWriter
        from agent_trace.proxy.server import ProxyConfig, serve
        upstream = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
        cfg = ProxyConfig(upstream_base=upstream, writer=CaptureWriter(caps), default_session=session)
        httpd = serve(cfg, port=0)
        os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{httpd.server_address[1]}"
        print(f"[capture] proxy :{httpd.server_address[1]} -> {upstream}", flush=True)

    backend = ClaudeCodeBackend(model=args.model, max_turns=args.max_turns)
    rt = launch(str(task), run_dir, backend=backend,
                compute=args.compute, node=args.server)
    _inject_runtime_hint(rt.workspace, est, timeout)
    print(f"[launch] node={list(rt.metax_nodes)} workspace={rt.workspace}", flush=True)
    res = run(rt, timeout=timeout)
    print(f"[run] returncode={res.returncode} trajectory={res.trajectory_path}", flush=True)

    if capture:
        from agent_trace.build.anthropic_adapter import to_chat_session
        from agent_trace.build.registry import get_builder
        from agent_trace.export.sft import trajectory_to_sft
        from agent_trace.store.loader import load_session
        sess = load_session(caps, session)
        if sess.completions:
            traj = get_builder("message_prefix_merging").build(to_chat_session(sess))
            (run_dir / "trajectory" / "trajectory.merged.json").write_text(
                json.dumps(dataclasses.asdict(traj), ensure_ascii=False, indent=2), encoding="utf-8")
            with open(run_dir / "trajectory" / "sft.jsonl", "w", encoding="utf-8") as f:
                for s in trajectory_to_sft(traj):
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"[capture] completions={len(sess.completions)} traces={len(traj.traces)}", flush=True)
        else:
            print("[capture] WARNING: 0 completions captured", flush=True)

    write_run_record(
        run_dir,
        {
            "claim_id": meta.get("claim_id", args.claim_id),
            "spec_hash": spec_hash,
            "backend": backend.name,
            "node": args.server,
            "status": "ran",
            "returncode": res.returncode,
            "session_id": res.session_id,
            "trajectory_path": str(res.trajectory_path),
        },
    )
    traj_meta = run_dir / "trajectory" / "metadata.json"
    traj_meta.parent.mkdir(parents=True, exist_ok=True)
    traj_meta.write_text(
        json.dumps(
            {"claim_id": meta.get("claim_id", args.claim_id), "spec_hash": spec_hash, "run_dir": str(run_dir)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] run_dir={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
