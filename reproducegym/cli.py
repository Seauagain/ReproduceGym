"""Command-line entry point.

    reproducegym parse     --url <arxiv|pdf>             # stage 0: source -> 00-parse/
    reproducegym build     <runs/<id>|paper.md> [--claim]# stage 1: parse bundle -> tasks
    reproducegym reproduce <task_dir>                    # stage 2: task -> run/reward
    reproducegym triage    <paper.md>                    # extract + triage + profile
    reproducegym dataset   <name> --task DIR [--task ..] # flatten -> datasets/<name>
    reproducegym score     <task_dir> <workspace>        # hidden reward.sh -> float

Thin wrapper over orchestrator / pipeline / dataset / verify.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproducegym.config import REPO_ROOT
from reproducegym.dataset import build_dataset
from reproducegym.orchestrator import _read_paper
from reproducegym.pipeline.build_claim_tasks import build_claim_tasks
from reproducegym.runlayout import PaperLayout, write_index, write_run_record
from reproducegym.pipeline.extract_claims import extract_claims
from reproducegym.pipeline.parse import ParseError, parse_paper
from reproducegym.pipeline.triage import triage, write_resource_profile
from reproducegym.sandbox.launcher import launch
from reproducegym.sandbox.run_guard import run_guarded
from reproducegym.verify import score


def _make_client():
    from reproducegym.models import ClaudeClient

    return ClaudeClient()


def _cmd_reproduce(args) -> int:
    runtime = launch(
        args.task_dir,
        args.run_dir,
        backend=args.backend,
        metax_nodes=json.loads(args.metax) if args.metax else None,
        compute=args.compute,
        node=args.node,
    )
    rr = run_guarded(runtime, timeout=args.timeout)
    reward = score(runtime.task_dir, runtime.workspace) if not args.no_score else None
    write_run_record(
        runtime.run_dir,
        {
            "claim_id": runtime.metadata.get("claim_id"),
            "spec_hash": runtime.metadata.get("spec_hash"),
            "backend": getattr(getattr(runtime, "backend", None), "name", args.backend),
            "node": args.node,
            "status": "ran",
            "returncode": getattr(rr, "returncode", None),
            "reward": reward,
            "session_id": getattr(rr, "session_id", None),
            "trajectory_path": str(rr.trajectory_path),
        },
    )
    layout = PaperLayout.from_task_dir(runtime.task_dir)
    if layout is not None:
        write_index(layout)
    print(f"claim_id:   {runtime.metadata.get('claim_id')}")
    print(f"spec_hash:  {runtime.metadata.get('spec_hash')}")
    print(f"task_dir:   {runtime.task_dir}")
    print(f"run_dir:    {runtime.run_dir}")
    print(f"trajectory: {rr.trajectory_path}")
    print(f"reward:     {reward}")
    return 0


def _cmd_build(args) -> int:
    try:
        res = build_claim_tasks(
            paper=args.paper,
            paper_id=args.paper_id,
            out=args.out,
            claim_ids=[args.claim] if args.claim else [],
            parse_images=args.parse_images,
            vl_min_confidence=args.vl_min_confidence,
            strict_vl=not args.non_strict_vl,
            baseline_check=not args.no_baseline_check,
            max_claims=None if args.max_claims == 0 else args.max_claims,
            refresh_claims=args.refresh_claims,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_parse(args) -> int:
    try:
        res = parse_paper(
            url=args.url,
            pdf=args.pdf,
            md=args.md,
            out=args.out,
            paper_id=args.paper_id,
            language=args.language,
            model=args.model,
            timeout=args.timeout,
        )
    except ParseError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_triage(args) -> int:
    client = _make_client()
    paper_text, derived = _read_paper(args.paper)
    claims = extract_claims(paper_text, client=client)
    paper_id = args.paper_id or derived
    layout = (
        PaperLayout(Path(args.out_dir))
        if args.out_dir
        else PaperLayout.for_paper(REPO_ROOT / "runs", paper_id)
    )
    out_dir = layout.extract_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "claims.json").write_text(
        json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    res = triage(claims, client=client, out_dir=out_dir)
    write_resource_profile(claims, out_dir)
    write_index(layout, paper_id=paper_id)
    print(f"build:   {res['build']}")
    print(f"v0:      {res['v0']}")
    print(f"out_dir: {out_dir}")
    return 0


def _cmd_dataset(args) -> int:
    ds = build_dataset(args.name, args.task, datasets_root=args.datasets_root, clean=args.clean)
    print(ds)
    return 0


def _cmd_score(args) -> int:
    print(score(args.task_dir, args.workspace, clamp=not args.no_clamp))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reproducegym", description="Automated RL-paper reproduction gym")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reproduce", help="run an already-rendered task")
    r.add_argument("task_dir")
    r.add_argument("--run-dir", dest="run_dir")
    r.add_argument("--backend", default="claude-code")
    r.add_argument("--timeout", type=float)
    r.add_argument("--no-score", action="store_true")
    r.add_argument("--metax", help="JSON node inventory forwarded to the sandbox")
    r.add_argument(
        "--compute",
        help="compute source: a path (servers.md/.yaml/.json) or scheme "
        "(servers-md:<path>, lbg:<project=..,gpu=..,timeout=..>)",
    )
    r.add_argument("--node", help="select a single ssh node alias from the inventory")
    r.set_defaults(func=_cmd_reproduce)

    b = sub.add_parser("build", help="parse bundle (runs/<id>) or paper.md -> hash-versioned claim tasks")
    b.add_argument("paper", help="a parse bundle dir (runs/<id> or its 00-parse/) or a raw paper.md")
    b.add_argument("--claim")
    b.add_argument("--paper-id", dest="paper_id")
    b.add_argument("--out", default=str(REPO_ROOT / "runs"))
    b.add_argument("--parse-images", "--解析图片", choices=["auto", "always", "never"], default="auto")
    b.add_argument("--vl-min-confidence", type=float, default=0.0)
    b.add_argument("--max-claims", type=int, default=3, help="render top N claims by selection score; 0 = all")
    b.add_argument("--refresh-claims", action="store_true", help="ignore cached claim candidates and re-extract")
    b.add_argument("--non-strict-vl", action="store_true")
    b.add_argument("--no-baseline-check", action="store_true")
    b.set_defaults(func=_cmd_build)

    pa = sub.add_parser("parse", help="source (url/pdf/md) -> runs/<id>/00-parse/ (MinerU)")
    pa_src = pa.add_mutually_exclusive_group(required=True)
    pa_src.add_argument("--url", help="arXiv id / abs / pdf link, or any direct PDF URL")
    pa_src.add_argument("--pdf", help="local PDF path")
    pa_src.add_argument("--md", help="local Markdown path (figures from sibling images/)")
    pa.add_argument("--paper-id", dest="paper_id")
    pa.add_argument("--out", default=str(REPO_ROOT / "runs"))
    pa.add_argument("--language", default="en")
    pa.add_argument("--model", default="pipeline")
    pa.add_argument("--timeout", type=int, default=900)
    pa.set_defaults(func=_cmd_parse)

    t = sub.add_parser("triage", help="extract claims + triage + resource profile")
    t.add_argument("paper")
    t.add_argument("--paper-id", dest="paper_id")
    t.add_argument("--out-dir", dest="out_dir")
    t.set_defaults(func=_cmd_triage)

    d = sub.add_parser("dataset", help="flatten task dirs -> datasets/<name>/")
    d.add_argument("name")
    d.add_argument("--task", action="append", required=True, dest="task")
    d.add_argument("--datasets-root", dest="datasets_root")
    d.add_argument("--clean", action="store_true")
    d.set_defaults(func=_cmd_dataset)

    s = sub.add_parser("score", help="run hidden reward.sh -> float")
    s.add_argument("task_dir")
    s.add_argument("workspace")
    s.add_argument("--no-clamp", action="store_true")
    s.set_defaults(func=_cmd_score)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
