"""Command-line entry point.

    reprogym reproduce <paper.md|pdf> [--claim ID]   # 1-8 end-to-end
    reprogym build     <paper.md|pdf> [--claim ID]   # pipeline only (no run)
    reprogym parse     <paper.pdf> -o <out>          # MinerU PDF -> paper.md
    reprogym triage    <paper.md>                    # extract + triage + profile
    reprogym dataset   <name> --task DIR [--task ..] # flatten -> datasets/<name>
    reprogym score     <task_dir> <workspace>        # hidden reward.sh -> float

Thin wrapper over orchestrator / pipeline / dataset / verify.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reprogym.config import REPO_ROOT
from reprogym.dataset import build_dataset
from reprogym.orchestrator import _read_paper, build_task, reproduce
from reprogym.pipeline.extract_claims import extract_claims
from reprogym.pipeline.parse import parse_pdf
from reprogym.pipeline.triage import triage, write_resource_profile
from reprogym.verify import score


def _make_client():
    from reprogym.models import ClaudeClient

    return ClaudeClient()


def _cmd_reproduce(args) -> int:
    res = reproduce(
        args.paper,
        args.claim,
        paper_id=args.paper_id,
        backend=args.backend,
        work_dir=args.work_dir,
        run_dir=args.run_dir,
        do_score=not args.no_score,
        metax_nodes=json.loads(args.metax) if args.metax else None,
        compute=args.compute,
        node=args.node,
    )
    print(f"claim_id:   {res.claim_id}")
    print(f"task_dir:   {res.task_dir}")
    print(f"trajectory: {res.trajectory_path}")
    print(f"reward:     {res.reward}")
    return 0


def _cmd_build(args) -> int:
    res = build_task(args.paper, args.claim, paper_id=args.paper_id, work_dir=args.work_dir)
    print(f"claim_id: {res.claim_id}")
    print(f"task_dir: {res.task_dir}")
    print("validation: ok" if not res.validation else f"validation: {res.validation}")
    return 0


def _cmd_parse(args) -> int:
    out = parse_pdf(args.pdf, args.out, language=args.language, model=args.model, timeout=args.timeout)
    print(out)
    return 0


def _cmd_triage(args) -> int:
    client = _make_client()
    paper_text, derived = _read_paper(args.paper)
    claims = extract_claims(paper_text, client=client)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "runs" / (args.paper_id or derived)
    res = triage(claims, client=client, out_dir=out_dir)
    write_resource_profile(claims, out_dir)
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
    p = argparse.ArgumentParser(prog="reprogym", description="Automated RL-paper reproduction gym")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reproduce", help="end-to-end: paper -> reward")
    r.add_argument("paper")
    r.add_argument("--claim")
    r.add_argument("--paper-id", dest="paper_id")
    r.add_argument("--work-dir", dest="work_dir")
    r.add_argument("--run-dir", dest="run_dir")
    r.add_argument("--backend", default="claude-code")
    r.add_argument("--no-score", action="store_true")
    r.add_argument("--metax", help="JSON node inventory forwarded to the sandbox")
    r.add_argument(
        "--compute",
        help="compute source: a path (servers.md/.yaml/.json) or scheme "
        "(servers-md:<path>, lbg:<project=..,gpu=..,timeout=..>)",
    )
    r.add_argument("--node", help="select a single ssh node alias from the inventory")
    r.set_defaults(func=_cmd_reproduce)

    b = sub.add_parser("build", help="pipeline only: paper -> validated task")
    b.add_argument("paper")
    b.add_argument("--claim")
    b.add_argument("--paper-id", dest="paper_id")
    b.add_argument("--work-dir", dest="work_dir")
    b.set_defaults(func=_cmd_build)

    pa = sub.add_parser("parse", help="MinerU PDF -> paper.md + figures/")
    pa.add_argument("pdf")
    pa.add_argument("-o", "--out", required=True)
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
