#!/usr/bin/env python3
"""parse.py stage entrypoint: source (url/pdf/md) -> runs/<paper_id>/00-parse/.

    python parse_paper.py --url 2503.20783
    python parse_paper.py --url https://arxiv.org/abs/2503.20783
    python parse_paper.py --pdf /path/to/paper.pdf
    python parse_paper.py --md  /path/to/paper.md

Produces a structured paper.md + local figures/ + figures.index.json via the
MinerU cloud open API. Build consumes this bundle: build_claim_tasks.py --paper runs/<paper_id>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from reproducegym.pipeline.parse import ParseError, parse_paper


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parse a paper source into a 00-parse/ bundle.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="arXiv id / abs / pdf link, or any direct PDF URL")
    src.add_argument("--pdf", help="local PDF path")
    src.add_argument("--md", help="local Markdown path (figures resolved from sibling images/)")
    ap.add_argument("--paper-id", dest="paper_id", help="paper id under runs/; auto-derived if omitted")
    ap.add_argument("--out", default=str(REPO / "runs"), help="runs root")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model", default="pipeline", help="MinerU model: pipeline | vlm")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)

    try:
        result = parse_paper(
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
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
