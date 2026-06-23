#!/usr/bin/env python3
"""Build claim/task bundles from one paper, without launching reproduction runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from reproducegym.pipeline.build_claim_tasks import build_claim_tasks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build multimodal claim-task bundles only.")
    ap.add_argument("--paper", required=True, help="local Markdown paper with local image refs")
    ap.add_argument("--paper-id", help="paper id under runs/; defaults to paper filename stem")
    ap.add_argument("--out", default=str(REPO / "runs"), help="runs root")
    ap.add_argument("--claim-id", action="append", default=[], help="build only this claim id/source id; repeatable")
    ap.add_argument(
        "--max-claims",
        type=int,
        default=3,
        help="when --claim-id is not used, render only the top N claims by selection score (default: 3; 0 = all)",
    )
    ap.add_argument("--refresh-claims", action="store_true", help="ignore cached claim_candidates.* and re-extract claims")
    ap.add_argument("--no-baseline-check", action="store_true", help="do not write baseline reward/check.py")
    ap.add_argument(
        "--parse-images",
        "--解析图片",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "image-enhanced extraction mode. auto = run only when local figures and a "
            "multimodal model are configured; always = require both; never = text-only"
        ),
    )
    ap.add_argument("--vl-min-confidence", type=float, default=0.0)
    ap.add_argument("--non-strict-vl", action="store_true", help="skip malformed VL responses instead of failing")
    args = ap.parse_args(argv)

    try:
        result = build_claim_tasks(
            paper=args.paper,
            paper_id=args.paper_id,
            out=args.out,
            claim_ids=args.claim_id,
            parse_images=args.parse_images,
            vl_min_confidence=args.vl_min_confidence,
            strict_vl=not args.non_strict_vl,
            baseline_check=not args.no_baseline_check,
            max_claims=None if args.max_claims == 0 else args.max_claims,
            refresh_claims=args.refresh_claims,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
