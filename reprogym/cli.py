"""Command-line entry point.

    reprogym reproduce <paper.pdf> [--claim <claim_id>]   # 1-7 end-to-end
    reprogym build     <paper.pdf>                         # pipeline only (no run)
    reprogym dataset   <name> --paper <paper_id> ...       # flatten -> datasets/<name>

Thin wrapper over orchestrator / pipeline / dataset. Stub only.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("scaffold: wire argparse -> orchestrator/pipeline/dataset")


if __name__ == "__main__":
    raise SystemExit(main())
