"""Step 1: PDF -> Markdown + figures, via MinerU.

Implemented over the `mineru-open-api` CLI (pip install mineru-open-api),
authenticated by MINERU_TOKEN. NOT a raw REST base_url.

Single:
    mineru-open-api extract <pdf> -o <out_dir> --language en --model pipeline \
        --formula --table --timeout 900

Batch (preferred for dataset building; do NOT loop single files):
    mineru-open-api extract <dir>/*.pdf -o <out_dir> --language en --model pipeline --timeout 1800
    # or --list <file-of-paths>; <=200 files/request, server-side concurrent.
    # For >200, split -l 200 and submit each chunk.

Output: a flat <basename>.md per input + a shared images/ folder (hash-named,
referenced as ![](images/<hash>.jpg)). ReproGym maps this to
sandboxes/<paper_id>/paper.md + figures/. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def parse_pdf(pdf_path: Path, out_dir: Path) -> Path:
    raise NotImplementedError("scaffold: shell `mineru-open-api extract` -> paper.md + figures/")


def parse_pdfs_batch(pdf_paths: list[Path], out_dir: Path) -> list[Path]:
    raise NotImplementedError("scaffold: one native `mineru-open-api` batch (<=200/req), chunk if more")
