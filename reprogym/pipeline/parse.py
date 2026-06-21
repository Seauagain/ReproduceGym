"""Step 1: PDF -> Markdown + figures, via MinerU cloud API.

Outputs sandboxes/<paper_id>/paper.md, paper.json, and figures/ (image files +
captions/refs) used downstream by figure-param extraction. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def parse_pdf(pdf_path: Path, out_dir: Path) -> Path:
    raise NotImplementedError("scaffold: MinerU API -> paper.md + figures/")
