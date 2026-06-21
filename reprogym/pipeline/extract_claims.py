"""Step 2a: extract reproducible claims from paper.md, via Claude.

Single structured transformation driven by prompts/extract_claims.md. Produces,
per claim: statement, anchors (section/figure/table refs), conditions, and any
text-stated params. Figure-only numbers are filled later by figure-param
extraction. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def extract_claims(paper_md: Path) -> list[dict]:
    raise NotImplementedError("scaffold: Claude + prompts/extract_claims.md")
