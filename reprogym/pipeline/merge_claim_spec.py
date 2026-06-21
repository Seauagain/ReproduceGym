"""Step 2c: merge claim text + figure params into a canonical claim spec.

Combines extract_claims output with figure_params, classifies each param as
paper_specified / author_repo_config / paper_unspecified (with provenance), sets
per-leaf exposure, and writes sandboxes/<paper>/claims/<claim_id>.yaml validated
against schema/claim_spec.schema.json. This file is the single source of truth.
Stub only.
"""

from __future__ import annotations

from pathlib import Path


def merge_claim_spec(claim: dict, figure_params: dict, out_path: Path) -> Path:
    raise NotImplementedError("scaffold: merge -> validated claim spec")
