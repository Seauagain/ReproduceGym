"""Step 2.5: triage extracted claims -> which become sandbox tasks + the v0.

Scores claims (scientific value, training value, verifiability, cost,
information completeness, diversity) via prompts/claim_triage.md and writes
sandboxes/<paper>/paper_triage.yaml (build[]/defer[]/v0/rationale). Also emits
sandboxes/<paper>/resource_profile.yaml from per-claim cost/requires_training.
Only build[] claims proceed to merge_claim_spec + render_task. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def triage(claims: list[dict], out_dir: Path) -> dict:
    raise NotImplementedError("scaffold: Claude + prompts/claim_triage.md -> paper_triage.yaml")


def write_resource_profile(claims: list[dict], out_dir: Path) -> Path:
    raise NotImplementedError("scaffold: aggregate cost/requires_training -> resource_profile.yaml")
