"""Legacy host-side orchestration helpers.

The active workflow is intentionally two-stage:

1. paper -> task bundles: :mod:`reproducegym.pipeline.build_claim_tasks`
2. task bundle -> run attempt: :mod:`run.py` or ``reproducegym reproduce <task_dir>``

The old paper -> task -> run orchestration entrypoints remain only to raise a
clear migration error instead of silently building text-only tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reproducegym.sandbox.runner import RunResult


@dataclass
class BuildResult:
    paper_id: str
    claim_id: str
    claim_spec: dict[str, Any]
    claim_spec_path: Path
    task_dir: Path
    validation: list[str]


@dataclass
class ReproduceResult:
    paper_id: str
    claim_id: str
    claim_spec: dict[str, Any]
    claim_spec_path: Path
    task_dir: Path
    run_result: RunResult
    trajectory_path: Path
    reward: float | None
    validation: list[str]


class ReproduceError(RuntimeError):
    pass


def _read_paper(paper: str | Path) -> tuple[str, str]:
    """Return (paper_text, derived_paper_id)."""
    p = Path(paper) if isinstance(paper, (str, Path)) else None
    if isinstance(paper, Path) or (isinstance(paper, str) and "\n" not in paper and p and p.is_file()):
        return p.read_text(encoding="utf-8"), p.stem
    return str(paper), "paper"


def _select_claim(claims: list[dict], claim_id: str | None) -> dict:
    if not claims:
        raise ReproduceError("no claims were extracted from the paper")
    if claim_id is None:
        return claims[0]
    for c in claims:
        if c.get("claim_id") == claim_id or c.get("source_claim_id") == claim_id:
            return c
    have = [c.get("claim_id") for c in claims]
    legacy = [c.get("source_claim_id") for c in claims if c.get("source_claim_id")]
    raise ReproduceError(f"claim_id {claim_id!r} not found; have {have}; legacy ids {legacy}")


def build_task(
    paper: str | Path,
    claim_id: str | None = None,
    *,
    client: Any = None,
    paper_id: str | None = None,
    work_dir: str | Path | None = None,
    baseline_check: bool = True,
    figure_evidence: list[dict[str, Any]] | None = None,
) -> BuildResult:
    """Disabled legacy paper -> task helper."""
    raise ReproduceError(
        "orchestrator.build_task is disabled. Use build_claim_tasks.py or "
        "reproducegym build for the paper -> task stage."
    )


def reproduce(
    paper: str | Path,
    claim_id: str | None = None,
    *,
    client: Any = None,
    backend: Any = "claude-code",
    sandbox: Any = None,
    paper_id: str | None = None,
    work_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    metax_nodes: Any = None,
    compute: str | None = None,
    node: str | None = None,
    lbg_runner: Any = None,
    timeout: float | None = None,
    baseline_check: bool = True,
    do_score: bool = True,
) -> ReproduceResult:
    """Disabled legacy paper -> task -> run helper."""
    raise ReproduceError(
        "orchestrator.reproduce is disabled. Build tasks first, then run a rendered "
        "task with run.py or reproducegym reproduce <task_dir>."
    )
