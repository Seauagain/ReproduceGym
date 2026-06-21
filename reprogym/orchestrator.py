"""Main control agent (host-side). Drives the end-to-end reproduction.

MD paper -> extract claims -> merge into a canonical claim spec -> render a
ClawGym-pure task -> (baseline) verifier -> consistency gate -> launch a host
sandbox -> run the reproduction agent (which ssh's to MetaX for GPU) while
recording the trajectory -> score with the hidden reward.

Everything here is light/host-side. Heavy work happens inside the sandbox and on
remote GPU nodes. The reward/ verifier is never mounted for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reprogym.config import REPO_ROOT
from reprogym.pipeline.extract_claims import extract_claims
from reprogym.pipeline.merge_claim_spec import merge_claim_spec
from reprogym.pipeline.render_check import write_baseline_check
from reprogym.pipeline.render_task import render_task
from reprogym.pipeline.validate_task import validate_task
from reprogym.sandbox.launcher import launch
from reprogym.sandbox.runner import RunResult, run
from reprogym.verify import score


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
        if c.get("claim_id") == claim_id:
            return c
    raise ReproduceError(f"claim_id {claim_id!r} not found; have {[c.get('claim_id') for c in claims]}")


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
    timeout: float | None = None,
    baseline_check: bool = True,
    do_score: bool = True,
) -> ReproduceResult:
    paper_text, derived_id = _read_paper(paper)
    paper_id = paper_id or derived_id

    if client is None:
        from reprogym.models import ClaudeClient

        client = ClaudeClient()

    # 1-2. extract + select
    claims = extract_claims(paper_text, client=client)
    claim = _select_claim(claims, claim_id)
    cid = claim["claim_id"]

    # 3. merge into canonical spec (single source of truth)
    build_root = Path(work_dir) if work_dir is not None else REPO_ROOT / "runs" / paper_id
    claims_dir = build_root / "claims"
    task_dir = build_root / "tasks" / cid
    spec = merge_claim_spec(claim, paper_id=paper_id, out_path=claims_dir / f"{cid}.yaml")

    # 4. render task (ClawGym-pure) + baseline verifier
    render_task(spec, task_dir)
    if baseline_check and not (task_dir / "reward" / "check.py").exists():
        write_baseline_check(spec, task_dir / "reward")

    # 5. consistency gate
    problems = validate_task(task_dir, spec)
    if problems:
        raise ReproduceError("task failed validation: " + "; ".join(problems))

    # 6-7. launch host sandbox + run agent + record trajectory
    runtime = launch(task_dir, run_dir, backend=backend, sandbox=sandbox, metax_nodes=metax_nodes)
    rr = run(runtime, timeout=timeout)

    # 8. hidden scoring
    reward = score(task_dir, runtime.workspace) if do_score else None

    return ReproduceResult(
        paper_id=paper_id,
        claim_id=cid,
        claim_spec=spec,
        claim_spec_path=claims_dir / f"{cid}.yaml",
        task_dir=task_dir,
        run_result=rr,
        trajectory_path=rr.trajectory_path,
        reward=reward,
        validation=problems,
    )
