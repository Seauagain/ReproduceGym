"""Main control agent (host-side). Drives the end-to-end reproduction.

MD paper -> extract claims -> merge into a canonical claim spec -> render a
ClawGym-pure task -> (baseline) verifier -> consistency gate -> launch a host
sandbox -> run the reproduction agent (which ssh's to MetaX for GPU) while
recording the trajectory -> score with the hidden reward.

Everything here is light/host-side. Heavy work happens inside the sandbox and on
remote GPU nodes. The reward/ verifier is never mounted for the agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reproducegym.config import REPO_ROOT
from reproducegym.pipeline.extract_claims import extract_claims
from reproducegym.pipeline.merge_claim_spec import merge_claim_spec
from reproducegym.pipeline.render_check import write_baseline_check
from reproducegym.pipeline.render_task import render_task
from reproducegym.pipeline.validate_task import validate_task
from reproducegym.runlayout import PaperLayout, write_index, write_run_record
from reproducegym.sandbox.launcher import launch
from reproducegym.sandbox.run_guard import run_guarded
from reproducegym.sandbox.runner import RunResult
from reproducegym.verify import score


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
        if c.get("claim_id") == claim_id:
            return c
    raise ReproduceError(f"claim_id {claim_id!r} not found; have {[c.get('claim_id') for c in claims]}")


def build_task(
    paper: str | Path,
    claim_id: str | None = None,
    *,
    client: Any = None,
    paper_id: str | None = None,
    work_dir: str | Path | None = None,
    baseline_check: bool = True,
) -> BuildResult:
    """Pipeline only: paper MD -> claims -> spec -> rendered task -> validated."""
    paper_text, derived_id = _read_paper(paper)
    paper_id = paper_id or derived_id

    if client is None:
        from reproducegym.models import ClaudeClient

        client = ClaudeClient()

    # 1-2. extract + select
    claims = extract_claims(paper_text, client=client)
    claim = _select_claim(claims, claim_id)
    cid = claim["claim_id"]

    # Layout: one numbered, self-describing directory per paper.
    layout = PaperLayout(Path(work_dir)) if work_dir is not None else PaperLayout.for_paper(
        REPO_ROOT / "runs", paper_id
    )
    task_dir = layout.task_dir(cid)

    # 1b. persist extraction inputs/outputs (01-extract) + a paper snapshot.
    layout.extract_dir.mkdir(parents=True, exist_ok=True)
    (layout.extract_dir / "claims.json").write_text(
        json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not layout.paper_path.exists():
        layout.paper_path.write_text(paper_text, encoding="utf-8")

    # 3. merge into canonical spec (02-spec; single source of truth)
    spec = merge_claim_spec(claim, paper_id=paper_id, out_path=layout.spec_path(cid))

    # 4. render task (03-task, ClawGym-pure) + baseline verifier
    render_task(spec, task_dir)
    if baseline_check and not (task_dir / "reward" / "check.py").exists():
        write_baseline_check(spec, task_dir / "reward")

    # 5. consistency gate
    problems = validate_task(task_dir, spec)
    if problems:
        raise ReproduceError("task failed validation: " + "; ".join(problems))

    # 5b. refresh the human/agent index.
    write_index(layout, paper_id=paper_id)

    return BuildResult(
        paper_id=paper_id,
        claim_id=cid,
        claim_spec=spec,
        claim_spec_path=layout.spec_path(cid),
        task_dir=task_dir,
        validation=problems,
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
    b = build_task(
        paper,
        claim_id,
        client=client,
        paper_id=paper_id,
        work_dir=work_dir,
        baseline_check=baseline_check,
    )

    # 6-7. launch host sandbox + run agent + record trajectory. The run guard
    # always reclaims provisioned (e.g. Bohrium) compute, even on error/timeout.
    runtime = launch(
        b.task_dir,
        run_dir,
        backend=backend,
        sandbox=sandbox,
        metax_nodes=metax_nodes,
        compute=compute,
        node=node,
    )
    rr = run_guarded(runtime, timeout=timeout, runner=lbg_runner)

    # 8. hidden scoring
    reward = score(b.task_dir, runtime.workspace) if do_score else None

    # 8b. record the attempt + refresh the index so the run dir stays self-describing.
    try:
        write_run_record(
            runtime.run_dir,
            {
                "claim_id": b.claim_id,
                "paper_id": b.paper_id,
                "backend": getattr(runtime.backend, "name", str(backend)),
                "node": node,
                "status": "scored" if reward is not None else "ran",
                "reward": reward,
                "returncode": rr.returncode,
                "session_id": rr.session_id,
                "trajectory_path": str(rr.trajectory_path),
            },
        )
        layout = PaperLayout.from_task_dir(b.task_dir)
        if layout is not None:
            write_index(layout, paper_id=b.paper_id)
    except OSError:
        pass

    return ReproduceResult(
        paper_id=b.paper_id,
        claim_id=b.claim_id,
        claim_spec=b.claim_spec,
        claim_spec_path=b.claim_spec_path,
        task_dir=b.task_dir,
        run_result=rr,
        trajectory_path=rr.trajectory_path,
        reward=reward,
        validation=b.validation,
    )
