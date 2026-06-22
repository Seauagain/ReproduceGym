"""Step 5: run the reproduction agent and record the trajectory.

Issues the task user_query to the in-sandbox agent. The agent works locally and
ssh's to verl/MetaX nodes for GPU when needed -- those are plain shell actions
captured into the trajectory, not wrapped in submit/poll. The MetaX inventory is
forwarded into the sandbox env (REPRODUCEGYM_METAX_NODES) so the agent can resolve
aliases. Noise distillation over raw cluster logs is offline and out of scope here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from reproducegym.metax import nodes_to_env
from reproducegym.redact import collect_secrets, redact_trajectory
from reproducegym.sandbox.launcher import Runtime
from reproducegym.trajectory import Trajectory


@dataclass
class RunResult:
    trajectory: Trajectory
    trajectory_path: Path
    session_id: str | None
    returncode: int
    workspace: Path
    stderr: str = ""


def _next_trajectory_path(run_dir: Path) -> Path:
    traj_dir = run_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    base = traj_dir / "trajectory.jsonl"
    if not base.exists():
        return base
    n = 1
    while (traj_dir / f"trajectory.{n}.jsonl").exists():
        n += 1
    return traj_dir / f"trajectory.{n}.jsonl"


def run(
    runtime: Runtime,
    user_query: str | None = None,
    *,
    resume_session_id: str | None = None,
    timeout: float | None = None,
) -> RunResult:
    prompt = user_query if user_query is not None else runtime.user_query
    resume = resume_session_id is not None
    session_id = resume_session_id or str(uuid.uuid4())

    argv = runtime.backend.build_command(prompt, session_id=session_id, resume=resume)
    env = runtime.backend.build_env(_os_environ())
    if runtime.metax_nodes:
        env["REPRODUCEGYM_METAX_NODES"] = nodes_to_env(runtime.metax_nodes)
    for provider in getattr(runtime, "providers", []):
        env.update(provider.env(run_tag=runtime.run_tag))

    result = runtime.sandbox.run(argv, cwd=runtime.workspace, env=env, timeout=timeout)

    traj = runtime.backend.parse(
        result.stdout,
        meta={
            "agent": runtime.backend.name,
            "task_id": runtime.metadata.get("claim_id") or runtime.metadata.get("paper_id"),
            "resumed": resume,
        },
    )
    sid = traj.meta.get("session_id") or session_id
    traj.meta.setdefault("session_id", sid)
    traj.meta["returncode"] = result.returncode

    # Mask injected credentials before the trajectory is persisted / 回流训练.
    secret_keys = list(getattr(runtime.backend, "env_keys", ())) + ["BOHRIUM_ACCESS_KEY"]
    for provider in getattr(runtime, "providers", []):
        secret_keys += list(getattr(provider, "env_keys", ()))
    redact_trajectory(traj, collect_secrets(env, secret_keys))

    path = _next_trajectory_path(runtime.run_dir)
    traj.dump(path)

    return RunResult(
        trajectory=traj,
        trajectory_path=path,
        session_id=sid,
        returncode=result.returncode,
        workspace=runtime.workspace,
        stderr=result.stderr,
    )


def _os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)
