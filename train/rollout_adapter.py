"""Training rollout mode: feed ReproduceGym tasks to the RL rollout.

Two entry points:

- as_rollout_source: build a flat datasets/<name>/ (via reproducegym.dataset) from
  selected sandbox tasks and return it as the rollout source_path. Same task dir,
  zero changes -> the external rollout produces trajectories + reward for policy
  updates.
- rollout: a self-contained on-policy convenience that launches the host sandbox,
  runs the agent on ONE already-built task, records the trajectory, and scores it
  with the hidden reward -- the training-mode counterpart to orchestrator.reproduce
  (which starts from a paper). The task must already carry reward/check.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reproducegym.dataset import build_dataset
from reproducegym.sandbox.launcher import launch
from reproducegym.sandbox.runner import run
from reproducegym.verify import score


def as_rollout_source(
    name: str,
    task_dirs: list[str | Path],
    *,
    datasets_root: str | Path | None = None,
    clean: bool = False,
) -> Path:
    """Flatten selected tasks into datasets/<name>/ and return the rollout source path."""
    return build_dataset(name, task_dirs, datasets_root=datasets_root, clean=clean)


def rollout(
    task_dir: str | Path,
    *,
    backend: Any = "claude-code",
    sandbox: Any = None,
    run_dir: str | Path | None = None,
    metax_nodes: Any = None,
    timeout: float | None = None,
    do_score: bool = True,
) -> dict[str, Any]:
    """Run one built task end-to-end and return its trajectory + reward."""
    runtime = launch(task_dir, run_dir, backend=backend, sandbox=sandbox, metax_nodes=metax_nodes)
    rr = run(runtime, timeout=timeout)
    reward = score(task_dir, runtime.workspace) if do_score else None
    return {
        "reward": reward,
        "trajectory": rr.trajectory,
        "trajectory_path": rr.trajectory_path,
        "session_id": rr.session_id,
        "returncode": rr.returncode,
        "workspace": runtime.workspace,
    }
