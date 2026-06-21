"""Step 4: launch a sandbox on the host for a rendered task.

Reads the task's data_entry.json, prepares the agent workspace from input_files/
(reward/ stays out), and assembles a Runtime bundling the chosen agent backend,
the host sandbox, the user_query, task metadata, and the MetaX node inventory the
in-sandbox agent may ssh into. Nothing heavy happens here; runner does the work.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from reprogym.compute.sources import load_inventory
from reprogym.config import REPO_ROOT
from reprogym.metax import MetaxNode, install_compute_access, load_metax_config, load_nodes
from reprogym.sandbox.backends import AgentBackend, get_backend
from reprogym.sandbox.sandbox import LocalSandbox, Sandbox
from reprogym.sandbox.workspace import prepare_workspace


@dataclass
class Runtime:
    task_dir: Path
    run_dir: Path
    workspace: Path
    backend: AgentBackend
    sandbox: Sandbox
    user_query: str
    metadata: dict[str, Any] = field(default_factory=dict)
    metax_nodes: dict[str, MetaxNode] = field(default_factory=dict)


def _default_run_dir(task_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "runs" / f"{task_id}-{stamp}"


def _resolve_compute_spec(compute: str | None) -> str | None:
    """Compute source spec: explicit arg > REPROGYM_COMPUTE > REPROGYM_SERVERS_MD."""
    if compute:
        return compute
    if os.environ.get("REPROGYM_COMPUTE"):
        return os.environ["REPROGYM_COMPUTE"]
    if os.environ.get("REPROGYM_SERVERS_MD"):
        return "servers-md:" + os.environ["REPROGYM_SERVERS_MD"]
    return None


def launch(
    task_dir: str | Path,
    run_dir: str | Path | None = None,
    *,
    backend: str | AgentBackend = "claude-code",
    sandbox: Sandbox | None = None,
    metax_nodes: Any = None,
    compute: str | None = None,
    clean: bool = False,
) -> Runtime:
    task_dir = Path(task_dir)
    data_entry = json.loads((task_dir / "data_entry.json").read_text(encoding="utf-8"))
    task_id = data_entry.get("task_id", task_dir.name)

    run_dir = Path(run_dir) if run_dir is not None else _default_run_dir(task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = prepare_workspace(task_dir, run_dir / "workspace", clean=clean)

    # Resolve nodes: explicit arg > compute source (servers.md/yaml/json) >
    # yaml config file > REPROGYM_METAX_NODES env.
    cfg = load_metax_config()
    spec = _resolve_compute_spec(compute)
    if metax_nodes is not None:
        nodes = load_nodes(metax_nodes)
    elif spec:
        nodes = load_inventory(spec)
    elif cfg.get("nodes"):
        nodes = cfg["nodes"]
    else:
        nodes = load_nodes(None)

    # If we have nodes, give the in-sandbox agent a usable way to reach them.
    if nodes:
        install_compute_access(
            workspace,
            nodes,
            launch_template=cfg.get("launch_template", ""),
            notes=cfg.get("notes", ""),
            remote_workdir=cfg.get("remote_workdir", ""),
        )

    return Runtime(
        task_dir=task_dir,
        run_dir=run_dir,
        workspace=workspace,
        backend=get_backend(backend),
        sandbox=sandbox or LocalSandbox(),
        user_query=data_entry.get("user_query", ""),
        metadata=data_entry.get("metadata", {}),
        metax_nodes=nodes,
    )
