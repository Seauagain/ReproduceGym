"""Step 4: launch a sandbox on the host for a rendered task.

Reads the task's data_entry.json, prepares the agent workspace from input_files/
(reward/ stays out), and assembles a Runtime bundling the chosen agent backend,
the host sandbox, the user_query, task metadata, and the MetaX node inventory the
in-sandbox agent may ssh into. Nothing heavy happens here; runner does the work.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from reproducegym.compute.providers import ComputeProvider, LbgProvider
from reproducegym.compute.sources import load_inventory
from reproducegym.config import REPO_ROOT
from reproducegym.metax import MetaxNode, install_compute_access, load_metax_config, load_nodes
from reproducegym.runlayout import PaperLayout
from reproducegym.sandbox.backends import AgentBackend, get_backend
from reproducegym.sandbox.sandbox import LocalSandbox, Sandbox
from reproducegym.sandbox.workspace import prepare_workspace


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
    providers: list[ComputeProvider] = field(default_factory=list)
    run_tag: str = ""


def _default_run_dir(task_dir: Path, task_id: str) -> Path:
    """Nest the attempt under the paper layout (04-run/<claim>/NNN) when the task
    dir follows the layout; otherwise fall back to a flat timestamped dir."""
    layout = PaperLayout.from_task_dir(task_dir)
    if layout is not None:
        return layout.next_run_dir(task_id)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "runs" / f"{task_id}-{stamp}"


def _run_tag(run_dir: Path) -> str:
    """Stable per-run label; provisioned sandboxes are named with this prefix so
    the host teardown sweep can reclaim exactly this run's resources."""
    return re.sub(r"[^a-z0-9-]+", "-", run_dir.name.lower()).strip("-") or "reproducegym-run"


def _resolve_compute_spec(compute: str | None) -> str | None:
    """Compute source spec: explicit arg > REPRODUCEGYM_COMPUTE > REPRODUCEGYM_SERVERS_MD."""
    if compute:
        return compute
    if os.environ.get("REPRODUCEGYM_COMPUTE"):
        return os.environ["REPRODUCEGYM_COMPUTE"]
    if os.environ.get("REPRODUCEGYM_SERVERS_MD"):
        return "servers-md:" + os.environ["REPRODUCEGYM_SERVERS_MD"]
    return None


def launch(
    task_dir: str | Path,
    run_dir: str | Path | None = None,
    *,
    backend: str | AgentBackend = "claude-code",
    sandbox: Sandbox | None = None,
    metax_nodes: Any = None,
    compute: str | None = None,
    node: str | None = None,
    clean: bool = False,
) -> Runtime:
    task_dir = Path(task_dir)
    data_entry = json.loads((task_dir / "data_entry.json").read_text(encoding="utf-8"))
    task_id = data_entry.get("task_id", task_dir.name)

    run_dir = Path(run_dir) if run_dir is not None else _default_run_dir(task_dir, task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = prepare_workspace(task_dir, run_dir / "workspace", clean=clean)

    # Resolve compute: explicit ssh nodes > compute source > yaml config > env.
    # A spec may select ssh nodes (servers.md/yaml/json) OR a provisioned-sandbox
    # provider (lbg:<params>); the two models are mutually exclusive per run.
    cfg = load_metax_config()
    spec = _resolve_compute_spec(compute)
    nodes: dict[str, MetaxNode] = {}
    providers: list[ComputeProvider] = []
    if metax_nodes is not None:
        nodes = load_nodes(metax_nodes)
    elif spec:
        scheme, _, rest = spec.partition(":")
        if scheme == "lbg":
            providers = [LbgProvider.from_spec(rest)]
        else:
            nodes = load_inventory(spec)
    elif cfg.get("nodes"):
        nodes = cfg["nodes"]
    else:
        nodes = load_nodes(None)

    # Host-side node selection: narrow the ssh inventory to a single alias.
    if node:
        if node not in nodes:
            raise KeyError(f"node {node!r} not in inventory; have {sorted(nodes)}")
        nodes = {node: nodes[node]}

    run_tag = _run_tag(run_dir)

    # Give the in-sandbox agent a usable way to reach the chosen compute.
    if nodes:
        install_compute_access(
            workspace,
            nodes,
            launch_template=cfg.get("launch_template", ""),
            notes=cfg.get("notes", ""),
            remote_workdir=cfg.get("remote_workdir", ""),
        )
    for provider in providers:
        provider.install(workspace, run_tag=run_tag)

    return Runtime(
        task_dir=task_dir,
        run_dir=run_dir,
        workspace=workspace,
        backend=get_backend(backend),
        sandbox=sandbox or LocalSandbox(),
        user_query=data_entry.get("user_query", ""),
        metadata=data_entry.get("metadata", {}),
        metax_nodes=nodes,
        providers=providers,
        run_tag=run_tag,
    )
