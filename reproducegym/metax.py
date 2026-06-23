"""MetaX / verl remote-node access (host-side helpers).

The reproduction agent works locally in its host sandbox and reaches GPU nodes by
plain ssh -- ops are ordinary shell actions captured into the trajectory, not
wrapped in a submit/poll abstraction. This module only provides the node
inventory and a correct ssh command builder; it never runs anything itself.

Node inventory comes from (in priority order): an explicit argument, the
REPRODUCEGYM_METAX_NODES env var (JSON), or nothing. The runner forwards the
inventory into the sandbox env so the in-sandbox agent can resolve aliases.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from reproducegym.config import REPO_ROOT

SSH_DEFAULT_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "metax_nodes.yaml"


@dataclass
class MetaxNode:
    alias: str
    host: str
    user: str = "root"
    port: int = 22
    key_path: str | None = None
    workdir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_node(alias: str, raw: dict[str, Any]) -> MetaxNode:
    fields = {k: raw[k] for k in ("host", "user", "port", "key_path", "workdir") if k in raw}
    return MetaxNode(alias=alias, **fields)


def load_nodes(source: Any = None) -> dict[str, MetaxNode]:
    """Load the node inventory from a dict/list/JSON string/env into {alias: node}."""
    if source is None:
        raw = os.environ.get("REPRODUCEGYM_METAX_NODES")
        source = json.loads(raw) if raw else {}
    elif isinstance(source, str):
        source = json.loads(source)

    nodes: dict[str, MetaxNode] = {}
    if isinstance(source, dict):
        for alias, raw in source.items():
            nodes[alias] = _coerce_node(alias, raw)
    elif isinstance(source, list):
        for raw in source:
            alias = raw["alias"]
            nodes[alias] = _coerce_node(alias, {k: v for k, v in raw.items() if k != "alias"})
    else:
        raise TypeError(f"unsupported node source: {type(source).__name__}")
    return nodes


def ssh_command(node: MetaxNode, remote_cmd: str, *, opts: list[str] | None = None) -> list[str]:
    """Build an ssh argv that runs `remote_cmd` on `node` (no execution here)."""
    cmd = ["ssh"]
    cmd += SSH_DEFAULT_OPTS if opts is None else opts
    if node.port and node.port != 22:
        cmd += ["-p", str(node.port)]
    if node.key_path:
        cmd += ["-i", node.key_path]
    cmd.append(f"{node.user}@{node.host}")
    cmd.append(remote_cmd)
    return cmd


def nodes_to_env(nodes: dict[str, MetaxNode]) -> str:
    """Serialize the inventory for REPRODUCEGYM_METAX_NODES (forwarded into the sandbox)."""
    return json.dumps({alias: node.to_dict() for alias, node in nodes.items()})


def load_metax_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the MetaX config file (nodes + launch_template + notes).

    Source: explicit path, else REPRODUCEGYM_METAX_CONFIG env, else config/metax_nodes.yaml.
    Missing file -> {} (no error). Shape:

        nodes: {alias: {host, user, port, key_path, workdir}}
        remote_workdir: "/workspace/verl"
        launch_template: "cd {workdir} && bash run_grpo.sh ..."
        notes: "free text shown to the agent"
    """
    p = Path(path) if path else Path(os.environ.get("REPRODUCEGYM_METAX_CONFIG", DEFAULT_CONFIG_PATH))
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    nodes_raw = data.get("nodes") or {}
    return {
        "nodes": load_nodes(nodes_raw) if nodes_raw else {},
        "remote_workdir": data.get("remote_workdir", ""),
        "launch_template": data.get("launch_template", ""),
        "notes": data.get("notes", ""),
    }


# A self-contained ssh wrapper dropped into the agent's workspace. It resolves an
# alias from the sibling metax_nodes.json and execs ssh -- no reproducegym import, so
# it works inside any sandbox that has python3 + ssh.
METAX_SSH_SCRIPT = '''#!/usr/bin/env python3
"""Resolve a MetaX node alias from metax_nodes.json and ssh to it.

usage: python3 metax_ssh.py <alias> ["<remote command>"]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
nodes = json.load(open(os.path.join(HERE, "metax_nodes.json")))

if len(sys.argv) < 2 or sys.argv[1] not in nodes:
    sys.stderr.write("usage: python3 metax_ssh.py <alias> [cmd]; aliases: %s\\n" % ", ".join(nodes))
    raise SystemExit(2)

n = nodes[sys.argv[1]]
cmd = sys.argv[2] if len(sys.argv) > 2 else ""
argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
if int(n.get("port", 22)) != 22:
    argv += ["-p", str(n["port"])]
if n.get("key_path"):
    argv += ["-i", os.path.expanduser(n["key_path"])]
argv.append("%s@%s" % (n.get("user", "root"), n["host"]))
if cmd:
    argv.append(cmd)
os.execvp("ssh", argv)
'''


def render_compute_access(
    nodes: dict[str, MetaxNode],
    *,
    launch_template: str = "",
    notes: str = "",
    remote_workdir: str = "",
) -> str:
    """Markdown the in-sandbox agent can act on to reach compute.

    Compute choice is intentionally NOT baked into the task itself: this doc is
    written as a standalone file and only *referenced* from task.md, so the task
    statement stays about the science and the agent decides where to run.
    """
    lines = ["# Compute access", ""]
    lines.append("You choose where to run this task:")
    lines += [
        "",
        "- **Local** — run directly in this sandbox. Fine for light / CPU-only work, "
        "analysis, or small experiments.",
        "- **Remote GPU (MetaX / verl)** — ssh to the nodes below for GPU training. "
        "Choose this when the claim needs training or a GPU.",
        "",
        "Reach the remote nodes through the provided wrapper (it reads `metax_nodes.json`):",
        "",
        "    python3 metax_ssh.py <alias> \"<remote command>\"",
        "",
    ]
    lines.append("Available nodes:")
    for alias, n in nodes.items():
        wd = f" (workdir: {n.workdir})" if n.workdir else ""
        lines.append(f"- `{alias}` -> {n.user}@{n.host}{wd}")
    lines += ["", "Example:", "", "    python3 metax_ssh.py "
              + (next(iter(nodes)) if nodes else "<alias>") + " \"nvidia-smi\"", ""]
    if remote_workdir:
        lines += [f"Remote working directory: `{remote_workdir}`", ""]
    if launch_template:
        lines += ["verl launch template:", "", "```bash", launch_template.strip(), "```", ""]
    if notes:
        lines += ["Notes:", "", notes.strip(), ""]
    lines.append(
        "Treat remote ops as ordinary shell actions. Keep heavy logs on the remote; "
        "copy back only the metrics you need into `output/`."
    )
    return "\n".join(lines) + "\n"


def install_compute_access(
    workspace: str | Path,
    nodes: dict[str, MetaxNode],
    *,
    launch_template: str = "",
    notes: str = "",
    remote_workdir: str = "",
    task_md_name: str = "task.md",
) -> list[Path]:
    """Drop metax_nodes.json + metax_ssh.py + compute_access.md into the workspace
    and append a compute-access section to the workspace task.md so the agent sees it."""
    workspace = Path(workspace)
    written: list[Path] = []

    inv = workspace / "metax_nodes.json"
    inv.write_text(nodes_to_env(nodes), encoding="utf-8")
    written.append(inv)

    wrapper = workspace / "metax_ssh.py"
    wrapper.write_text(METAX_SSH_SCRIPT, encoding="utf-8")
    wrapper.chmod(0o755)
    written.append(wrapper)

    md = render_compute_access(
        nodes, launch_template=launch_template, notes=notes, remote_workdir=remote_workdir
    )
    doc = workspace / "compute_access.md"
    doc.write_text(md, encoding="utf-8")
    written.append(doc)

    # Reference the compute doc from task.md instead of inlining it: the task stays
    # about the claim; *how/where* to run is a separate, swappable concern.
    task_md = workspace / task_md_name
    if task_md.is_file():
        ref = (
            "\n\n## Compute\n\n"
            "Where you run this task is your choice — run **locally** in this sandbox, "
            "or use the **remote MetaX / verl GPU nodes** for training. Node aliases, the "
            f"ssh wrapper, env setup and the launch recipe are in [`{doc.name}`](./{doc.name}); "
            "read it before doing any heavy compute.\n"
        )
        task_md.write_text(task_md.read_text(encoding="utf-8") + ref, encoding="utf-8")

    return written
