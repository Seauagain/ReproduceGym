"""MetaX / verl remote-node access (host-side helpers).

The reproduction agent works locally in its host sandbox and reaches GPU nodes by
plain ssh -- ops are ordinary shell actions captured into the trajectory, not
wrapped in a submit/poll abstraction. This module only provides the node
inventory and a correct ssh command builder; it never runs anything itself.

Node inventory comes from (in priority order): an explicit argument, the
REPROGYM_METAX_NODES env var (JSON), or nothing. The runner forwards the
inventory into the sandbox env so the in-sandbox agent can resolve aliases.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

SSH_DEFAULT_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]


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
        raw = os.environ.get("REPROGYM_METAX_NODES")
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
    """Serialize the inventory for REPROGYM_METAX_NODES (forwarded into the sandbox)."""
    return json.dumps({alias: node.to_dict() for alias, node in nodes.items()})
