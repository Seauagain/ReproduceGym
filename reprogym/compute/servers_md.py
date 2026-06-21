"""Parse a servers.md node registry into a sanitized ssh node inventory.

servers.md is the existing single source of truth for compute. Each reachable
node is documented with a per-node ```yaml block; backup passwords live only in
prose ("备用密码 ..."), never in those blocks. We therefore parse ONLY the yaml
blocks and copy a FIELD WHITELIST, so by construction no password / secret can
enter the inventory even if someone later pastes one into a block.

Output is ``{alias: MetaxNode}`` -- the same shape the ssh access-card renderer
already consumes. This adapter is optional: core never imports it.

Limitation (Stage 1): only blocks that carry both ``alias`` and ``hostname``/
``host`` are captured. Table-based pools and gateway/double-@ hops (paracloud,
bscc) are out of scope here and handled by later adapters.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from reprogym.metax import MetaxNode

_YAML_BLOCK = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL)


def _node_from_block(data: dict) -> MetaxNode | None:
    alias = data.get("alias")
    host = data.get("hostname") or data.get("host")
    if not alias or not host:
        return None
    try:
        port = int(data.get("port", 22) or 22)
    except (TypeError, ValueError):
        port = 22
    # WHITELIST: only these fields are copied out -- passwords/tokens can't ride along.
    return MetaxNode(
        alias=str(alias),
        host=str(host),
        user=str(data.get("user", "root")),
        port=port,
        key_path=None,  # publickey auth only; never carry key material/passwords
        workdir=str(data["workdir"]) if data.get("workdir") else None,
    )


def parse_servers_md(text: str) -> dict[str, MetaxNode]:
    nodes: dict[str, MetaxNode] = {}
    for block in _YAML_BLOCK.findall(text):
        try:
            data = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        node = _node_from_block(data)
        if node is not None:
            nodes[node.alias] = node
    return nodes


def parse_servers_md_file(path: str | Path) -> dict[str, MetaxNode]:
    return parse_servers_md(Path(path).read_text(encoding="utf-8"))
