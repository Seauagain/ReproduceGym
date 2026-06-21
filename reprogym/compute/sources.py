"""Resolve a compute source spec into a ``{alias: MetaxNode}`` inventory.

A spec is one of:
  * ``None`` / ``""``           -> empty inventory
  * ``"scheme:rest"``           -> dispatched to the named adapter
  * a bare path                 -> sniffed by suffix (.md -> servers-md, .yaml ->
                                   yaml, .json -> json)

Keeping this a tiny registry means adding a platform/format is one parser + one
entry; launcher and CLI stay agnostic to where nodes come from.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from reprogym.compute.servers_md import parse_servers_md_file
from reprogym.metax import MetaxNode, load_nodes


def _from_servers_md(rest: str) -> dict[str, MetaxNode]:
    return parse_servers_md_file(rest)


def _from_yaml(rest: str) -> dict[str, MetaxNode]:
    data = yaml.safe_load(Path(rest).read_text(encoding="utf-8")) or {}
    if isinstance(data, dict) and "nodes" in data:
        data = data["nodes"]
    return load_nodes(data)


def _from_json(rest: str) -> dict[str, MetaxNode]:
    return load_nodes(Path(rest).read_text(encoding="utf-8"))


def _from_env(rest: str = "") -> dict[str, MetaxNode]:
    return load_nodes(None)


SOURCES = {
    "servers-md": _from_servers_md,
    "yaml": _from_yaml,
    "json": _from_json,
    "env": _from_env,
}

_SUFFIX = {
    ".md": "servers-md",
    ".markdown": "servers-md",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}


def load_inventory(spec: str | None) -> dict[str, MetaxNode]:
    if not spec:
        return {}
    # "scheme:rest" -- only when it isn't just a path that happens to contain ':'
    if ":" in spec and not Path(spec).exists():
        scheme, _, rest = spec.partition(":")
        if scheme in SOURCES:
            return SOURCES[scheme](rest)
    # bare path: sniff by suffix
    p = Path(spec)
    scheme = _SUFFIX.get(p.suffix.lower())
    if scheme:
        return SOURCES[scheme](str(p))
    raise ValueError(
        f"cannot resolve compute source {spec!r}; use a scheme ({sorted(SOURCES)}) "
        f"or a path ending in {sorted(_SUFFIX)}"
    )
