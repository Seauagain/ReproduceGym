"""Compute inventory adapters (decoupled from core).

The core pipeline only ever sees a generic ``{alias: MetaxNode}`` inventory. How
that inventory is sourced -- a servers.md registry, a yaml/json file, or an env
var -- lives here as small, optional adapters. Adding a platform or format means
adding one parser + one registry entry; core and CLI do not change.
"""

from reproducegym.compute.providers import ComputeProvider, LbgProvider, render_lbg_card
from reproducegym.compute.servers_md import parse_servers_md, parse_servers_md_file
from reproducegym.compute.sources import SOURCES, load_inventory

__all__ = [
    "parse_servers_md",
    "parse_servers_md_file",
    "load_inventory",
    "SOURCES",
    "ComputeProvider",
    "LbgProvider",
    "render_lbg_card",
]
