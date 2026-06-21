"""S1: compute source registry -- spec -> {alias: MetaxNode}, format-agnostic."""

from __future__ import annotations

import json

import pytest

from reprogym.compute.sources import load_inventory

SERVERS_MD = """
```yaml
alias:    nodeA
hostname: 10.0.0.1
port:     2222
user:     root
```
"""


def test_none_and_empty_spec_return_empty():
    assert load_inventory(None) == {}
    assert load_inventory("") == {}


def test_servers_md_via_suffix(tmp_path):
    p = tmp_path / "servers.md"
    p.write_text(SERVERS_MD, encoding="utf-8")
    nodes = load_inventory(str(p))
    assert set(nodes) == {"nodeA"}
    assert nodes["nodeA"].port == 2222


def test_servers_md_via_scheme(tmp_path):
    p = tmp_path / "registry.txt"  # non-.md suffix -> need explicit scheme
    p.write_text(SERVERS_MD, encoding="utf-8")
    nodes = load_inventory(f"servers-md:{p}")
    assert set(nodes) == {"nodeA"}


def test_json_source(tmp_path):
    p = tmp_path / "nodes.json"
    p.write_text(json.dumps({"n1": {"host": "1.2.3.4", "port": 22}}), encoding="utf-8")
    nodes = load_inventory(str(p))
    assert nodes["n1"].host == "1.2.3.4"


def test_yaml_source_nodes_key(tmp_path):
    p = tmp_path / "metax.yaml"
    p.write_text("nodes:\n  n2:\n    host: 5.6.7.8\n    port: 33\n", encoding="utf-8")
    nodes = load_inventory(str(p))
    assert nodes["n2"].port == 33


def test_unknown_suffix_raises(tmp_path):
    p = tmp_path / "nodes.bin"
    p.write_text("noop", encoding="utf-8")
    with pytest.raises(ValueError):
        load_inventory(str(p))
