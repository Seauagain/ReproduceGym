"""M3: MetaX node inventory + ssh command builder."""

from __future__ import annotations

import json

import pytest

from reproducegym.metax import MetaxNode, load_nodes, nodes_to_env, ssh_command


def test_load_nodes_from_dict():
    nodes = load_nodes({"verl": {"host": "10.0.0.5", "user": "ml", "port": 2222}})
    assert nodes["verl"].host == "10.0.0.5"
    assert nodes["verl"].user == "ml" and nodes["verl"].port == 2222
    assert nodes["verl"].alias == "verl"


def test_load_nodes_from_list():
    nodes = load_nodes([{"alias": "a", "host": "h1"}, {"alias": "b", "host": "h2"}])
    assert set(nodes) == {"a", "b"} and nodes["b"].host == "h2"


def test_load_nodes_from_json_string():
    nodes = load_nodes(json.dumps({"verl": {"host": "h"}}))
    assert nodes["verl"].host == "h"


def test_load_nodes_from_env(monkeypatch):
    monkeypatch.setenv("REPRODUCEGYM_METAX_NODES", json.dumps({"x": {"host": "envhost"}}))
    nodes = load_nodes()
    assert nodes["x"].host == "envhost"


def test_load_nodes_empty_when_unset(monkeypatch):
    monkeypatch.delenv("REPRODUCEGYM_METAX_NODES", raising=False)
    assert load_nodes() == {}


def test_ssh_command_basic():
    node = MetaxNode(alias="verl", host="1.2.3.4", user="root")
    cmd = ssh_command(node, "nvidia-smi")
    assert cmd[0] == "ssh"
    assert cmd[-2:] == ["root@1.2.3.4", "nvidia-smi"]


def test_ssh_command_with_port_and_key():
    node = MetaxNode(alias="verl", host="h", user="u", port=2200, key_path="/k/id")
    cmd = ssh_command(node, "ls")
    assert "-p" in cmd and "2200" in cmd
    assert "-i" in cmd and "/k/id" in cmd


def test_nodes_to_env_roundtrip():
    nodes = load_nodes({"verl": {"host": "h", "user": "u"}})
    restored = load_nodes(nodes_to_env(nodes))
    assert restored["verl"].host == "h" and restored["verl"].user == "u"
