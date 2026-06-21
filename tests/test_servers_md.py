"""S1: parse servers.md into a sanitized ssh node inventory (no secret leakage)."""

from __future__ import annotations

from reprogym.compute.servers_md import parse_servers_md
from reprogym.metax import nodes_to_env

# A servers.md-shaped fixture: a proper per-node yaml block + a PROSE backup
# password line (exactly how the real file stores passwords) + a shared block
# without an alias (must be ignored).
SAMPLE = """
## verl-grpo-44487

```ssh-config
Host verl-grpo-44487
  Hostname 106.75.252.110
  Port 44487
  User root
```

```yaml
alias:       verl-grpo-44487
cluster:     metax-cluster
hostname:    106.75.252.110
port:        44487
user:        root
auth:        publickey
python_exe:  /opt/conda/bin/python
launcher:    nohup
workdir:     /mnt/public/code/hlk/xuzhiqin
path_fix:    "export PATH=/opt/conda/bin:$PATH"
```

> 备用密码登录（公钥失效时）：`Xg6UAF&Z`（建议失效后删）。

## metax-cluster (shared, no alias -> ignored)

```yaml
cluster:     metax-cluster
hostname:    106.75.252.110
user:        root
```
"""


def test_parses_node_with_host_port_user():
    nodes = parse_servers_md(SAMPLE)
    assert set(nodes) == {"verl-grpo-44487"}
    n = nodes["verl-grpo-44487"]
    assert n.host == "106.75.252.110"
    assert n.port == 44487
    assert n.user == "root"
    assert n.workdir == "/mnt/public/code/hlk/xuzhiqin"


def test_no_password_leaks_into_inventory():
    nodes = parse_servers_md(SAMPLE)
    blob = nodes_to_env(nodes)
    assert "Xg6UAF&Z" not in blob          # prose password never parsed
    assert nodes["verl-grpo-44487"].key_path is None  # publickey only, no key material


def test_block_without_alias_or_host_is_skipped():
    nodes = parse_servers_md(SAMPLE)
    assert "metax-cluster" not in nodes


def test_empty_or_no_blocks_returns_empty():
    assert parse_servers_md("# just prose, no yaml blocks") == {}


def test_real_servers_md_if_present():
    # Opportunistic: if the real registry is reachable, it must parse without
    # leaking any of the known backup passwords.
    import pathlib

    p = pathlib.Path(__file__).resolve().parents[2] / "servers.md"
    if not p.is_file():
        return
    nodes = parse_servers_md(p.read_text(encoding="utf-8"))
    blob = nodes_to_env(nodes)
    for leaked in ["Xg6UAF&Z", "MfSwAT~G2n9", "gP2^oDm#I"]:
        assert leaked not in blob
    # the default verl node should be discoverable
    assert any("verl-grpo" in alias for alias in nodes)
