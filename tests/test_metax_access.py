"""R7: MetaX access wiring -- config, compute-access rendering, in-sandbox wrapper."""

from __future__ import annotations

import json

import pytest

from reproducegym.metax import (
    METAX_SSH_SCRIPT,
    install_compute_access,
    load_metax_config,
    load_nodes,
    render_compute_access,
)


def _nodes():
    return load_nodes(
        {
            "verl-grpo": {"host": "10.0.0.10", "user": "root", "workdir": "/ws/verl"},
            "verl-pool": {"host": "10.0.0.11", "user": "ml", "port": 2222},
        }
    )


def test_render_compute_access_lists_aliases_and_wrapper():
    md = render_compute_access(
        _nodes(),
        launch_template="cd {workdir} && bash run.sh",
        notes="activate env first",
        remote_workdir="/ws/verl",
    )
    assert "metax_ssh.py" in md
    assert "verl-grpo" in md and "verl-pool" in md
    assert "root@10.0.0.10" in md
    assert "cd {workdir} && bash run.sh" in md
    assert "activate env first" in md
    assert "/ws/verl" in md


def test_metax_ssh_script_is_valid_python_and_self_contained():
    compile(METAX_SSH_SCRIPT, "metax_ssh.py", "exec")
    # must not depend on reproducegym; resolves alias from sibling json + execs ssh
    assert "import reproducegym" not in METAX_SSH_SCRIPT
    assert "metax_nodes.json" in METAX_SSH_SCRIPT
    assert "os.execvp" in METAX_SSH_SCRIPT


def test_install_compute_access_writes_files_and_appends_task_md(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "task.md").write_text("# original task\n", encoding="utf-8")

    written = install_compute_access(
        ws, _nodes(), launch_template="bash run.sh", notes="n"
    )

    inv = ws / "metax_nodes.json"
    wrapper = ws / "metax_ssh.py"
    doc = ws / "compute_access.md"
    assert {inv, wrapper, doc} <= set(written)

    parsed = json.loads(inv.read_text())
    assert parsed["verl-grpo"]["host"] == "10.0.0.10"

    assert wrapper.read_text() == METAX_SSH_SCRIPT
    assert wrapper.stat().st_mode & 0o111  # executable

    task = (ws / "task.md").read_text()
    assert task.startswith("# original task")
    assert "Compute access" in task and "metax_ssh.py" in task


def test_install_compute_access_no_task_md_is_ok(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    written = install_compute_access(ws, _nodes())
    assert (ws / "compute_access.md") in written
    assert not (ws / "task.md").exists()


def test_load_metax_config_missing_returns_empty(tmp_path):
    assert load_metax_config(tmp_path / "nope.yaml") == {}


def test_load_metax_config_reads_yaml(tmp_path):
    cfg = tmp_path / "metax_nodes.yaml"
    cfg.write_text(
        "nodes:\n"
        "  verl:\n"
        "    host: 1.2.3.4\n"
        "    user: root\n"
        "    workdir: /ws\n"
        "remote_workdir: /ws\n"
        "launch_template: |\n"
        "  cd /ws && bash run.sh\n"
        "notes: be careful\n",
        encoding="utf-8",
    )
    loaded = load_metax_config(cfg)
    assert loaded["nodes"]["verl"].host == "1.2.3.4"
    assert loaded["remote_workdir"] == "/ws"
    assert "bash run.sh" in loaded["launch_template"]
    assert loaded["notes"] == "be careful"


def test_load_metax_config_via_env(tmp_path, monkeypatch):
    cfg = tmp_path / "m.yaml"
    cfg.write_text("nodes:\n  v:\n    host: h\n", encoding="utf-8")
    monkeypatch.setenv("REPRODUCEGYM_METAX_CONFIG", str(cfg))
    loaded = load_metax_config()
    assert loaded["nodes"]["v"].host == "h"
