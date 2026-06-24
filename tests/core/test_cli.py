"""R5: CLI argparse wiring + offline subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reproducegym import cli
from reproducegym.pipeline.render_check import write_baseline_check
from reproducegym.pipeline.render_task import render_task


def test_parser_reproduce_args():
    args = cli.build_parser().parse_args(["reproduce", "runs/p/03-task/c001/h", "--no-score"])
    assert args.command == "reproduce" and args.task_dir == "runs/p/03-task/c001/h"
    assert args.no_score is True


def test_parser_reproduce_compute_node():
    args = cli.build_parser().parse_args(
        ["reproduce", "task", "--compute", "lbg:project=1", "--node", "verl-grpo-44487"]
    )
    assert args.compute == "lbg:project=1"
    assert args.node == "verl-grpo-44487"


def test_no_command_errors():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_reproduce_dispatch(monkeypatch, capsys, tmp_path):
    captured = {}
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_launch(task_dir, rd, **kw):
        captured["task_dir"] = task_dir
        captured["run_dir"] = rd
        captured["kw"] = kw
        return SimpleNamespace(
            task_dir=Path("/t"),
            run_dir=run_dir,
            workspace=Path("/w"),
            backend=SimpleNamespace(name="claude-code"),
            metadata={"claim_id": "c1", "spec_hash": "abc"},
        )

    monkeypatch.setattr(cli, "launch", fake_launch)
    monkeypatch.setattr(
        cli,
        "run_guarded",
        lambda runtime, timeout=None: SimpleNamespace(
            trajectory_path="/tr", returncode=0, session_id="s1"
        ),
    )
    rc = cli.main(["reproduce", "task-dir", "--no-score"])
    assert rc == 0
    assert captured["task_dir"] == "task-dir"
    assert "reward:     None" in capsys.readouterr().out
    # the run is now persisted like run.py so it lands in manifest scans
    rec = json.loads((run_dir / "run.json").read_text())
    assert rec["claim_id"] == "c1" and rec["spec_hash"] == "abc"
    assert rec["status"] == "ran" and rec["returncode"] == 0


def test_build_dispatch(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        cli, "build_claim_tasks",
        lambda **kw: seen.update(kw) or {"paper_id": "p", "built": [{"claim_id": "c1", "task_dir": "/t"}]},
    )
    assert cli.main(["build", "paper.md", "--refresh-claims"]) == 0
    assert seen["max_claims"] == 3
    assert seen["refresh_claims"] is True
    assert '"claim_id": "c1"' in capsys.readouterr().out


def test_triage_dispatch(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "_make_client", lambda: object())
    monkeypatch.setattr(cli, "extract_claims", lambda text, client: [{"claim_id": "c1"}])
    monkeypatch.setattr(cli, "triage", lambda claims, client, out_dir: {"build": ["c1"], "v0": "c1"})
    monkeypatch.setattr(cli, "write_resource_profile", lambda claims, out_dir: tmp_path / "rp.yaml")
    rc = cli.main(["triage", "paper.md", "--out-dir", str(tmp_path)])
    assert rc == 0
    assert "v0:      c1" in capsys.readouterr().out


def test_parse_md_passthrough(tmp_path, capsys):
    md = tmp_path / "in.md"
    (tmp_path / "fig1.png").write_bytes(b"img")
    md.write_text("# hi\n![Fig. 1](fig1.png)", encoding="utf-8")
    runs = tmp_path / "runs"
    rc = cli.main(["parse", "--md", str(md), "--out", str(runs), "--paper-id", "in"])
    assert rc == 0
    parse_dir = runs / "in" / "00-parse"
    assert (parse_dir / "paper.md").read_text() == "# hi\n![Fig. 1](fig1.png)"
    assert (parse_dir / "figures" / "fig1.png").is_file()
    assert (parse_dir / "figures.index.json").is_file()


def test_parser_parse_requires_one_source():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["parse"])  # none of --url/--pdf/--md
    args = cli.build_parser().parse_args(["parse", "--url", "2503.20783"])
    assert args.url == "2503.20783" and args.command == "parse"


def test_dataset_offline(tmp_path, valid_claim_spec, capsys):
    t1 = render_task(valid_claim_spec, tmp_path / "p" / "tasks" / "c1")
    rc = cli.main(
        ["dataset", "s", "--task", str(t1), "--datasets-root", str(tmp_path / "ds")]
    )
    assert rc == 0
    ds = (tmp_path / "ds" / "s")
    assert ds.is_dir() and sum(1 for _ in ds.iterdir()) == 1


def test_score_offline(tmp_path, valid_claim_spec, capsys):
    task_dir = render_task(valid_claim_spec, tmp_path / "task")
    write_baseline_check(valid_claim_spec, task_dir / "reward")
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    rows = ["condition,step,len"]
    rows += [f"baseline,{i},100" for i in range(50)]
    rows += [f"treatment,{i},70" for i in range(50)]
    (ws / "output" / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    # result.json carries no verdict; scoring recomputes from metrics.csv.
    (ws / "output" / "result.json").write_text("{}", encoding="utf-8")
    rc = cli.main(["score", str(task_dir), str(ws)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "0.8"
