"""R5: CLI argparse wiring + offline subcommands."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reprogym import cli
from reprogym.pipeline.render_check import write_baseline_check
from reprogym.pipeline.render_task import render_task


def test_parser_reproduce_args():
    args = cli.build_parser().parse_args(["reproduce", "p.md", "--claim", "c1", "--no-score"])
    assert args.command == "reproduce" and args.paper == "p.md"
    assert args.claim == "c1" and args.no_score is True


def test_parser_reproduce_compute_node():
    args = cli.build_parser().parse_args(
        ["reproduce", "p.md", "--compute", "lbg:project=1", "--node", "verl-grpo-44487"]
    )
    assert args.compute == "lbg:project=1"
    assert args.node == "verl-grpo-44487"


def test_no_command_errors():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_reproduce_dispatch(monkeypatch, capsys):
    captured = {}

    def fake_reproduce(paper, claim, **kw):
        captured["paper"] = paper
        captured["claim"] = claim
        captured["kw"] = kw
        return SimpleNamespace(claim_id="c1", task_dir="/t", trajectory_path="/tr", reward=0.8)

    monkeypatch.setattr(cli, "reproduce", fake_reproduce)
    rc = cli.main(["reproduce", "paper.md", "--claim", "c1", "--no-score"])
    assert rc == 0
    assert captured["paper"] == "paper.md" and captured["claim"] == "c1"
    assert captured["kw"]["do_score"] is False
    assert "reward:     0.8" in capsys.readouterr().out


def test_build_dispatch(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "build_task",
        lambda paper, claim, **kw: SimpleNamespace(claim_id="c1", task_dir="/t", validation=[]),
    )
    assert cli.main(["build", "paper.md"]) == 0
    assert "validation: ok" in capsys.readouterr().out


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
    md.write_text("# hi", encoding="utf-8")
    rc = cli.main(["parse", str(md), "-o", str(tmp_path / "out")])
    assert rc == 0
    assert (tmp_path / "out" / "paper.md").read_text() == "# hi"


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
