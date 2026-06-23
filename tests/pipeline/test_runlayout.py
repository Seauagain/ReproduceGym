"""Run-directory layout: stage paths, attempt numbering, and the index."""

from __future__ import annotations

import json

from reproducegym.runlayout import (
    EXTRACT,
    PARSE,
    RUN,
    SPEC,
    TASK,
    PaperLayout,
    build_manifest,
    render_readme,
    write_index,
    write_run_record,
)


def test_parse_stage_paths(tmp_path):
    layout = PaperLayout.for_paper(tmp_path, "dr-grpo")
    assert layout.parse_dir == tmp_path / "dr-grpo" / PARSE
    assert layout.parsed_paper_path == tmp_path / "dr-grpo" / PARSE / "paper.md"
    assert layout.parsed_figures_dir == tmp_path / "dr-grpo" / PARSE / "figures"
    assert layout.figure_index_path == tmp_path / "dr-grpo" / PARSE / "figures.index.json"
    assert layout.claims_md_path == tmp_path / "dr-grpo" / "CLAIMS.md"


def test_manifest_reports_parsed_bundle(tmp_path):
    layout = PaperLayout.for_paper(tmp_path, "p")
    layout.parse_dir.mkdir(parents=True)
    assert build_manifest(layout, paper_id="p")["has_parsed_bundle"] is False
    layout.parsed_paper_path.write_text("# paper\n")
    m = build_manifest(layout, paper_id="p")
    assert m["has_parsed_bundle"] is True
    assert f"`{PARSE}/`" in render_readme(m)


def test_stage_paths(tmp_path):
    layout = PaperLayout.for_paper(tmp_path, "dr-grpo")
    assert layout.extract_dir == tmp_path / "dr-grpo" / EXTRACT
    assert layout.spec_dir == tmp_path / "dr-grpo" / SPEC
    assert layout.task_dir("c1") == tmp_path / "dr-grpo" / TASK / "c1"
    assert layout.task_dir("c1", "abc123") == tmp_path / "dr-grpo" / TASK / "c1" / "abc123"
    assert layout.spec_path("c1") == tmp_path / "dr-grpo" / SPEC / "c1.yaml"
    assert layout.spec_path("c1", "abc123") == tmp_path / "dr-grpo" / SPEC / "c1.abc123.yaml"
    assert layout.run_base("c1") == tmp_path / "dr-grpo" / RUN / "c1"
    assert layout.run_base("c1", "abc123") == tmp_path / "dr-grpo" / RUN / "c1" / "abc123"


def test_next_run_dir_increments(tmp_path):
    layout = PaperLayout.for_paper(tmp_path, "p")
    first = layout.next_run_dir("c1")
    assert first.name == "001"
    first.mkdir(parents=True)
    second = layout.next_run_dir("c1")
    assert second.name == "002"


def test_from_task_dir_roundtrip(tmp_path):
    layout = PaperLayout.for_paper(tmp_path, "p")
    task_dir = layout.task_dir("c1")
    recovered = PaperLayout.from_task_dir(task_dir)
    assert recovered is not None
    assert recovered.root == layout.root
    hashed = layout.task_dir("c1", "abc123")
    recovered_hash = PaperLayout.from_task_dir(hashed)
    assert recovered_hash is not None
    assert recovered_hash.root == layout.root
    # A non-layout path yields None.
    assert PaperLayout.from_task_dir(tmp_path / "random" / "c1") is None


def _seed_layout(tmp_path):
    layout = PaperLayout.for_paper(tmp_path, "p")
    layout.extract_dir.mkdir(parents=True)
    (layout.extract_dir / "candidate_claims.json").write_text(
        json.dumps([{"claim_id": "c1"}, {"claim_id": "c2"}])
    )
    (layout.extract_dir / "selected_claims.json").write_text(
        json.dumps([
            {
                "claim_id": "c1",
                "display_title": "Claim one",
                "statement": "Claim one is reproducible.",
                "claim_type": "mechanism",
                "cost": "S",
                "requires_training": False,
                "verifiability": "high",
                "selection_rank": 1,
                "selection_score": 9.0,
                "selection_reason": "cheap and diagnostic",
                "required_experiments": ["run the proxy eval"],
                "intermediate_steps": ["prepare data", "compute metric"],
                "implementation_notes": "Use the released config.",
            }
        ])
    )
    (layout.extract_dir / "claims.json").write_text(
        json.dumps([{"claim_id": "c1"}])
    )
    layout.spec_dir.mkdir(parents=True)
    (layout.spec_path("c1")).write_text("name: c1\n")
    layout.task_dir("c1").mkdir(parents=True)
    (layout.task_dir("c1") / "data_entry.json").write_text("{}")
    layout.task_dir("c2").mkdir(parents=True)
    (layout.task_dir("c2") / "data_entry.json").write_text("{}")
    run_dir = layout.next_run_dir("c1")
    write_run_record(run_dir, {"status": "scored", "reward": 0.8, "backend": "claude-code"})
    return layout


def test_build_manifest_scans_tree(tmp_path):
    layout = _seed_layout(tmp_path)
    m = build_manifest(layout, paper_id="p")
    assert m["paper_id"] == "p"
    assert m["claims"] == ["c1"]
    assert m["candidate_claims"] == ["c1", "c2"]
    assert m["specs"] == ["c1"]
    assert m["tasks"] == ["c1"]
    assert m["stale_tasks"] == ["c2"]
    assert m["claim_table"][0]["display_title"] == "Claim one"
    assert m["claim_table"][0]["task"] == "c1"
    assert m["runs"]["c1"][0]["reward"] == 0.8
    assert m["runs"]["c1"][0]["id"] == "001"


def test_write_index_emits_readme_and_manifest(tmp_path):
    layout = _seed_layout(tmp_path)
    manifest = write_index(layout, paper_id="p")
    assert layout.manifest_path.is_file()
    assert layout.readme_path.is_file()
    assert layout.claims_md_path.is_file()
    readme = layout.readme_path.read_text()
    assert "Reproduction run: `p`" in readme
    assert "CLAIMS.md" in readme
    assert "best reward" in readme
    assert "0.8" in readme
    claims_md = layout.claims_md_path.read_text()
    assert "Claim one" in claims_md
    assert "run the proxy eval" in claims_md
    # manifest on disk matches the returned one's structure
    on_disk = json.loads(layout.manifest_path.read_text())
    assert on_disk["claims"] == manifest["claims"]


def test_render_readme_no_runs(tmp_path):
    layout = PaperLayout.for_paper(tmp_path, "empty")
    layout.root.mkdir(parents=True)
    readme = render_readme(build_manifest(layout, paper_id="empty"))
    assert "No attempts yet" in readme
