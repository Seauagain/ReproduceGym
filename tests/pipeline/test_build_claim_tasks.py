from __future__ import annotations

import json
from pathlib import Path

import pytest

import reproducegym.config as config
from reproducegym.models import multimodal_figure_configured
from reproducegym.pipeline.build_claim_tasks import (
    _resolve_paper_input,
    build_claim_tasks,
    should_parse_images,
)
from reproducegym.pipeline.claim_selection import rank_claims
from tests.helpers import FakeVL, make_parse_bundle


CLAIMS_JSON = json.dumps(
    [
        {
            "statement": "Removing std normalization removes the length bias.",
            "claim_type": "mechanism",
            "display_title": "std bias",
            "importance_rank": 1,
            "metrics": [{"name": "len_ratio", "formula": "a/b", "direction": "lower_is_better"}],
            "anchors": [{"kind": "figure", "ref": "Fig. 1"}],
        }
    ]
)

FIG_EVIDENCE = json.dumps(
    {
        "figure_ref": "Fig. 1",
        "params": [{"name": "policy_steps", "value": 150, "visibility": "visible"}],
        "targets": [{"name": "len_ratio_target", "value": 0.42}],
        "confidence": 0.9,
    }
)


class SeqLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def complete(self, prompt: str, **_):
        self.calls.append(prompt)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def test_rank_claims_balances_importance_and_economy():
    claims = [
        {"statement": "headline", "claim_type": "headline", "importance_rank": 1, "cost": "XL", "verifiability": "medium", "requires_training": True},
        {"statement": "cheap mechanism", "claim_type": "mechanism", "importance_rank": 1, "cost": "S", "verifiability": "high", "requires_training": False},
        {"statement": "less important", "claim_type": "ablation", "importance_rank": 2, "cost": "S", "verifiability": "high", "requires_training": False},
    ]
    ordered = rank_claims(claims)
    assert [c["statement"] for c in ordered] == ["cheap mechanism", "less important", "headline"]
    assert [c["selection_rank"] for c in ordered] == [1, 2, 3]
    assert ordered[0]["selection_score"] > ordered[-1]["selection_score"]


def test_parse_images_auto_requires_figures_and_config():
    assert should_parse_images("auto", has_figures=False, configured=True) is False
    assert should_parse_images("auto", has_figures=True, configured=False) is False
    assert should_parse_images("auto", has_figures=True, configured=True) is True


def test_parse_images_always_requires_figures_and_config():
    with pytest.raises(ValueError):
        should_parse_images("always", has_figures=False, configured=True)
    with pytest.raises(ValueError):
        should_parse_images("always", has_figures=True, configured=False)
    assert should_parse_images("always", has_figures=True, configured=True) is True


def test_parse_images_never_skips_even_when_available():
    assert should_parse_images("never", has_figures=True, configured=True) is False


def test_multimodal_configured_accepts_generic_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "MULTIMODAL_API_KEY=x\nMULTIMODAL_BASE_URL=x\nMULTIMODAL_VISION_MODEL=x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DEFAULT_ENV_PATH", env)
    assert multimodal_figure_configured() is True


def test_resolve_paper_input_bundle_and_raw(tmp_path):
    bundle = make_parse_bundle(tmp_path)
    paper_md, figs, index, pid = _resolve_paper_input(bundle, None)
    assert paper_md == bundle / "00-parse" / "paper.md"
    assert figs == bundle / "00-parse" / "figures"
    assert index == bundle / "00-parse" / "figures.index.json"
    assert pid == "paper1"
    # a raw paper.md resolves figures from its sibling figures/ dir
    raw = tmp_path / "raw"
    (raw / "figures").mkdir(parents=True)
    (raw / "paper.md").write_text("# x", encoding="utf-8")
    pmd, figs2, idx2, pid2 = _resolve_paper_input(raw / "paper.md", None)
    assert pmd == raw / "paper.md" and figs2 == raw / "figures" and idx2 is None and pid2 == "paper"


def test_build_consumes_parse_bundle_and_resolves_figures(tmp_path, make_llm):
    bundle = make_parse_bundle(tmp_path)
    out = tmp_path / "out"
    res = build_claim_tasks(
        paper=bundle,
        out=out,
        parse_images="always",
        claude_client=make_llm(CLAIMS_JSON),
        multimodal_client=FakeVL({"fig1.jpg": FIG_EVIDENCE}, default="{}"),
    )
    assert res["paper_id"] == "paper1" and res["image_evidence"] is True
    assert len(res["built"]) == 1
    task_dir = Path(res["built"][0]["task_dir"])
    extract_dir = out / "paper1" / "01-extract"
    assert (extract_dir / "claim_candidates.raw.json").is_file()
    assert (extract_dir / "claim_candidates.dedup.json").is_file()
    assert (extract_dir / "candidate_claims.json").is_file()
    assert (extract_dir / "selected_claims.json").is_file()
    assert (extract_dir / "claim_selection.json").is_file()
    assert (extract_dir / "claim_figure_evidence.index.json").is_file()
    assert any((extract_dir / "claim_figure_evidence").glob("*.yaml"))
    # parsed figures remain in 00-parse; build/task stages do not copy image bytes
    assert (bundle / "00-parse" / "figures" / "fig1.jpg").is_file()
    assert not (extract_dir / "figures" / "fig1.jpg").exists()
    assert not (extract_dir / "figure_vl_raw").exists()
    assert not (task_dir / "input_files" / "figures" / "fig1.jpg").exists()
    # visible figure param reaches params.yaml; the hidden target does not leak
    params = (task_dir / "input_files" / "params.yaml").read_text()
    assert "policy_steps" in params
    assert "len_ratio_target" not in params


def test_build_limits_rendered_tasks_to_max_claims(tmp_path, make_llm):
    bundle = make_parse_bundle(tmp_path)
    claims = json.dumps(
        [
            {"statement": "first", "claim_type": "mechanism", "importance_rank": 1, "cost": "S", "verifiability": "high", "metrics": [{"name": "a", "formula": "a", "direction": "higher_is_better"}]},
            {"statement": "second", "claim_type": "ablation", "importance_rank": 2, "cost": "S", "verifiability": "high", "metrics": [{"name": "b", "formula": "b", "direction": "higher_is_better"}]},
            {"statement": "third", "claim_type": "headline", "importance_rank": 3, "cost": "XL", "verifiability": "medium", "metrics": [{"name": "c", "formula": "c", "direction": "higher_is_better"}]},
        ]
    )
    res = build_claim_tasks(
        paper=bundle,
        out=tmp_path / "out",
        parse_images="never",
        claude_client=make_llm(claims),
        max_claims=2,
    )
    assert res["n_candidate_claims"] == 3
    assert res["n_selected_claims"] == 2
    assert len(res["built"]) == 2
    extract_dir = tmp_path / "out" / "paper1" / "01-extract"
    assert len(json.loads((extract_dir / "candidate_claims.json").read_text())) == 3
    assert len(json.loads((extract_dir / "selected_claims.json").read_text())) == 2


def test_build_limits_vl_reads_to_selected_claims(tmp_path, make_llm):
    bundle = make_parse_bundle(tmp_path)
    claims = json.dumps(
        [
            {"statement": "first", "claim_type": "mechanism", "importance_rank": 1, "cost": "S", "verifiability": "high", "anchors": [{"kind": "figure", "ref": "Fig. 1"}], "metrics": [{"name": "a", "formula": "a", "direction": "higher_is_better"}]},
            {"statement": "second", "claim_type": "ablation", "importance_rank": 2, "cost": "S", "verifiability": "high", "anchors": [{"kind": "figure", "ref": "Fig. 1"}], "metrics": [{"name": "b", "formula": "b", "direction": "higher_is_better"}]},
            {"statement": "third", "claim_type": "headline", "importance_rank": 3, "cost": "XL", "verifiability": "medium", "anchors": [{"kind": "figure", "ref": "Fig. 1"}], "metrics": [{"name": "c", "formula": "c", "direction": "higher_is_better"}]},
        ]
    )
    vl = FakeVL({"fig1.jpg": FIG_EVIDENCE}, default="{}")
    res = build_claim_tasks(
        paper=bundle,
        out=tmp_path / "out",
        parse_images="always",
        claude_client=make_llm(claims),
        multimodal_client=vl,
        max_claims=1,
    )
    extract_dir = tmp_path / "out" / "paper1" / "01-extract"
    evidence_index = json.loads((extract_dir / "claim_figure_evidence.index.json").read_text())
    assert res["n_candidate_claims"] == 3
    assert res["n_selected_claims"] == 1
    assert len(vl.calls) == 1
    assert len(evidence_index) == 1


def test_refresh_claims_ignores_cached_candidates(tmp_path):
    bundle = make_parse_bundle(tmp_path)
    out = tmp_path / "out"
    first = json.dumps([
        {"statement": "old", "claim_type": "mechanism", "importance_rank": 1, "metrics": [{"name": "old", "formula": "old", "direction": "higher_is_better"}]},
    ])
    second = json.dumps([
        {"statement": "new", "claim_type": "mechanism", "importance_rank": 1, "metrics": [{"name": "new", "formula": "new", "direction": "higher_is_better"}]},
    ])
    build_claim_tasks(paper=bundle, out=out, parse_images="never", claude_client=SeqLLM([first]), max_claims=1)
    build_claim_tasks(
        paper=bundle,
        out=out,
        parse_images="never",
        claude_client=SeqLLM([second]),
        max_claims=1,
        refresh_claims=True,
    )
    selected = json.loads((out / "paper1" / "01-extract" / "selected_claims.json").read_text())
    assert selected[0]["statement"] == "new"


def test_build_raw_md_warns_when_figures_unresolved(tmp_path, make_llm, capsys):
    # markdown references a local image that does not exist -> auto skips, but warns
    paper = tmp_path / "paper.md"
    paper.write_text("# P\n\n![](images/missing.jpg)\n", encoding="utf-8")
    build_claim_tasks(
        paper=paper,
        out=tmp_path / "out",
        parse_images="auto",
        claude_client=make_llm(CLAIMS_JSON),
    )
    assert "has image references but none resolved" in capsys.readouterr().out
