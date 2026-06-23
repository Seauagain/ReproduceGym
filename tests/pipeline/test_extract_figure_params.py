"""R2: read experimental params off figures via an injected multimodal client."""

from __future__ import annotations

import json

import pytest
import yaml

from reproducegym.pipeline.extract_figure_params import (
    FigureParamError,
    extract_claim_figure_evidence,
    extract_figure_evidence,
    extract_figure_params,
    figures_for_claim,
    parse_figure_params_json,
)
from reproducegym.pipeline.merge_claim_spec import merge_claim_spec
from tests.helpers import FakeVL, make_figures


def test_parse_tags_hidden_and_source():
    raw = json.dumps([{"name": "group_size", "value": 8, "confidence": 0.9}])
    entries = parse_figure_params_json(raw, default_source="Fig. 5")
    e = entries[0]
    assert e["status"] == "paper_specified" and e["exposure"] == "hidden"
    assert e["source"] == "Fig. 5" and e["value"] == 8


def test_parse_strips_fence_and_unwraps():
    raw = "```json\n" + json.dumps({"params": [{"name": "lr", "value": 1e-6}]}) + "\n```"
    assert parse_figure_params_json(raw, default_source="Fig. 1")[0]["name"] == "lr"


def test_merge_across_figures(tmp_path):
    figs = make_figures(tmp_path, ["fig1.jpg", "fig2.png"])
    vl = FakeVL(
        {
            "fig1.jpg": json.dumps([{"name": "group_size", "value": 8}]),
            "fig2.png": json.dumps([{"name": "max_len", "value": 3000, "unit": "tokens"}]),
        }
    )
    out = extract_figure_params(figs, client=vl)
    assert set(out) == {"group_size", "max_len"}
    assert out["group_size"]["source"] == "Fig. 1"
    assert out["max_len"]["unit"] == "tokens" and out["max_len"]["exposure"] == "hidden"


def test_extract_evidence_passes_caption_context(tmp_path):
    figs = make_figures(tmp_path, ["fig5.png"])
    vl = FakeVL(
        {
            "fig5.png": json.dumps(
                {
                    "figure_ref": "Fig. 5",
                    "params": [
                        {
                            "name": "policy_iteration_steps",
                            "value": 150,
                            "visibility": "visible",
                            "confidence": 0.95,
                        }
                    ],
                    "axis_ranges": {"x": {"max": 150}},
                    "confidence": 0.9,
                }
            )
        }
    )
    evidence = extract_figure_evidence(
        figs,
        client=vl,
        figures_index=[
            {
                "figure_ref": "Fig. 5",
                "image_file": "fig5.png",
                "caption": "Figure 5: length over policy iteration step",
                "context": "Nearby text",
            }
        ],
    )

    assert evidence[0]["params"][0]["exposure"] == "visible"
    assert evidence[0]["axis_ranges"]["x"]["max"] == 150
    assert "Figure 5: length" in vl.calls[0][1]
    assert "Nearby text" in vl.calls[0][1]


def test_figures_for_claim_selects_only_anchored_refs():
    figures = [
        {"figure_ref": "Fig. 1", "image_file": "fig1.jpg"},
        {"figure_ref": "Fig. 2", "image_file": "fig2.jpg"},
    ]
    claim = {"anchors": [{"kind": "figure", "ref": "Fig. 2"}]}
    out = figures_for_claim(claim, figures)
    assert [f["image_file"] for f in out] == ["fig2.jpg"]


def test_figures_for_claim_matches_panels_not_prefixes():
    figures = [
        {"figure_ref": "Fig. 5(a)", "image_file": "fig5a.jpg"},
        {"figure_ref": "Fig. 15", "image_file": "fig15.jpg"},
    ]
    claim = {"anchors": [{"kind": "figure", "ref": "Fig. 5"}]}
    out = figures_for_claim(claim, figures)
    assert [f["image_file"] for f in out] == ["fig5a.jpg"]


def test_extract_claim_figure_evidence_reads_only_anchor_and_includes_claim(tmp_path):
    figs = make_figures(tmp_path, ["fig1.jpg", "fig2.jpg"])
    vl = FakeVL(
        {
            "fig2.jpg": json.dumps(
                {
                    "figure_ref": "Fig. 2",
                    "params": [{"name": "policy_steps", "value": 150, "visibility": "visible"}],
                    "confidence": 0.9,
                }
            )
        }
    )
    claim = {
        "claim_id": "c001_len",
        "statement": "Dr. GRPO reduces response length.",
        "claim_type": "mechanism",
        "anchors": [{"kind": "figure", "ref": "Fig. 2"}],
    }
    evidence = extract_claim_figure_evidence(
        claim,
        figs,
        client=vl,
        figures_index=[
            {"figure_ref": "Fig. 1", "image_file": "fig1.jpg", "caption": "other"},
            {"figure_ref": "Fig. 2", "image_file": "fig2.jpg", "caption": "length curve"},
        ],
    )
    assert [name for name, _prompt in vl.calls] == ["fig2.jpg"]
    assert "Dr. GRPO reduces response length" in vl.calls[0][1]
    assert evidence[0]["params"][0]["name"] == "policy_steps"


def test_malformed_skipped_unless_strict(tmp_path):
    figs = make_figures(tmp_path, ["fig1.jpg", "fig2.png"])
    vl = FakeVL({"fig1.jpg": json.dumps([{"name": "a", "value": 1}]), "fig2.png": "not json"})
    out = extract_figure_params(figs, client=vl)
    assert set(out) == {"a"}
    with pytest.raises((FigureParamError, json.JSONDecodeError)):
        extract_figure_params(figs, client=vl, strict=True)


def test_min_confidence_filters(tmp_path):
    figs = make_figures(tmp_path, ["fig1.jpg"])
    vl = FakeVL({"fig1.jpg": json.dumps([
        {"name": "a", "value": 1, "confidence": 0.9},
        {"name": "b", "value": 2, "confidence": 0.2},
    ])})
    out = extract_figure_params(figs, client=vl, min_confidence=0.5)
    assert set(out) == {"a"}


def test_dedup_keeps_higher_confidence(tmp_path):
    figs = make_figures(tmp_path, ["fig1.jpg", "fig2.png"])
    vl = FakeVL({
        "fig1.jpg": json.dumps([{"name": "x", "value": 1, "confidence": 0.4}]),
        "fig2.png": json.dumps([{"name": "x", "value": 2, "confidence": 0.95}]),
    })
    out = extract_figure_params(figs, client=vl)
    assert out["x"]["value"] == 2


def test_empty_dir_returns_empty(tmp_path):
    assert extract_figure_params(tmp_path / "nope", client=FakeVL({})) == {}


def test_out_path_written(tmp_path):
    figs = make_figures(tmp_path, ["fig1.jpg"])
    vl = FakeVL({"fig1.jpg": json.dumps([{"name": "a", "value": 1}])})
    out_path = tmp_path / "figure_params.yaml"
    extract_figure_params(figs, client=vl, out_path=out_path)
    loaded = yaml.safe_load(out_path.read_text())
    assert loaded["a"]["value"] == 1


def test_feeds_merge_claim_spec(tmp_path):
    figs = make_figures(tmp_path, ["fig5.jpg"])
    vl = FakeVL({"fig5.jpg": json.dumps([{"name": "group_size", "value": 8, "confidence": 0.9}])})
    fp = extract_figure_params(figs, client=vl)
    claim = {"claim_id": "c1", "statement": "s", "claim_type": "mechanism",
             "metrics": [{"name": "m", "formula": "f", "direction": "higher_is_better"}]}
    spec = merge_claim_spec(claim, figure_params=fp, paper_id="p")
    params = {p["name"]: p for p in spec["params"]}
    assert params["group_size"]["exposure"] == "hidden"
