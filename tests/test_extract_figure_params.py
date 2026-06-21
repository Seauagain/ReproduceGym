"""R2: read experimental params off figures via an injected Qwen-VL client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reprogym.pipeline.extract_figure_params import (
    FigureParamError,
    extract_figure_params,
    parse_figure_params_json,
)
from reprogym.pipeline.merge_claim_spec import merge_claim_spec


class FakeVL:
    def __init__(self, by_name: dict[str, str]):
        self.by_name = by_name
        self.calls: list[tuple[str, str]] = []

    def read_figure(self, image_path, prompt):
        self.calls.append((Path(image_path).name, prompt))
        return self.by_name.get(Path(image_path).name, "[]")


def _figs(tmp_path, names):
    d = tmp_path / "figures"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"img")
    return d


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
    figs = _figs(tmp_path, ["fig1.jpg", "fig2.png"])
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


def test_malformed_skipped_unless_strict(tmp_path):
    figs = _figs(tmp_path, ["fig1.jpg", "fig2.png"])
    vl = FakeVL({"fig1.jpg": json.dumps([{"name": "a", "value": 1}]), "fig2.png": "not json"})
    out = extract_figure_params(figs, client=vl)
    assert set(out) == {"a"}
    with pytest.raises((FigureParamError, json.JSONDecodeError)):
        extract_figure_params(figs, client=vl, strict=True)


def test_min_confidence_filters(tmp_path):
    figs = _figs(tmp_path, ["fig1.jpg"])
    vl = FakeVL({"fig1.jpg": json.dumps([
        {"name": "a", "value": 1, "confidence": 0.9},
        {"name": "b", "value": 2, "confidence": 0.2},
    ])})
    out = extract_figure_params(figs, client=vl, min_confidence=0.5)
    assert set(out) == {"a"}


def test_dedup_keeps_higher_confidence(tmp_path):
    figs = _figs(tmp_path, ["fig1.jpg", "fig2.png"])
    vl = FakeVL({
        "fig1.jpg": json.dumps([{"name": "x", "value": 1, "confidence": 0.4}]),
        "fig2.png": json.dumps([{"name": "x", "value": 2, "confidence": 0.95}]),
    })
    out = extract_figure_params(figs, client=vl)
    assert out["x"]["value"] == 2


def test_empty_dir_returns_empty(tmp_path):
    assert extract_figure_params(tmp_path / "nope", client=FakeVL({})) == {}


def test_out_path_written(tmp_path):
    figs = _figs(tmp_path, ["fig1.jpg"])
    vl = FakeVL({"fig1.jpg": json.dumps([{"name": "a", "value": 1}])})
    out_path = tmp_path / "figure_params.yaml"
    extract_figure_params(figs, client=vl, out_path=out_path)
    loaded = yaml.safe_load(out_path.read_text())
    assert loaded["a"]["value"] == 1


def test_feeds_merge_claim_spec(tmp_path):
    figs = _figs(tmp_path, ["fig5.jpg"])
    vl = FakeVL({"fig5.jpg": json.dumps([{"name": "group_size", "value": 8, "confidence": 0.9}])})
    fp = extract_figure_params(figs, client=vl)
    claim = {"claim_id": "c1", "statement": "s", "claim_type": "mechanism",
             "metrics": [{"name": "m", "formula": "f", "direction": "higher_is_better"}]}
    spec = merge_claim_spec(claim, figure_params=fp, paper_id="p")
    params = {p["name"]: p for p in spec["params"]}
    assert params["group_size"]["exposure"] == "hidden"
