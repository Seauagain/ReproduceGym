from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import reproducegym.config as config
from reproducegym.models import multimodal_figure_configured
from reproducegym.pipeline.build_claim_tasks import (
    _resolve_paper_input,
    _normalize_conditions,
    _normalize_claim_type,
    _normalize_refined_claim,
    _normalize_params,
    _normalize_thresholds,
    _verification_contract_from_claim,
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
            "metrics": [{"name": "len_ratio", "formula": "mean(treatment.len) / mean(baseline.len)", "direction": "lower_is_better"}],
            "anchors": [{"kind": "figure", "ref": "Fig. 1"}],
        }
    ]
)

CLAIM_WITH_NUMERIC_TARGET_JSON = json.dumps(
    [
        {
            "statement": "Treatment reaches the reported target.",
            "claim_type": "mechanism",
            "display_title": "targeted",
            "importance_rank": 1,
            "metrics": [{"name": "score", "formula": "mean(treatment.score)", "direction": "higher_is_better"}],
            "thresholds": [
                {
                    "metric": "score",
                    "pass_threshold": 0.7,
                    "target_value": 0.8,
                    "tolerance_abs": 0.1,
                    "source": "Table 1",
                    "target_evidence": {"source": "Table 1"},
                    "rationale": "Table 1 reports the score target.",
                }
            ],
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


def test_threshold_normalizer_accepts_llm_shapes():
    thresholds = _normalize_thresholds(
        [
            {
                "metric": "score",
                "pass_threshold": "0.82",
                "exposure": "primary",
                "target_evidence": "Table 1 reports 0.82",
                "tolerance_abs": None,
                "condition": "ignored",
            }
        ]
    )

    assert thresholds == [
        {
            "metric": "score",
            "pass_threshold": 0.82,
            "exposure": "hidden",
            "target_evidence": {"source": "Table 1 reports 0.82"},
        }
    ]


def test_threshold_normalizer_exposes_neutral_directional_thresholds():
    thresholds = _normalize_thresholds(
        [{"metric": "gap", "pass_threshold": 0.0, "exposure": "hidden", "target_value": 0.08}]
    )

    assert thresholds[0]["exposure"] == "visible"
    assert thresholds[0]["target_value"] == 0.08


def test_param_normalizer_accepts_llm_shapes():
    params = _normalize_params(
        [
            {
                "name": "batch_size",
                "value": 512,
                "use": "config",
                "confidence": "high",
                "extra": "ignored",
            }
        ]
    )

    assert params == [
        {
            "name": "batch_size",
            "value": 512,
            "status": "paper_specified",
            "use": "reproduction_param",
            "confidence": 0.85,
        }
    ]


def test_condition_normalizer_coerces_string_switches():
    conditions = _normalize_conditions(
        [
            {
                "label": "distill_qwen_1_5b",
                "description": "distilled model",
                "switches": "model=='DeepSeek-R1-Distill-Qwen-1.5B' and benchmark=='AIME_2024'",
            }
        ]
    )

    assert conditions == [
        {
            "label": "distill_qwen_1_5b",
            "description": "distilled model",
            "switches": {
                "expression": "model=='DeepSeek-R1-Distill-Qwen-1.5B' and benchmark=='AIME_2024'"
            },
        }
    ]


def test_contract_normalizer_sanitizes_formula_identifiers():
    contract = _verification_contract_from_claim(
        {
            "conditions": [
                {"label": "Oat-Zero-7B", "description": "Oat"},
                {"label": "4shot", "description": "four shot"},
            ],
            "metrics": [
                {
                    "name": "aime_2024_accuracy",
                    "formula": "mean(Oat-Zero-7B.AIME2024_accuracy)",
                    "direction": "higher_is_better",
                },
                {
                    "name": "4shot_avg",
                    "formula": "mean(4shot.accuracy)",
                    "direction": "higher_is_better",
                },
            ],
        }
    )

    assert contract["conditions"][0]["label"] == "oat_zero_7b"
    assert contract["conditions"][0]["source_label"] == "Oat-Zero-7B"
    assert contract["conditions"][1]["label"] == "c_4shot"
    assert contract["metrics"][0]["formula"] == "mean(oat_zero_7b.AIME2024_accuracy)"
    assert contract["metrics"][1]["name"] == "m_4shot_avg"
    assert contract["metrics"][1]["formula"] == "mean(c_4shot.accuracy)"


def test_normalize_refined_claim_preserves_evidence_uid():
    bundle = {
        "claim_uid": "clm_original",
        "figure_evidence": [],
    }
    refined = {
        "claim_uid": "clm_original",
        "statement": "Refined wording changes without changing evidence identity.",
        "evidence_anchors": [{"kind": "figure", "ref": "Fig. 5"}],
        "metrics": [{"name": "score", "formula": "mean(run.score)", "direction": "higher_is_better"}],
        "thresholds": [{"metric": "score", "pass_threshold": 0.7, "target_value": 0.8, "source": "Fig. 5"}],
    }

    out = _normalize_refined_claim(refined, bundle)

    assert out["claim_uid"] == "clm_original"


def test_claim_type_normalizer_maps_verifier_type_to_eval_only():
    assert _normalize_claim_type("directional_comparison") == "eval_only"
    assert _normalize_claim_type("ablation") == "ablation"


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
    assert Path(res["token_usage"]).is_file()
    assert Path(res["token_usage_summary"]).is_file()
    task_dir = Path(res["built"][0]["task_dir"])
    extract_dir = out / "paper1" / "01-extract"
    assert (extract_dir / "candidate_claims.json").is_file()
    assert (extract_dir / "paper_evidence_index.json").is_file()
    assert (extract_dir / "triaged_claims.json").is_file()
    assert (extract_dir / "target_points").is_dir()
    assert (extract_dir / "refined_claims.json").is_file()
    assert (extract_dir / "claim_verification_report.json").is_file()
    assert (extract_dir / "selected_claims_for_build.json").is_file()
    assert (out / "paper1" / "build_validation.json").is_file()
    assert (out / "paper1" / "task_manifest.json").is_file()
    assert not (extract_dir / "selected_claims.json").exists()
    assert not (extract_dir / "claims.json").exists()
    assert (extract_dir / "claim_selection.json").is_file()
    assert (extract_dir / "claim_evidence.index.json").is_file()
    assert any((extract_dir / "claim_evidence").glob("*.json"))
    # parsed figures remain in 00-parse; build/task stages do not copy image bytes
    assert (bundle / "00-parse" / "figures" / "fig1.jpg").is_file()
    assert not (extract_dir / "figures" / "fig1.jpg").exists()
    assert not (extract_dir / "figure_vl_raw").exists()
    assert not (task_dir / "input_files" / "figures" / "fig1.jpg").exists()
    # visible figure param reaches params.yaml; the hidden target does not leak
    params = (task_dir / "input_files" / "params.yaml").read_text()
    assert "policy_steps" in params
    assert "len_ratio_target" not in params
    targets = yaml.safe_load((task_dir / "reward" / "targets.yaml").read_text())
    assert targets["primary_thresholds"]["len_ratio"]["pass_threshold"] == 0.546
    assert targets["primary_thresholds"]["len_ratio"]["target_evidence"]["source"] == "Fig. 1"
    assert targets["verification"]["pool"] == "rlvr"
    selected = json.loads((extract_dir / "selected_claims_for_build.json").read_text())
    assert selected[0]["claim_uid"]
    assert selected[0]["contract_hash"]
    assert selected[0]["reward_curves"]
    validation = json.loads((out / "paper1" / "build_validation.json").read_text())
    assert validation["tasks"][0]["accepted"] is True
    assert validation["tasks"][0]["synthetic_selftests"]["target"]["reward"] == 1.0
    task_manifest = json.loads((out / "paper1" / "task_manifest.json").read_text())
    assert task_manifest["tasks"][0]["claim_id"] == selected[0]["claim_id"]
    assert task_manifest["tasks"][0]["task_dir"] == str(task_dir)


def test_build_limits_rendered_tasks_to_max_claims(tmp_path, make_llm):
    bundle = make_parse_bundle(tmp_path)
    claims = json.dumps(
        [
            {"statement": "first", "claim_type": "mechanism", "importance_rank": 1, "cost": "S", "verifiability": "high", "metrics": [{"name": "a", "formula": "mean(a)", "direction": "higher_is_better"}], "thresholds": [{"metric": "a", "pass_threshold": 0.7, "target_value": 0.8, "source": "Table 1", "target_evidence": {"source": "Table 1"}, "rationale": "Table 1 reports the target."}]},
            {"statement": "second", "claim_type": "ablation", "importance_rank": 2, "cost": "S", "verifiability": "high", "metrics": [{"name": "b", "formula": "mean(b)", "direction": "higher_is_better"}], "thresholds": [{"metric": "b", "pass_threshold": 0.7, "target_value": 0.8, "source": "Table 1", "target_evidence": {"source": "Table 1"}, "rationale": "Table 1 reports the target."}]},
            {"statement": "third", "claim_type": "headline", "importance_rank": 3, "cost": "XL", "verifiability": "medium", "metrics": [{"name": "c", "formula": "mean(c)", "direction": "higher_is_better"}], "thresholds": [{"metric": "c", "pass_threshold": 0.7, "target_value": 0.8, "source": "Table 1", "target_evidence": {"source": "Table 1"}, "rationale": "Table 1 reports the target."}]},
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
    assert len(json.loads((extract_dir / "selected_claims_for_build.json").read_text())) == 2


def test_expanding_max_claims_invalidates_stale_refined_cache(tmp_path, make_llm):
    bundle = make_parse_bundle(tmp_path)
    claims = json.dumps(
        [
            {"statement": "first", "claim_type": "mechanism", "importance_rank": 1, "cost": "S", "verifiability": "high", "metrics": [{"name": "a", "formula": "mean(a)", "direction": "higher_is_better"}], "thresholds": [{"metric": "a", "pass_threshold": 0.7, "target_value": 0.8, "source": "Table 1", "target_evidence": {"source": "Table 1"}, "rationale": "Table 1 reports the target."}]},
            {"statement": "second", "claim_type": "ablation", "importance_rank": 2, "cost": "S", "verifiability": "high", "metrics": [{"name": "b", "formula": "mean(b)", "direction": "higher_is_better"}], "thresholds": [{"metric": "b", "pass_threshold": 0.7, "target_value": 0.8, "source": "Table 1", "target_evidence": {"source": "Table 1"}, "rationale": "Table 1 reports the target."}]},
        ]
    )
    out = tmp_path / "out"
    build_claim_tasks(
        paper=bundle,
        out=out,
        parse_images="never",
        claude_client=make_llm(claims),
        max_claims=1,
    )
    res = build_claim_tasks(
        paper=bundle,
        out=out,
        parse_images="never",
        claude_client=make_llm(claims),
        max_claims=2,
    )

    assert res["n_selected_claims"] == 2
    refined = json.loads((out / "paper1" / "01-extract" / "refined_claims.json").read_text())
    assert {claim["statement"] for claim in refined} == {"first", "second"}


def test_build_limits_vl_reads_to_selected_claims(tmp_path, make_llm):
    bundle = make_parse_bundle(tmp_path)
    claims = json.dumps(
        [
            {"statement": "first", "claim_type": "mechanism", "importance_rank": 1, "cost": "S", "verifiability": "high", "anchors": [{"kind": "figure", "ref": "Fig. 1"}], "metrics": [{"name": "a", "formula": "mean(a)", "direction": "higher_is_better"}]},
            {"statement": "second", "claim_type": "ablation", "importance_rank": 2, "cost": "S", "verifiability": "high", "anchors": [{"kind": "figure", "ref": "Fig. 1"}], "metrics": [{"name": "b", "formula": "mean(b)", "direction": "higher_is_better"}]},
            {"statement": "third", "claim_type": "headline", "importance_rank": 3, "cost": "XL", "verifiability": "medium", "anchors": [{"kind": "figure", "ref": "Fig. 1"}], "metrics": [{"name": "c", "formula": "mean(c)", "direction": "higher_is_better"}]},
        ]
    )
    fig_evidence = json.dumps(
        {
            "figure_ref": "Fig. 1",
            "targets": [{"name": "a_target", "metric": "a", "value": 0.42}],
            "confidence": 0.9,
        }
    )
    vl = FakeVL({"fig1.jpg": fig_evidence}, default="{}")
    res = build_claim_tasks(
        paper=bundle,
        out=tmp_path / "out",
        parse_images="always",
        claude_client=make_llm(claims),
        multimodal_client=vl,
        max_claims=1,
    )
    extract_dir = tmp_path / "out" / "paper1" / "01-extract"
    evidence_index = json.loads((extract_dir / "claim_evidence.index.json").read_text())
    assert res["n_candidate_claims"] == 3
    assert res["n_selected_claims"] == 1
    triaged = json.loads((extract_dir / "triaged_claims.json").read_text())
    assert len(vl.calls) == 1
    assert len(evidence_index) == 1
    assert [c["route"] for c in triaged] == ["evidence_binding", "exploration", "exploration"]


def test_refresh_claims_ignores_cached_candidates(tmp_path):
    bundle = make_parse_bundle(tmp_path)
    out = tmp_path / "out"
    first = json.dumps([
        {"statement": "old", "claim_type": "mechanism", "importance_rank": 1, "metrics": [{"name": "old", "formula": "mean(old)", "direction": "higher_is_better"}], "thresholds": [{"metric": "old", "pass_threshold": 0.7, "target_value": 0.8, "source": "Table 1", "target_evidence": {"source": "Table 1"}, "rationale": "Table 1 reports the target."}]},
    ])
    second = json.dumps([
        {"statement": "new", "claim_type": "mechanism", "importance_rank": 1, "metrics": [{"name": "new", "formula": "mean(new)", "direction": "higher_is_better"}], "thresholds": [{"metric": "new", "pass_threshold": 0.7, "target_value": 0.8, "source": "Table 1", "target_evidence": {"source": "Table 1"}, "rationale": "Table 1 reports the target."}]},
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
    selected = json.loads((out / "paper1" / "01-extract" / "selected_claims_for_build.json").read_text())
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


def test_parse_images_auto_skips_malformed_vl_json(tmp_path, make_llm):
    bundle = make_parse_bundle(tmp_path)
    vl = FakeVL({"fig1.jpg": "{not valid json"}, default="{not valid json}")

    res = build_claim_tasks(
        paper=bundle,
        out=tmp_path / "out",
        parse_images="auto",
        claude_client=make_llm(CLAIM_WITH_NUMERIC_TARGET_JSON),
        multimodal_client=vl,
        max_claims=1,
    )

    assert len(vl.calls) == 1
    assert len(res["built"]) == 1
    assert res["image_evidence"] is False


def test_directional_only_claim_is_not_built_or_manifested(tmp_path, make_llm):
    bundle = make_parse_bundle(tmp_path)
    claims = json.dumps(
        [
            {
                "statement": "A is better than B.",
                "claim_type": "mechanism",
                "importance_rank": 1,
                "metrics": [
                    {
                        "name": "gap",
                        "formula": "mean(a.score) - mean(b.score)",
                        "direction": "higher_is_better",
                    }
                ],
            }
        ]
    )

    res = build_claim_tasks(
        paper=bundle,
        out=tmp_path / "out",
        parse_images="never",
        claude_client=make_llm(claims),
        max_claims=1,
    )

    assert res["n_selected_claims"] == 0
    assert res["built"] == []
    manifest = json.loads((tmp_path / "out" / "paper1" / "task_manifest.json").read_text())
    assert manifest["tasks"] == []
