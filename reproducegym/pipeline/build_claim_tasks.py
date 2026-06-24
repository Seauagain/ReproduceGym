"""Build paper-derived claim/task bundles.

This is the single implementation of stage 1:

    paper -> figure index/evidence -> ranked claims -> spec -> task bundle

It never launches a sandbox or a GPU job.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from reproducegym.claim_spec import dump_claim_spec
from reproducegym.config import load_dotenv
from reproducegym.models import ClaudeClient, MultimodalFigureClient, multimodal_figure_configured
from reproducegym.pipeline.extract_claims import (
    ExtractError,
    dedup_claim_candidates,
    extract_claim_candidates,
    extract_global_claim_candidates,
    finalize_claims,
    refine_claim_with_evidence,
)
from reproducegym.pipeline.claim_selection import (
    DEFAULT_MAX_CLAIMS,
    rank_claims,
    selection_table,
    select_top_claims,
)
from reproducegym.pipeline.extract_figure_params import extract_claim_figure_evidence
from reproducegym.pipeline.merge_claim_spec import merge_claim_spec
from reproducegym.pipeline.paper_assets import (
    collect_markdown_figures,
    count_image_refs,
    write_figure_index,
)
from reproducegym.pipeline.render_check import write_baseline_check
from reproducegym.pipeline.render_task import render_task
from reproducegym.pipeline.rlvr_task_contract import (
    assign_final_claim_ids,
    build_claim_verification_report,
    ensure_claim_uid,
    select_claims_for_build,
)
from reproducegym.pipeline.token_usage import RecordingLLMClient, RecordingVLClient, TokenUsageRecorder
from reproducegym.pipeline.validate_task import validate_task
from reproducegym.verifier.engine import _aggregate_scores, _curve_score
from reproducegym.runlayout import PARSE, PaperLayout, write_index


def _resolve_paper_input(
    paper: str | Path, paper_id: str | None
) -> tuple[Path, Path | None, Path | None, str]:
    """Locate the markdown + its figures from a parse bundle or a raw paper file.

    Returns (paper_md, figures_dir, prebuilt_index, paper_id). A directory is read
    as a parse bundle (its 00-parse/ stage, or itself if it is one); a file is read
    as a raw paper.md whose figures live in a sibling figures/ dir.
    """
    paper = Path(paper)
    if paper.is_dir():
        pdir = paper / PARSE if (paper / PARSE).is_dir() else paper
        paper_md = pdir / "paper.md"
        if not paper_md.is_file():
            raise ValueError(
                f"no paper.md under {pdir}; run parse_paper.py to build the parse bundle first"
            )
        derived = paper.name if (paper / PARSE).is_dir() else paper.parent.name
        figures_dir = pdir / "figures"
        index = pdir / "figures.index.json"
        return paper_md, figures_dir if figures_dir.is_dir() else None, (
            index if index.is_file() else None
        ), (paper_id or derived)
    figures_dir = paper.parent / "figures"
    return paper, figures_dir if figures_dir.is_dir() else None, None, (paper_id or paper.stem)


def select_claims(claims: list[dict[str, Any]], wanted: list[str]) -> list[dict[str, Any]]:
    if not wanted:
        return claims
    out = []
    for cid in wanted:
        match = next(
            (
                c for c in claims
                if c.get("claim_id") == cid or c.get("source_claim_id") == cid
            ),
            None,
        )
        if match is None:
            raise ValueError(f"claim {cid!r} not found; have {[c['claim_id'] for c in claims]}")
        out.append(match)
    return out


def should_parse_images(mode: str, *, has_figures: bool, configured: bool) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        if not has_figures:
            raise ValueError("--parse-images=always requires local Markdown image references")
        if not configured:
            raise ValueError(
                "--parse-images=always requires a configured multimodal model "
                "(MULTIMODAL_* / VISION_* / QWEN_* keys in .env)"
            )
        return True
    if mode != "auto":
        raise ValueError(f"unknown parse_images mode: {mode!r}")
    return has_figures and configured


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _paper_evidence_index(paper_md: Path, figures: list[dict[str, Any]]) -> dict[str, Any]:
    text = paper_md.read_text(encoding="utf-8")
    paragraphs = []
    offset = 0
    current_section = ""
    for i, block in enumerate(text.split("\n\n"), start=1):
        start = text.find(block, offset)
        if start < 0:
            start = offset
        end = start + len(block)
        offset = end
        stripped = block.strip()
        if stripped.startswith("#"):
            current_section = stripped.splitlines()[0].lstrip("#").strip()
        if stripped:
            paragraphs.append(
                {
                    "id": f"p{i:04d}",
                    "section": current_section,
                    "start": start,
                    "end": end,
                    "text": stripped,
                }
            )
    figure_records = []
    for fig in figures:
        ref = str(fig.get("figure_ref") or "")
        caption = str(fig.get("caption") or "")
        context = str(fig.get("context") or "")
        nearby = [
            p["id"]
            for p in paragraphs
            if (ref and ref.lower() in p["text"].lower())
            or (caption and caption[:40].lower() in p["text"].lower())
        ][:5]
        figure_records.append(
            {
                "figure_ref": ref,
                "image_file": fig.get("image_file"),
                "source_path": fig.get("source_path"),
                "caption": caption,
                "context": context[:4000],
                "nearby_paragraph_ids": nearby,
            }
        )
    return {
        "source": str(paper_md),
        "n_chars": len(text),
        "paragraphs": paragraphs,
        "figures": figure_records,
    }


def _anchor_matches(anchor: dict[str, Any], fig: dict[str, Any]) -> bool:
    ref = str(anchor.get("ref", "")).lower().replace(" ", "")
    fig_ref = str(fig.get("figure_ref", "")).lower().replace(" ", "")
    return bool(ref and ref in fig_ref)


def _claim_slices_from_index(claim: dict[str, Any], index: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = [a for a in (claim.get("evidence_anchors") or claim.get("anchors") or []) if isinstance(a, dict)]
    paragraphs = index.get("paragraphs") or []
    wanted_ids: set[str] = set()
    for anchor in anchors:
        ref = str(anchor.get("ref") or "").lower()
        for para in paragraphs:
            if ref and ref in str(para.get("text") or "").lower():
                wanted_ids.add(str(para["id"]))
    if not wanted_ids:
        statement_terms = [
            t.lower()
            for t in str(claim.get("statement") or "").replace("-", " ").split()
            if len(t) >= 5
        ][:8]
        for para in paragraphs:
            low = str(para.get("text") or "").lower()
            if any(term in low for term in statement_terms):
                wanted_ids.add(str(para["id"]))
            if len(wanted_ids) >= 4:
                break
    if not wanted_ids and paragraphs:
        wanted_ids.add(str(paragraphs[0]["id"]))
    return [
        {
            "source": index.get("source"),
            "kind": "paragraph",
            "paragraph_id": para["id"],
            "section": para.get("section"),
            "text": para.get("text"),
        }
        for para in paragraphs
        if para.get("id") in wanted_ids
    ][:8]


def _triage_claims(claims: list[dict[str, Any]], *, max_claims: int | None) -> list[dict[str, Any]]:
    limit = len(claims) if max_claims is None else max(0, int(max_claims))
    out = []
    for i, claim in enumerate(claims):
        c = dict(claim)
        route = "evidence_binding" if i < limit else "exploration"
        c["route"] = route
        c["route_reason"] = (
            "selected for claim-scoped evidence binding"
            if route == "evidence_binding"
            else "outside max_claims_for_evidence budget"
        )
        c["importance_score"] = max(0.0, 1.0 - 0.05 * i)
        c["quantifiability_score"] = 1.0 if (c.get("anchors") or c.get("evidence_anchors") or c.get("metrics")) else 0.3
        c["reproducibility_score"] = 0.8 if c.get("metrics") else 0.4
        c["cost_score"] = {"S": 0.1, "M": 0.4, "L": 0.7, "XL": 1.0}.get(str(c.get("cost") or "M"), 0.4)
        out.append(c)
    return out


def _target_points_from_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for fig in bundle.get("figure_evidence") or []:
        for raw in fig.get("targets") or []:
            if not isinstance(raw, dict):
                continue
            value = raw.get("value")
            if _number(value) is None:
                continue
            source = {"kind": "figure", "ref": raw.get("source") or fig.get("figure_ref")}
            if raw.get("panel"):
                source["panel"] = raw.get("panel")
            points.append(
                {
                    "observable": raw.get("metric") or raw.get("name"),
                    "condition": raw.get("condition"),
                    "value": _number(value),
                    "unit": raw.get("unit"),
                    "source": source,
                    "read_from": raw.get("read_from"),
                    "confidence": _number(raw.get("confidence")),
                    "tolerance": raw.get("tolerance"),
                }
            )
    return points


def _curve_selftest_rewards(spec: dict[str, Any]) -> dict[str, Any]:
    curves = spec.get("reward_curves") or {}
    if not curves:
        return {
            "target": {"reward": 0.0, "error": "no reward_curves"},
            "threshold": {"reward": 0.0, "error": "no reward_curves"},
            "poor": {"reward": 0.0, "error": "no reward_curves"},
            "missing": {"reward": 0.0},
        }
    by_case = {"target": {}, "threshold": {}, "poor": {}}
    for name, curve in curves.items():
        points = sorted(curve.get("points") or [], key=lambda p: float(p["reward"]))
        if len(points) < 3:
            return {
                "target": {"reward": 0.0, "error": f"curve {name} has fewer than 3 points"},
                "threshold": {"reward": 0.0},
                "poor": {"reward": 0.0},
                "missing": {"reward": 0.0},
            }
        try:
            by_case["poor"][name] = _curve_score(points[0]["value"], curve)
            by_case["threshold"][name] = _curve_score(points[1]["value"], curve)
            by_case["target"][name] = _curve_score(points[-1]["value"], curve)
        except Exception as exc:  # noqa: BLE001 - embedded scorer reports verifier errors
            return {
                "target": {"reward": 0.0, "error": f"curve {name} failed selftest: {exc}"},
                "threshold": {"reward": 0.0},
                "poor": {"reward": 0.0},
                "missing": {"reward": 0.0},
            }
    return {
        case: {"reward": round(float(_aggregate_scores(scores, spec) or 0.0), 6), "metric_rewards": scores}
        for case, scores in by_case.items()
    } | {"missing": {"reward": 0.0}}


def _write_build_validation(layout: PaperLayout, built: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = []
    for item in built:
        spec = item["spec"]
        problems = list(item.get("validation_problems") or [])
        selftests = _curve_selftest_rewards(spec)
        verification = spec.get("verification") or {}
        metric_names = {
            str(metric.get("name"))
            for metric in spec.get("metrics") or []
            if isinstance(metric, dict) and metric.get("name")
        }
        curve_metrics = set((spec.get("reward_curves") or {}).keys())
        threshold_targets = {
            str(threshold.get("metric"))
            for threshold in spec.get("thresholds") or []
            if threshold.get("metric") and threshold.get("target_value") is not None
        }
        acceptance_problems: list[str] = []
        if verification.get("pool") != "rlvr":
            acceptance_problems.append(f"verification pool is {verification.get('pool') or 'unset'}")
        if verification.get("mode") == "unverifiable":
            acceptance_problems.append("verification mode is unverifiable")
        missing_targets = sorted(metric_names - threshold_targets)
        if missing_targets:
            acceptance_problems.append(
                "missing paper-grounded target_value for metric(s): " + ", ".join(missing_targets)
            )
        missing_curves = sorted(metric_names - curve_metrics)
        if missing_curves:
            acceptance_problems.append(
                "missing reward curve for metric(s): " + ", ".join(missing_curves)
            )
        ordered = (
            selftests.get("target", {}).get("reward", 0.0)
            >= selftests.get("threshold", {}).get("reward", 0.0)
            >= selftests.get("poor", {}).get("reward", 0.0)
            >= selftests.get("missing", {}).get("reward", 0.0)
        )
        accepted = (
            not problems
            and not acceptance_problems
            and ordered
            and selftests.get("target", {}).get("reward") == 1.0
        )
        tasks.append(
            {
                "claim_id": spec["claim_id"],
                "claim_uid": spec.get("claim_uid"),
                "contract_hash": spec.get("contract_hash"),
                "spec_hash": spec["spec_hash"],
                "task_dir": item["task_dir"],
                "pool": verification.get("pool") or "exploration",
                "validation_problems": problems,
                "acceptance_problems": acceptance_problems,
                "synthetic_selftests": selftests,
                "accepted": accepted,
                "acceptance_reason": "accepted" if accepted else "validation, gating, or selftest failed",
            }
        )
    doc = {"tasks": tasks}
    _write_json(layout.root / "build_validation.json", doc)
    return doc


def _write_task_manifest(layout: PaperLayout, *, paper_id: str, build_validation: dict[str, Any]) -> dict[str, Any]:
    tasks = [
        {
            "claim_id": task["claim_id"],
            "claim_uid": task.get("claim_uid"),
            "contract_hash": task.get("contract_hash"),
            "spec_hash": task["spec_hash"],
            "task_dir": task["task_dir"],
            "pool": task.get("pool") or "rlvr",
        }
        for task in build_validation.get("tasks") or []
        if task.get("accepted")
    ]
    doc = {"paper_id": paper_id, "tasks": tasks}
    _write_json(layout.root / "task_manifest.json", doc)
    return doc


CLAIM_TYPES = {"eval_only", "mechanism", "ablation", "scaling", "headline", "diagnostic"}


def _normalize_claim_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in CLAIM_TYPES:
        return text
    if text in {"directional_comparison", "numeric_threshold", "table_or_curve_point", "artifact_metric"}:
        return "eval_only"
    return "mechanism"


def _verification_contract_from_claim(claim: dict[str, Any]) -> dict[str, Any]:
    contract = dict(claim.get("verification_contract") or {})
    contract.setdefault("type", claim.get("verification", {}).get("mode") or claim.get("likely_pool") or "artifact_metric")
    contract.setdefault("conditions", list(claim.get("conditions") or []))
    contract["metrics"] = _normalize_metrics(list(contract.get("metrics") or claim.get("metrics") or []))
    contract["params"] = _normalize_params(list(contract.get("params") or claim.get("params") or []))
    contract["thresholds"] = _normalize_thresholds(list(contract.get("thresholds") or claim.get("thresholds") or []))
    contract.setdefault("verdict_rules", dict(claim.get("verdict_rules") or {}))
    return contract


def _normalize_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "lower_is_better" or text in {"lower", "smaller_is_better", "less_is_better"}:
        return "lower_is_better"
    return "higher_is_better"


def _normalize_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in metrics:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        item = dict(raw)
        item["name"] = str(item["name"])
        item["formula"] = str(item.get("formula") or "")
        item["direction"] = _normalize_direction(item.get("direction"))
        out.append(item)
    return out


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _normalize_target_evidence(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str) and value.strip():
        return {"source": value.strip()}
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {}
    for key in ("param_name", "source", "read_from"):
        if value.get(key) is not None:
            out[key] = str(value[key])
    conf = _number(value.get("confidence"))
    if conf is not None:
        out["confidence"] = conf
    return out or None


def _normalize_thresholds(thresholds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in thresholds:
        if not isinstance(raw, dict) or not raw.get("metric"):
            continue
        pass_threshold = _number(raw.get("pass_threshold"))
        if pass_threshold is None:
            continue
        item: dict[str, Any] = {
            "metric": str(raw["metric"]),
            "pass_threshold": pass_threshold,
            "exposure": raw.get("exposure") if raw.get("exposure") in {"visible", "hidden"} else "hidden",
        }
        if pass_threshold == 0:
            item["exposure"] = "visible"
        for key in ("target_value", "tolerance_abs", "confidence"):
            num = _number(raw.get(key))
            if num is not None:
                item[key] = num
        for key in ("rationale", "source"):
            if raw.get(key) is not None:
                item[key] = str(raw[key])
        ev = _normalize_target_evidence(raw.get("target_evidence"))
        if ev is not None:
            item["target_evidence"] = ev
        tolerance = raw.get("tolerance")
        tol_num = _number(tolerance)
        if tol_num is not None:
            item["tolerance"] = tol_num
        elif isinstance(tolerance, dict):
            tol_obj = {k: _number(tolerance.get(k)) for k in ("rel", "abs")}
            tol_obj = {k: v for k, v in tol_obj.items() if v is not None}
            if tol_obj:
                item["tolerance"] = tol_obj
        out.append(item)
    return out


def _confidence(value: Any) -> float | None:
    num = _number(value)
    if num is not None:
        return num
    if isinstance(value, str):
        return {"high": 0.85, "medium": 0.6, "low": 0.3}.get(value.strip().lower())
    return None


def _normalize_params(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in params:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        item: dict[str, Any] = {
            "name": str(raw["name"]),
            "status": raw.get("status")
            if raw.get("status") in {"paper_specified", "author_repo_config", "paper_unspecified"}
            else "paper_specified",
        }
        if "value" in raw:
            item["value"] = raw.get("value")
        for key in ("unit", "source", "read_from", "metric", "condition"):
            if raw.get(key) is not None:
                item[key] = str(raw[key])
        use = raw.get("use")
        if use == "config":
            use = "reproduction_param"
        item["use"] = use if use in {"reproduction_param", "target", "context"} else "reproduction_param"
        conf = _confidence(raw.get("confidence"))
        if conf is not None:
            item["confidence"] = conf
        tolerance = raw.get("tolerance")
        tol_num = _number(tolerance)
        if tol_num is not None:
            item["tolerance"] = tol_num
        elif isinstance(tolerance, dict):
            tol_obj = {k: _number(tolerance.get(k)) for k in ("rel", "abs")}
            tol_obj = {k: v for k, v in tol_obj.items() if v is not None}
            if tol_obj:
                item["tolerance"] = tol_obj
        if raw.get("comparator") in {">=", "<=", ">", "<", "=="}:
            item["comparator"] = raw["comparator"]
        if isinstance(raw.get("applies_to_claim"), bool):
            item["applies_to_claim"] = raw["applies_to_claim"]
        out.append(item)
    return out


def _normalize_refined_claim(claim: dict[str, Any], evidence_bundle: dict[str, Any]) -> dict[str, Any]:
    out = ensure_claim_uid(claim)
    out.setdefault("source_mode", claim.get("source_mode") or "global")
    out["claim_type"] = _normalize_claim_type(out.get("claim_type"))
    out.setdefault("evidence_anchors", claim.get("evidence_anchors") or claim.get("anchors") or [])
    out.setdefault("reproduction_protocol", {
        "summary": claim.get("implementation_notes") or claim.get("notes") or claim.get("statement", ""),
        "required_outputs": ["output/result.json", "output/metrics.csv"],
    })
    contract = _verification_contract_from_claim(out)
    # Figure targets live in the evidence bundle; expose them to synthesis as
    # target params without forcing the LLM to copy them into refined output.
    params = list(contract.get("params") or [])
    for fig in evidence_bundle.get("figure_evidence") or []:
        params.extend(fig.get("params") or [])
        params.extend(fig.get("targets") or [])
    contract["params"] = _normalize_params(params)
    out["verification_contract"] = contract
    out["conditions"] = list(contract.get("conditions") or [])
    out["metrics"] = list(contract.get("metrics") or [])
    out["params"] = list(contract.get("params") or [])
    out["thresholds"] = list(contract.get("thresholds") or [])
    out["verdict_rules"] = dict(contract.get("verdict_rules") or {})
    return out


def _claim_evidence_bundle(
    claim: dict[str, Any],
    *,
    paper_index: dict[str, Any],
    figures: list[dict[str, Any]],
    figures_dir_for_vl: Path,
    do_parse_images: bool,
    vl: Any | None,
    token_recorder: TokenUsageRecorder,
    vl_min_confidence: float,
    strict_vl: bool,
) -> dict[str, Any]:
    claim = ensure_claim_uid(claim)
    anchors = [a for a in (claim.get("evidence_anchors") or claim.get("anchors") or []) if isinstance(a, dict)]
    figure_evidence: list[dict[str, Any]] = []
    if do_parse_images and vl is not None:
        figure_evidence = extract_claim_figure_evidence(
            claim,
            figures_dir_for_vl,
            client=RecordingVLClient(
                vl,
                token_recorder,
                step="extract_claim_figure_evidence",
                metadata={"claim_uid": claim["claim_uid"]},
            ),
            figures_index=figures,
            min_confidence=vl_min_confidence,
            strict=strict_vl,
        )
    return {
        "claim_uid": claim["claim_uid"],
        "statement": claim.get("statement"),
        "evidence_anchors": anchors,
        "paper_text_slices": _claim_slices_from_index(claim, paper_index),
        "figure_refs": [
            fig for fig in figures
            if any(_anchor_matches(anchor, fig) for anchor in anchors if anchor.get("kind") in {"figure", "table"})
        ],
        "figure_evidence": figure_evidence,
    }


def _selected_claim_to_spec_claim(claim: dict[str, Any]) -> dict[str, Any]:
    out = dict(claim)
    out["claim_type"] = _normalize_claim_type(out.get("claim_type"))
    contract = _verification_contract_from_claim(out)
    out["conditions"] = list(contract.get("conditions") or [])
    out["metrics"] = list(contract.get("metrics") or [])
    out["params"] = list(contract.get("params") or [])
    out["thresholds"] = _normalize_thresholds(list(out.get("accepted_targets") or contract.get("thresholds") or []))
    out["verdict_rules"] = dict(contract.get("verdict_rules") or out.get("verdict_rules") or {})
    out["reward_curves"] = dict(out.get("reward_curves") or {})
    return out


def build_claim_tasks(
    *,
    paper: str | Path,
    paper_id: str | None = None,
    out: str | Path,
    claim_ids: list[str] | None = None,
    parse_images: str = "auto",
    vl_min_confidence: float = 0.0,
    strict_vl: bool = True,
    baseline_check: bool = True,
    max_claims: int | None = DEFAULT_MAX_CLAIMS,
    refresh_claims: bool = False,
    claude_client: Any | None = None,
    multimodal_client: Any | None = None,
) -> dict[str, Any]:
    """Build hash-versioned task bundles from a paper."""

    load_dotenv(override=True)
    paper_md, src_figures_dir, prebuilt_index, paper_id = _resolve_paper_input(paper, paper_id)
    layout = PaperLayout.for_paper(out, paper_id)
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.extract_dir.mkdir(parents=True, exist_ok=True)
    token_recorder = TokenUsageRecorder(layout.root, paper_id=paper_id)

    build_figures_dir = layout.extract_dir / "figures"
    if prebuilt_index is not None:
        # Reuse the parse-stage figure index directly. Image bytes live only in
        # 00-parse/figures; 01-extract stores metadata/evidence, not another copy.
        shutil.rmtree(build_figures_dir, ignore_errors=True)
        figures = json.loads(prebuilt_index.read_text(encoding="utf-8"))
        figures_dir_for_vl = src_figures_dir or build_figures_dir
    else:
        # Raw paper.md: resolve refs against a sibling figures/ dir (the core fix --
        # markdown that writes images/x.jpg now resolves to the parsed figures/).
        figures = collect_markdown_figures(
            paper_md,
            figures_dir=src_figures_dir,
            copy_to=build_figures_dir,
            strict=parse_images == "always",
        )
        figures_dir_for_vl = build_figures_dir
    write_figure_index(figures, layout.extract_dir / "figures.index.json")
    paper_index_path = layout.extract_dir / "paper_evidence_index.json"
    paper_index = _paper_evidence_index(paper_md, figures)
    _write_json(paper_index_path, paper_index)
    if not figures and parse_images == "auto" and count_image_refs(paper_md) > 0:
        print(
            f"[build] WARNING: {paper_md} has image references but none resolved to local "
            "files; image-enhanced parsing skipped. Run parse_paper.py to fetch figures.",
            flush=True,
        )

    claude = claude_client or ClaudeClient()
    configured = multimodal_client is not None or multimodal_figure_configured()
    do_parse_images = should_parse_images(
        parse_images,
        has_figures=bool(figures),
        configured=configured,
    )
    vl = (multimodal_client or MultimodalFigureClient()) if do_parse_images else None

    candidate_path = layout.extract_dir / "candidate_claims.json"
    if candidate_path.is_file() and not refresh_claims:
        candidate_claims = json.loads(candidate_path.read_text(encoding="utf-8"))
        token_recorder.record_event(
            stage="build",
            step="candidate_claims.cache_hit",
            metadata={"path": str(candidate_path), "n_claims": len(candidate_claims)},
        )
    else:
        try:
            candidate_claims = extract_global_claim_candidates(
                paper_md,
                client=RecordingLLMClient(
                    claude,
                    token_recorder,
                    step="extract_global_claims",
                    metadata={"paper_md": str(paper_md), "source_mode": "global"},
                ),
                figures=figures,
            )
            source_mode = "global"
        except (ExtractError, json.JSONDecodeError, ValueError, TimeoutError, OSError, RuntimeError) as exc:
            token_recorder.record_event(
                stage="build",
                step="extract_global_claims.fallback",
                metadata={"fallback_reason": str(exc), "source_mode": "chunked_fallback"},
            )
            raw = extract_claim_candidates(
                paper_md,
                client=RecordingLLMClient(
                    claude,
                    token_recorder,
                    step="extract_claim_candidates",
                    metadata={"paper_md": str(paper_md), "source_mode": "chunked_fallback"},
                ),
                figures=figures,
                max_chunk_chars=18_000,
            )
            candidate_claims = dedup_claim_candidates(
                raw,
                client=RecordingLLMClient(
                    claude,
                    token_recorder,
                    step="dedup_claim_candidates",
                    metadata={"n_input_claims": len(raw), "source_mode": "chunked_fallback"},
                ),
            )
            source_mode = "chunked_fallback"
        candidate_claims = rank_claims(finalize_claims(candidate_claims, client=None))
        candidate_claims = [
            ensure_claim_uid({**claim, "source_mode": claim.get("source_mode") or source_mode})
            for claim in candidate_claims
        ]
        _write_json(candidate_path, candidate_claims)
    _write_json(layout.extract_dir / "claim_selection.json", selection_table(candidate_claims))

    candidate_claims = select_claims(candidate_claims, claim_ids or [])
    triaged_claims = _triage_claims(
        candidate_claims,
        max_claims=None if claim_ids else max_claims,
    )
    _write_json(layout.extract_dir / "triaged_claims.json", triaged_claims)
    evidence_claims = [claim for claim in triaged_claims if claim.get("route") == "evidence_binding"]

    claim_evidence_dir = layout.extract_dir / "claim_evidence"
    target_points_dir = layout.extract_dir / "target_points"
    if refresh_claims:
        shutil.rmtree(claim_evidence_dir, ignore_errors=True)
        shutil.rmtree(target_points_dir, ignore_errors=True)
    claim_evidence_dir.mkdir(parents=True, exist_ok=True)
    target_points_dir.mkdir(parents=True, exist_ok=True)
    aggregate_evidence: list[dict[str, Any]] = []
    evidence_bundles: dict[str, dict[str, Any]] = {}
    target_points_index: dict[str, list[dict[str, Any]]] = {}
    for claim in evidence_claims:
        claim = ensure_claim_uid(claim)
        evidence_path = claim_evidence_dir / f"{claim['claim_uid']}.json"
        if evidence_path.is_file() and not refresh_claims:
            bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
        else:
            start = time.perf_counter()
            bundle = _claim_evidence_bundle(
                claim,
                paper_index=paper_index,
                figures=figures,
                figures_dir_for_vl=figures_dir_for_vl,
                do_parse_images=do_parse_images,
                vl=vl,
                token_recorder=token_recorder,
                vl_min_confidence=vl_min_confidence,
                strict_vl=strict_vl and parse_images == "always",
            )
            token_recorder.record_event(
                stage="build",
                step="build_claim_evidence",
                elapsed_ms=_elapsed_ms(start),
                metadata={"claim_uid": claim["claim_uid"], "source_mode": claim.get("source_mode")},
            )
            _write_json(evidence_path, bundle)
        evidence_bundles[claim["claim_uid"]] = bundle
        aggregate_evidence.extend(bundle.get("figure_evidence") or [])
        target_points = _target_points_from_bundle(bundle)
        target_points_index[claim["claim_uid"]] = target_points
        _write_json(target_points_dir / f"{claim['claim_uid']}.json", target_points)
    _write_json(layout.extract_dir / "claim_evidence.index.json", evidence_bundles)
    _write_json(layout.extract_dir / "target_points.index.json", target_points_index)
    (layout.extract_dir / "figure_evidence.yaml").write_text(
        yaml.safe_dump(aggregate_evidence, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    refined_path = layout.extract_dir / "refined_claims.json"
    expected_refined_uids = {
        ensure_claim_uid(claim)["claim_uid"]
        for claim in evidence_claims
    }
    cached_refined_ok = False
    if refined_path.is_file() and not refresh_claims:
        cached_refined = json.loads(refined_path.read_text(encoding="utf-8"))
        cached_uids = {
            ensure_claim_uid(claim)["claim_uid"]
            for claim in cached_refined
            if isinstance(claim, dict)
        }
        cached_refined_ok = cached_uids == expected_refined_uids
    if cached_refined_ok:
        refined_claims = cached_refined
    else:
        refined_claims = []
        for claim in evidence_claims:
            claim = ensure_claim_uid(claim)
            bundle = evidence_bundles[claim["claim_uid"]]
            needs_llm_refine = not claim.get("verification_contract") and not claim.get("metrics")
            if needs_llm_refine:
                try:
                    refined = refine_claim_with_evidence(
                        claim,
                        bundle,
                        client=RecordingLLMClient(
                            claude,
                            token_recorder,
                            step="refine_claim_with_evidence",
                            metadata={"claim_uid": claim["claim_uid"], "source_mode": claim.get("source_mode")},
                        ),
                    )
                except (ExtractError, json.JSONDecodeError, ValueError, TimeoutError, OSError, RuntimeError):
                    refined = dict(claim)
            else:
                refined = dict(claim)
            refined_claims.append(_normalize_refined_claim(refined, bundle))
        _write_json(refined_path, refined_claims)

    report_path = layout.extract_dir / "claim_verification_report.json"
    start = time.perf_counter()
    verification_report = build_claim_verification_report(refined_claims)
    token_recorder.record_event(
        stage="build",
        step="build_claim_verification_report",
        elapsed_ms=_elapsed_ms(start),
        metadata={"n_claims": len(refined_claims)},
    )
    _write_json(report_path, verification_report)

    selected_claims = select_claims_for_build(
        refined_claims,
        verification_report,
        max_claims=max_claims if not claim_ids else None,
    )
    selected_claims = assign_final_claim_ids(selected_claims)
    _write_json(layout.extract_dir / "selected_claims_for_build.json", selected_claims)

    built = []
    for claim in selected_claims:
        spec = merge_claim_spec(
            _selected_claim_to_spec_claim(claim),
            paper_id=paper_id,
            figure_evidence=aggregate_evidence,
        )
        spec_path = layout.spec_path(spec["claim_id"], spec["spec_hash"])
        dump_claim_spec(spec, spec_path)
        task_dir = layout.task_dir(spec["claim_id"], spec["spec_hash"])
        render_task(spec, task_dir)
        if baseline_check:
            write_baseline_check(spec, task_dir / "reward")
        problems = validate_task(task_dir, spec)
        if problems:
            raise ValueError("task failed validation: " + "; ".join(problems))
        built.append(
            {
                "claim_id": spec["claim_id"],
                "claim_uid": spec.get("claim_uid"),
                "contract_hash": spec.get("contract_hash"),
                "spec_hash": spec["spec_hash"],
                "spec_path": str(spec_path),
                "task_dir": str(task_dir),
                "spec": spec,
                "validation_problems": problems,
            }
        )

    build_validation = _write_build_validation(layout, built)
    task_manifest = _write_task_manifest(layout, paper_id=paper_id, build_validation=build_validation)
    manifest = write_index(layout, paper_id=paper_id)
    token_summary = token_recorder.write_summary()
    return {
        "paper_id": paper_id,
        "built": [
            {k: v for k, v in item.items() if k not in {"spec", "validation_problems"}}
            for item in built
        ],
        "manifest": str(layout.manifest_path),
        "task_manifest": str(layout.root / "task_manifest.json"),
        "build_validation": str(layout.root / "build_validation.json"),
        "n_claims": len(manifest["claims"]),
        "n_candidate_claims": len(candidate_claims),
        "n_selected_claims": len(selected_claims),
        "max_claims": max_claims if not claim_ids else None,
        "selection": str(layout.extract_dir / "selected_claims_for_build.json"),
        "verification_report": str(layout.extract_dir / "claim_verification_report.json"),
        "parse_images": parse_images,
        "image_evidence": bool(aggregate_evidence),
        "token_usage": str(token_recorder.jsonl_path),
        "token_usage_summary": str(token_summary),
    }
