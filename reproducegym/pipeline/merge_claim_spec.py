"""Step 2c: merge extracted claim (+ figure params) into a canonical claim spec.

The bridge between extract_claims (step-1 claim dicts) and render_task (which needs
a schema-valid canonical spec). It selects only schema-allowed fields, fills the
required scaffolding (paper_id, tier, exposure_policy, required_outputs, ...) with
explicit defaults, folds in figure-derived params, then validates against
schema/claim_spec.schema.json. The result is the single source of truth and may be
written to runs/<paper>/02-spec/<claim_id>.yaml.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from reproducegym.claim_spec import dump_claim_spec, validate_claim_spec
from reproducegym.pipeline.claim_ids import claim_slug_from_id, slugify
from reproducegym.pipeline.spec_hash import with_spec_hash
from reproducegym.pipeline.synthesize_contract import apply_verification_contract

DEFAULT_REQUIRED_OUTPUTS = {"files": ["output/result.json", "output/metrics.csv"]}
DEFAULT_TIER = "T2_proxy"
DEFAULT_EXPOSURE_POLICY = "v0_full_paper_public"


class MergeError(ValueError):
    pass


def _norm_metric(m: dict[str, Any]) -> dict[str, Any]:
    if not m.get("name"):
        raise MergeError("metric missing 'name'")
    direction = m.get("direction") or "higher_is_better"
    if direction not in {"higher_is_better", "lower_is_better"}:
        direction = "higher_is_better"
    out = {
        "name": m["name"],
        "formula": m.get("formula") or f"recompute({m['name']})",
        "direction": direction,
    }
    if m.get("window"):
        out["window"] = m["window"]
    return out


def _norm_param(p: dict[str, Any]) -> dict[str, Any]:
    if not p.get("name"):
        raise MergeError("param missing 'name'")
    status = p.get("status")
    if status not in {"paper_specified", "author_repo_config", "paper_unspecified"}:
        status = "paper_specified" if p.get("value") is not None else "paper_unspecified"
    exposure = p.get("exposure") or p.get("visibility")
    if exposure not in {None, "visible", "hidden"}:
        exposure = None
    if exposure is None and p.get("use") == "target":
        exposure = "hidden"
    out: dict[str, Any] = {"name": p["name"], "status": status}
    for k in ("value", "unit", "source", "applies_to_claim", "local_substitute_allowed",
              "affects_strict_reproduction", "required_for_strict", "use", "confidence",
              "read_from", "metric", "condition", "tolerance", "comparator"):
        if k in p and p[k] is not None:
            out[k] = p[k]
    if exposure is not None:
        out["exposure"] = exposure
    return out


def _figure_params_to_entries(figure_params: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not figure_params:
        return []
    entries = []
    for name, raw in figure_params.items():
        raw = dict(raw) if isinstance(raw, dict) else {"value": raw}
        raw.setdefault("name", name)
        raw.setdefault("source", raw.get("source", "figure (multimodal)"))
        raw.setdefault("status", "paper_specified")
        entries.append(_norm_param(raw))
    return entries


def _norm_ref(ref: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", ref.lower())


def _claim_figure_refs(claim: dict[str, Any]) -> set[str]:
    refs = set()
    for anchor in claim.get("anchors", []) or []:
        if anchor.get("kind") == "figure" and anchor.get("ref"):
            refs.add(_norm_ref(str(anchor["ref"])))
    return refs


def _evidence_for_claim(
    claim: dict[str, Any],
    figure_evidence: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not figure_evidence:
        return []
    refs = _claim_figure_refs(claim)
    if not refs:
        return []
    out = []
    for fig in figure_evidence:
        if _norm_ref(str(fig.get("figure_ref", ""))) in refs:
            out.append(dict(fig))
    return out


def _figure_evidence_to_params(figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for fig in figures:
        for p in fig.get("params") or []:
            if isinstance(p, dict):
                raw = dict(p)
                raw.setdefault("source", fig.get("figure_ref", "figure (multimodal)"))
                raw.setdefault("use", "reproduction_param")
                raw.setdefault("visibility", "visible")
                entries.append(_norm_param(raw))
        for p in fig.get("targets") or []:
            if isinstance(p, dict):
                raw = dict(p)
                raw.setdefault("source", fig.get("figure_ref", "figure (multimodal)"))
                raw.setdefault("use", "target")
                raw.setdefault("visibility", "hidden")
                entries.append(_norm_param(raw))
    return entries


def _claim_meta(claim: dict[str, Any]) -> dict[str, Any]:
    claim_id = str(claim["claim_id"])
    m = re.match(r"^c(\d{3})_(.+)$", claim_id)
    claim_num = int(claim.get("claim_num") or (int(m.group(1)) if m else 1))
    slug = str(claim.get("claim_slug") or (m.group(2) if m else claim_slug_from_id(claim_id)))
    display = str(
        claim.get("display_title")
        or slug.replace("_", " ")
        or slugify(str(claim.get("statement", "claim")))
    ).strip()
    rank = int(claim.get("importance_rank") or claim_num)
    return {
        "claim_num": claim_num,
        "claim_slug": slug,
        "display_title": display,
        "importance_rank": rank,
    }


def merge_claim_spec(
    claim: dict[str, Any],
    figure_params: dict[str, Any] | None = None,
    figure_evidence: list[dict[str, Any]] | None = None,
    out_path: str | Path | None = None,
    *,
    paper_id: str,
    tier: str = DEFAULT_TIER,
    exposure_policy: str = DEFAULT_EXPOSURE_POLICY,
    thresholds: list[dict] | None = None,
    required_outputs: dict | None = None,
    verdict_rules: dict | None = None,
    reward: dict | None = None,
) -> dict[str, Any]:
    for key in ("claim_id", "statement", "claim_type"):
        if not claim.get(key):
            raise MergeError(f"claim missing required key {key!r}")

    meta = _claim_meta(claim)
    bound_figures = _evidence_for_claim(claim, figure_evidence)

    spec: dict[str, Any] = {
        "claim_id": claim["claim_id"],
        **meta,
        "paper_id": paper_id,
        "claim_type": claim["claim_type"],
        "tier": tier,
        "exposure_policy": exposure_policy,
        "statement": claim["statement"],
        "verifiability": claim.get("verifiability", "medium"),
        "metrics": [_norm_metric(m) for m in claim.get("metrics", [])],
        "thresholds": (
            thresholds if thresholds is not None else list(claim.get("thresholds") or [])
        ),
        "required_outputs": required_outputs or dict(DEFAULT_REQUIRED_OUTPUTS),
        "verdict_rules": verdict_rules or {},
    }

    for opt in ("requires_training", "cost", "anchors", "conditions", "matched_variables"):
        if claim.get(opt) is not None:
            spec[opt] = claim[opt]
    for opt in ("claim_uid", "contract_hash", "reproduction_protocol", "verification_contract", "verification", "reward_curves"):
        if claim.get(opt) is not None:
            spec[opt] = claim[opt]

    params = [_norm_param(p) for p in claim.get("params", [])]
    params += _figure_evidence_to_params(bound_figures)
    params += _figure_params_to_entries(figure_params)
    if params:
        spec["params"] = params
    if bound_figures:
        spec["figure_dependencies"] = bound_figures
    if claim.get("input_files"):
        spec["input_files"] = list(claim.get("input_files") or [])
    if reward:
        spec["reward"] = reward

    spec = apply_verification_contract(spec)
    spec = with_spec_hash(spec)
    validate_claim_spec(spec)
    if out_path is not None:
        dump_claim_spec(spec, out_path)
    return spec
