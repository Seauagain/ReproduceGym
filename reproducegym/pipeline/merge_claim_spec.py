"""Step 2c: merge extracted claim (+ figure params) into a canonical claim spec.

The bridge between extract_claims (step-1 claim dicts) and render_task (which needs
a schema-valid canonical spec). It selects only schema-allowed fields, fills the
required scaffolding (paper_id, tier, exposure_policy, required_outputs, ...) with
explicit defaults, folds in figure-derived params, then validates against
schema/claim_spec.schema.json. The result is the single source of truth and may be
written to sandboxes/<paper>/claims/<claim_id>.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reproducegym.claim_spec import dump_claim_spec, validate_claim_spec

DEFAULT_REQUIRED_OUTPUTS = {"files": ["output/result.json", "output/metrics.csv"]}
DEFAULT_TIER = "T2_proxy"
DEFAULT_EXPOSURE_POLICY = "v0_full_paper_public"


class MergeError(ValueError):
    pass


def _norm_metric(m: dict[str, Any]) -> dict[str, Any]:
    if not m.get("name"):
        raise MergeError("metric missing 'name'")
    out = {
        "name": m["name"],
        "formula": m.get("formula") or f"recompute({m['name']})",
        "direction": m.get("direction") or "higher_is_better",
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
    out: dict[str, Any] = {"name": p["name"], "status": status}
    for k in ("value", "unit", "source", "applies_to_claim", "local_substitute_allowed",
              "affects_strict_reproduction", "required_for_strict", "exposure"):
        if k in p and p[k] is not None:
            out[k] = p[k]
    return out


def _figure_params_to_entries(figure_params: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not figure_params:
        return []
    entries = []
    for name, raw in figure_params.items():
        raw = dict(raw) if isinstance(raw, dict) else {"value": raw}
        raw.setdefault("name", name)
        raw.setdefault("source", raw.get("source", "figure (Qwen-VL)"))
        raw.setdefault("status", "paper_specified")
        entries.append(_norm_param(raw))
    return entries


def merge_claim_spec(
    claim: dict[str, Any],
    figure_params: dict[str, Any] | None = None,
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

    spec: dict[str, Any] = {
        "claim_id": claim["claim_id"],
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

    params = [_norm_param(p) for p in claim.get("params", [])]
    params += _figure_params_to_entries(figure_params)
    if params:
        spec["params"] = params
    if reward:
        spec["reward"] = reward

    validate_claim_spec(spec)
    if out_path is not None:
        dump_claim_spec(spec, out_path)
    return spec
