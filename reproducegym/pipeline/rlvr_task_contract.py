"""RLVR task contract helpers for PDF-to-verifiable-task builds.

This module owns extraction-stage identity (`claim_uid`), verifier identity
(`contract_hash`), deterministic verification reports, and final build queue
selection. It deliberately keeps claim identity separate from final `claim_id`,
which is assigned only after selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from reproducegym.pipeline.claim_ids import normalize_claim_ids
from reproducegym.pipeline.formula_contract import formulas_problem
from reproducegym.pipeline.synthesize_contract import apply_verification_contract

HASH_LEN = 16


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _norm_statement(statement: Any) -> str:
    return re.sub(r"\s+", " ", str(statement or "").strip().lower())


def _anchor_key(anchor: dict[str, Any]) -> dict[str, str]:
    return {
        "kind": str(anchor.get("kind") or "").strip().lower(),
        "ref": str(anchor.get("ref") or "").strip().lower(),
    }


def _claim_anchors(claim: dict[str, Any]) -> list[dict[str, str]]:
    raw = claim.get("evidence_anchors") or claim.get("anchors") or []
    anchors = [_anchor_key(a) for a in raw if isinstance(a, dict)]
    return sorted(anchors, key=lambda x: (x["kind"], x["ref"]))


def compute_claim_uid(claim: dict[str, Any], *, length: int = HASH_LEN) -> str:
    """Stable claim identity from normalized statement + sorted evidence anchors."""

    payload = {
        "statement": _norm_statement(claim.get("statement")),
        "anchors": _claim_anchors(claim),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"clm_{digest[:length]}"


def ensure_claim_uid(claim: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(claim)
    out["claim_uid"] = str(out.get("claim_uid") or compute_claim_uid(out))
    return out


def _verification_contract(claim: dict[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(claim.get("verification_contract") or {})
    for key in ("conditions", "metrics", "params", "thresholds"):
        contract.setdefault(key, [])
    contract.setdefault("verdict_rules", {})
    return contract


def compute_contract_hash(
    claim: dict[str, Any],
    *,
    accepted_targets: list[dict[str, Any]] | None = None,
    length: int = HASH_LEN,
) -> str:
    """Hash only verifier-affecting facts, not reproduction protocol prose."""

    contract = _verification_contract(claim)
    payload = {
        "verification_contract": contract,
        "accepted_targets": accepted_targets if accepted_targets is not None else contract.get("thresholds", []),
        "evidence_refs": _claim_anchors(claim),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:length]


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_direction(metric_name: str, claim: dict[str, Any]) -> str:
    contract = _verification_contract(claim)
    for metric in contract.get("metrics") or claim.get("metrics") or []:
        if isinstance(metric, dict) and metric.get("name") == metric_name:
            direction = str(metric.get("direction") or "higher_is_better").strip().lower()
            return "lower_is_better" if direction in {"lower_is_better", "lower", "smaller_is_better", "less_is_better"} else "higher_is_better"
    return "higher_is_better"


def _threshold_has_scientific_target(threshold: dict[str, Any]) -> bool:
    return _as_float(threshold.get("target_value")) is not None


def reward_curve_from_threshold(
    threshold: dict[str, Any],
    *,
    direction: str,
) -> dict[str, Any] | None:
    metric = threshold.get("metric")
    pass_threshold = _as_float(threshold.get("pass_threshold"))
    if not metric or pass_threshold is None:
        return None
    target = _as_float(threshold.get("target_value"))
    if target is None:
        return None
    tol = _as_float(threshold.get("tolerance_abs"))
    if tol is None:
        tolerance = threshold.get("tolerance")
        if isinstance(tolerance, dict):
            rel = _as_float(tolerance.get("rel"))
            abs_tol = _as_float(tolerance.get("abs"))
            tol = abs_tol if abs_tol is not None else (abs(target) * rel if rel is not None else None)
        else:
            rel = _as_float(tolerance)
            tol = abs(target) * rel if rel is not None else None
    if tol is None or tol <= 0:
        tol = abs(target - pass_threshold) or max(abs(target) * 0.15, 0.1)
    if direction == "lower_is_better":
        if pass_threshold < target:
            return None
        curve_pass = pass_threshold if pass_threshold != target else target + tol
        fail = curve_pass + abs(curve_pass - target or tol)
    else:
        if pass_threshold > target:
            return None
        curve_pass = pass_threshold if pass_threshold != target else target - tol
        fail = curve_pass - abs(curve_pass - target or tol)
    rationale = str(threshold.get("rationale") or "paper target plus tolerance reward curve")
    return {
        "metric": str(metric),
        "direction": direction,
        "points": [
            {"value": float(f"{fail:.12g}"), "reward": 0.0},
            {"value": float(f"{curve_pass:.12g}"), "reward": 0.5},
            {"value": float(f"{target:.12g}"), "reward": 1.0},
        ],
        "source": threshold.get("target_evidence") or {"source": threshold.get("source")},
        "rationale": rationale,
    }


def reward_curves_from_thresholds(
    claim: dict[str, Any],
    thresholds: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    curves: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        metric = threshold.get("metric")
        if not metric:
            continue
        curve = reward_curve_from_threshold(
            threshold,
            direction=_metric_direction(str(metric), claim),
        )
        if curve is not None:
            curves[str(metric)] = curve
    return curves


def _contract_to_spec_like(claim: dict[str, Any]) -> dict[str, Any]:
    contract = _verification_contract(claim)
    return {
        "claim_id": claim.get("claim_id") or "c001_temp",
        "paper_id": claim.get("paper_id") or "paper",
        "statement": claim.get("statement", ""),
        "claim_type": claim.get("claim_type", "mechanism"),
        "metrics": copy.deepcopy(contract.get("metrics") or []),
        "conditions": copy.deepcopy(contract.get("conditions") or []),
        "params": copy.deepcopy(contract.get("params") or []),
        "thresholds": copy.deepcopy(contract.get("thresholds") or []),
        "verdict_rules": copy.deepcopy(contract.get("verdict_rules") or {}),
        "verification": copy.deepcopy(contract.get("verification") or {}),
    }


def _report_for_claim(claim: dict[str, Any]) -> dict[str, Any]:
    claim = ensure_claim_uid(claim)
    spec_like = apply_verification_contract(_contract_to_spec_like(claim))
    verification = dict(spec_like.get("verification") or {})
    accepted_targets = list(spec_like.get("thresholds") or [])
    reward_curves = reward_curves_from_thresholds(spec_like, accepted_targets)
    hash_claim = {
        **claim,
        "verification_contract": {
            **_verification_contract(claim),
            "metrics": copy.deepcopy(spec_like.get("metrics") or []),
            "thresholds": copy.deepcopy(accepted_targets),
        },
    }
    contract_hash = compute_contract_hash(hash_claim, accepted_targets=accepted_targets)
    formula_problems = formulas_problem(spec_like.get("metrics") or [])
    metric_names = [
        str(metric.get("name"))
        for metric in spec_like.get("metrics") or []
        if isinstance(metric, dict) and metric.get("name")
    ]
    target_metrics = {
        str(threshold.get("metric"))
        for threshold in accepted_targets
        if threshold.get("metric") and _threshold_has_scientific_target(threshold)
    }
    missing_reward_curves = sorted(set(metric_names) - set(reward_curves))
    if formula_problems:
        verification = {
            "mode": "unverifiable",
            "pool": "exploration",
            "reason": "unsupported verifier formula(s): "
            + "; ".join(f"{name}: {problem}" for name, problem in formula_problems.items()),
            "formula_problems": formula_problems,
        }
    elif verification.get("pool") == "rlvr" and (
        not metric_names
        or not set(metric_names).issubset(target_metrics)
        or missing_reward_curves
    ):
        reason_bits: list[str] = []
        no_target = sorted(set(metric_names) - target_metrics)
        if no_target:
            reason_bits.append(
                "missing paper-grounded target_value for metric(s): " + ", ".join(no_target)
            )
        if missing_reward_curves:
            reason_bits.append(
                "missing valid reward curve for metric(s): " + ", ".join(missing_reward_curves)
            )
        if not metric_names:
            reason_bits.append("claim has no primary metrics")
        verification = {
            **verification,
            "mode": "unverifiable",
            "pool": "exploration",
            "reason": "; ".join(reason_bits),
        }
    pool = verification.get("pool") or "exploration"
    reason = verification.get("reason")
    if pool == "exploration" and not reason:
        verification["reason"] = "no executable verifier contract"
    return {
        "claim_uid": claim["claim_uid"],
        "statement": claim.get("statement"),
        "claim_type": claim.get("claim_type"),
        "source_mode": claim.get("source_mode"),
        "likely_pool": claim.get("likely_pool"),
        "verification": verification,
        "buildable": not formula_problems,
        "metrics": copy.deepcopy(spec_like.get("metrics") or []),
        "diagnostic_metrics": copy.deepcopy(spec_like.get("diagnostic_metrics") or []),
        "accepted_targets": accepted_targets,
        "reward_curves": reward_curves,
        "rejected_targets": [],
        "contract_hash": contract_hash,
        "score_components": {
            "importance_rank": claim.get("importance_rank"),
            "pool": verification.get("pool"),
            "n_metrics": len(metric_names),
            "n_thresholds": len(accepted_targets),
            "n_reward_curves": len(reward_curves),
        },
        "selection_rationale": (
            "eligible for RLVR"
            if verification.get("pool") == "rlvr" and not formula_problems
            else verification.get("reason", "exploration")
        ),
    }


def build_claim_verification_report(refined_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one deterministic verification report row per refined claim."""

    return [_report_for_claim(claim) for claim in refined_claims]


def _rank_value(claim: dict[str, Any]) -> int:
    try:
        return int(claim.get("importance_rank") or 999)
    except (TypeError, ValueError):
        return 999


def _report_by_uid(report: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["claim_uid"]): item for item in report}


def _merge_report_fields(claim: dict[str, Any], report_item: dict[str, Any]) -> dict[str, Any]:
    out = ensure_claim_uid(claim)
    out["contract_hash"] = report_item["contract_hash"]
    out["verification"] = copy.deepcopy(report_item.get("verification") or {})
    if report_item.get("metrics") is not None:
        out["metrics"] = copy.deepcopy(report_item.get("metrics") or [])
    if report_item.get("diagnostic_metrics"):
        out["diagnostic_metrics"] = copy.deepcopy(report_item.get("diagnostic_metrics") or [])
    out["accepted_targets"] = copy.deepcopy(report_item.get("accepted_targets") or [])
    contract = copy.deepcopy(out.get("verification_contract") or {})
    if report_item.get("metrics") is not None:
        contract["metrics"] = copy.deepcopy(report_item.get("metrics") or [])
    contract["thresholds"] = copy.deepcopy(out["accepted_targets"])
    out["verification_contract"] = contract
    out["reward_curves"] = copy.deepcopy(report_item.get("reward_curves") or {})
    out["selection_rationale"] = report_item.get("selection_rationale")
    return out


def select_claims_for_build(
    refined_claims: list[dict[str, Any]],
    verification_report: list[dict[str, Any]],
    *,
    max_claims: int | None = 3,
) -> list[dict[str, Any]]:
    """Select only claims with paper-grounded RLVR reward contracts."""

    by_uid = _report_by_uid(verification_report)
    enriched = []
    for claim in refined_claims:
        c = ensure_claim_uid(claim)
        report_item = by_uid.get(c["claim_uid"])
        if report_item is None:
            continue
        enriched.append(_merge_report_fields(c, report_item))
    buildable = [c for c in enriched if by_uid.get(c["claim_uid"], {}).get("buildable", True)]
    rlvr = [c for c in buildable if (c.get("verification") or {}).get("pool") == "rlvr"]
    ordered = sorted(rlvr, key=_rank_value)
    return ordered if max_claims is None else ordered[:max_claims]


def assign_final_claim_ids(selected_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign final cNNN claim IDs while preserving claim_uid/contract_hash."""

    normalized = normalize_claim_ids(selected_claims)
    by_uid = {c["claim_uid"]: c for c in selected_claims if c.get("claim_uid")}
    out = []
    for claim in normalized:
        source = by_uid.get(claim.get("claim_uid"), {})
        if source.get("contract_hash"):
            claim["contract_hash"] = source["contract_hash"]
        if source.get("verification"):
            claim["verification"] = copy.deepcopy(source["verification"])
        if source.get("reward_curves"):
            claim["reward_curves"] = copy.deepcopy(source["reward_curves"])
        out.append(claim)
    return out
