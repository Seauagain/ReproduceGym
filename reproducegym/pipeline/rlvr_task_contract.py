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
    out["claim_uid"] = compute_claim_uid(out)
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
    contract_hash = compute_contract_hash(claim, accepted_targets=accepted_targets)
    formula_problems = formulas_problem(spec_like.get("metrics") or [])
    if formula_problems:
        verification = {
            "mode": "unverifiable",
            "pool": "exploration",
            "reason": "unsupported verifier formula(s): "
            + "; ".join(f"{name}: {problem}" for name, problem in formula_problems.items()),
            "formula_problems": formula_problems,
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
        "accepted_targets": accepted_targets,
        "rejected_targets": [],
        "contract_hash": contract_hash,
        "score_components": {
            "importance_rank": claim.get("importance_rank"),
            "pool": verification.get("pool"),
            "n_metrics": len(spec_like.get("metrics") or []),
            "n_thresholds": len(accepted_targets),
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
    out["accepted_targets"] = copy.deepcopy(report_item.get("accepted_targets") or [])
    out["selection_rationale"] = report_item.get("selection_rationale")
    return out


def select_claims_for_build(
    refined_claims: list[dict[str, Any]],
    verification_report: list[dict[str, Any]],
    *,
    max_claims: int | None = 3,
) -> list[dict[str, Any]]:
    """Prefer RLVR claims, then high-value exploration claims if slots remain."""

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
    exploration = [c for c in buildable if (c.get("verification") or {}).get("pool") != "rlvr"]
    ordered = sorted(rlvr, key=_rank_value) + sorted(exploration, key=_rank_value)
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
        out.append(claim)
    return out
