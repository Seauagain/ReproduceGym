"""Deterministic claim scoring and selection for economical reproduction queues."""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_CLAIMS = 3

_COST_ECONOMY = {"S": 4.0, "M": 3.0, "L": 1.5, "XL": 0.5}
_VERIFICATION = {"high": 3.0, "medium": 1.8, "low": 0.5}
_TYPE_VALUE = {
    "mechanism": 2.5,
    "ablation": 2.5,
    "diagnostic": 2.0,
    "eval_only": 1.5,
    "scaling": 1.0,
    "headline": 0.5,
}
_TYPE_RISK = {"headline": 1.5, "scaling": 1.0}


def _rank_value(claim: dict[str, Any]) -> float:
    try:
        rank = int(claim.get("importance_rank") or claim.get("claim_num") or 10)
    except (TypeError, ValueError):
        rank = 10
    # Keep importance meaningful, but bounded so economy can overrule expensive headlines.
    return max(0.0, 5.0 - min(rank, 5))


def _has_metrics(claim: dict[str, Any]) -> bool:
    return bool(claim.get("metrics"))


def _has_params(claim: dict[str, Any]) -> bool:
    return bool(claim.get("params"))


def _risk_penalty(claim: dict[str, Any]) -> float:
    penalty = _TYPE_RISK.get(str(claim.get("claim_type") or ""), 0.0)
    cost = str(claim.get("cost") or "M")
    if cost == "XL":
        penalty += 2.0
    elif cost == "L":
        penalty += 1.0
    if claim.get("requires_training"):
        penalty += 0.8
    if not _has_metrics(claim):
        penalty += 1.2
    if not _has_params(claim):
        penalty += 0.5
    notes = str(claim.get("notes") or "").lower()
    for marker in ("missing", "unspecified", "unavailable", "closed", "subjective", "ambiguous"):
        if marker in notes:
            penalty += 0.4
    return penalty


def score_claim(claim: dict[str, Any]) -> dict[str, Any]:
    """Return score components and final utility for one claim."""
    cost = str(claim.get("cost") or "M")
    verifiability = str(claim.get("verifiability") or "medium")
    claim_type = str(claim.get("claim_type") or "")
    components = {
        "scientific_value": _rank_value(claim),
        "economy": _COST_ECONOMY.get(cost, _COST_ECONOMY["M"]),
        "verification_strength": _VERIFICATION.get(verifiability, _VERIFICATION["medium"]),
        "diagnostic_value": _TYPE_VALUE.get(claim_type, 1.0),
        "feasibility": (1.0 if _has_metrics(claim) else 0.0) + (0.5 if _has_params(claim) else 0.0),
        "dependency_reuse": 0.5 if claim.get("training_group") or claim.get("shared_run_group") else 0.0,
        "risk_penalty": _risk_penalty(claim),
    }
    total = (
        components["scientific_value"]
        + components["economy"]
        + components["verification_strength"]
        + components["diagnostic_value"]
        + components["feasibility"]
        + components["dependency_reuse"]
        - components["risk_penalty"]
    )
    return {"selection_score": round(total, 3), "score_components": components}


def rank_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copied claims ordered by score, with selection metadata."""
    scored: list[dict[str, Any]] = []
    for claim in claims:
        c = dict(claim)
        score = score_claim(c)
        c.update(score)
        c["selection_reason"] = (
            f"score={c['selection_score']}; importance={c.get('importance_rank')}; "
            f"cost={c.get('cost', 'M')}; verifiability={c.get('verifiability', 'medium')}; "
            f"type={c.get('claim_type')}; risk={score['score_components']['risk_penalty']}"
        )
        scored.append(c)
    scored.sort(
        key=lambda c: (
            -float(c["selection_score"]),
            int(c.get("importance_rank") or c.get("claim_num") or 999),
            str(c.get("claim_id") or ""),
        )
    )
    for idx, claim in enumerate(scored, start=1):
        claim["selection_rank"] = idx
    return scored


def select_top_claims(claims: list[dict[str, Any]], max_claims: int | None = DEFAULT_MAX_CLAIMS) -> list[dict[str, Any]]:
    ranked = rank_claims(claims)
    return ranked if max_claims is None else ranked[:max_claims]


def selection_table(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": c.get("claim_id"),
            "display_title": c.get("display_title"),
            "selection_rank": c.get("selection_rank"),
            "selection_score": c.get("selection_score"),
            "score_components": c.get("score_components"),
            "selection_reason": c.get("selection_reason"),
            "cost": c.get("cost"),
            "verifiability": c.get("verifiability"),
            "claim_type": c.get("claim_type"),
            "requires_training": c.get("requires_training"),
        }
        for c in rank_claims(claims)
    ]
