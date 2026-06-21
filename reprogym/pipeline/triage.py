"""Step 2.5: triage extracted claims -> which become sandbox tasks + the v0.

Scores claims (scientific value, training value, verifiability, cost, info
completeness, diversity) via prompts/claim_triage.md and produces a triage
decision (build[]/defer[]/v0/rationale). Only build[] claims proceed to
merge_claim_spec + render_task. The resource profile is derived deterministically
from per-claim cost/requires_training (no LLM) for budgeting.

The LLM client is injected (.complete(prompt) -> str) so parsing/validation is
unit-tested offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_PATH = PROMPTS_DIR / "claim_triage.md"


class LLMClient(Protocol):
    def complete(self, prompt: str, **kwargs: Any) -> str: ...


class TriageError(ValueError):
    pass


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    m = re.match(r"^```[a-zA-Z0-9]*\s*\n(.*)\n```$", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def build_triage_prompt(claims: list[dict], *, prompt_path: str | Path | None = None) -> str:
    instructions = Path(prompt_path or PROMPT_PATH).read_text(encoding="utf-8")
    return f"{instructions}\n\n---\n# CLAIMS (JSON)\n\n{json.dumps(claims, ensure_ascii=False, indent=2)}\n"


def parse_triage_json(raw: str, claim_ids: list[str]) -> dict[str, Any]:
    data = json.loads(_strip_fence(raw))
    if not isinstance(data, dict):
        raise TriageError(f"expected a triage object, got {type(data).__name__}")

    known = set(claim_ids)
    build = list(data.get("build", []))
    defer: list[dict[str, str]] = []
    for d in data.get("defer", []):
        if isinstance(d, str):
            defer.append({"claim_id": d, "reason": ""})
        elif isinstance(d, dict) and d.get("claim_id"):
            defer.append({"claim_id": d["claim_id"], "reason": d.get("reason", "")})
    v0 = data.get("v0")

    for cid in build:
        if cid not in known:
            raise TriageError(f"build id {cid!r} is not a known claim")
    for d in defer:
        if d["claim_id"] not in known:
            raise TriageError(f"defer id {d['claim_id']!r} is not a known claim")
    if v0 is not None and v0 not in build:
        raise TriageError(f"v0 {v0!r} must be among build[]")

    return {
        "build": build,
        "defer": defer,
        "v0": v0,
        "rationale": data.get("rationale", ""),
        "scores": data.get("scores", {}),
    }


def triage(
    claims: list[dict],
    *,
    client: LLMClient,
    prompt_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not claims:
        raise TriageError("no claims to triage")
    prompt = build_triage_prompt(claims, prompt_path=prompt_path)
    raw = client.complete(prompt)
    result = parse_triage_json(raw, [c["claim_id"] for c in claims])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paper_triage.yaml").write_text(
            yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    return result


def write_resource_profile(claims: list[dict], out_dir: str | Path) -> Path:
    """Deterministically aggregate per-claim cost/requires_training for budgeting."""
    per_claim: dict[str, dict[str, Any]] = {}
    by_cost: dict[str, int] = {}
    requires_training = 0
    for c in claims:
        cid = c["claim_id"]
        cost = c.get("cost", "unknown")
        rt = bool(c.get("requires_training", False))
        per_claim[cid] = {
            "cost": cost,
            "requires_training": rt,
            "verifiability": c.get("verifiability", "unknown"),
            "claim_type": c.get("claim_type", "unknown"),
        }
        by_cost[cost] = by_cost.get(cost, 0) + 1
        if rt:
            requires_training += 1

    profile = {
        "claims": per_claim,
        "totals": {
            "n_claims": len(claims),
            "by_cost": by_cost,
            "requires_training": requires_training,
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resource_profile.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
