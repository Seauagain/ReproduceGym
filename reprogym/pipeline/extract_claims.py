"""Step 2a: extract reproducible claims from paper.md, via an LLM (Claude).

A single structured transformation driven by `prompts/extract_claims.md`. Given
the paper markdown, the LLM returns a JSON list of claim objects (statement,
anchors, claim_type, metrics, requires_training, cost, verifiability, text-stated
params, notes). Figure-only numbers are filled later by the figure-param pass and
folded into the canonical claim spec by `merge_claim_spec`.

The LLM is injected as a `client` with a `.complete(prompt) -> str` method so the
extraction logic (prompt assembly, JSON parsing, structural checks) is testable
without any network call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from reprogym.claim_spec import enum_values

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_PATH = PROMPTS_DIR / "extract_claims.md"

# Minimal keys an extracted claim must carry before it can feed merge_claim_spec.
REQUIRED_CLAIM_KEYS = ("claim_id", "statement", "claim_type")
_CLAIM_ID_RE = re.compile(r"^[a-z0-9_]+$")


class LLMClient(Protocol):
    def complete(self, prompt: str, **kwargs: Any) -> str: ...


class ExtractError(ValueError):
    """Raised when the LLM output cannot be parsed into valid claims."""


def _load_prompt(prompt_path: str | Path | None) -> str:
    path = Path(prompt_path) if prompt_path is not None else PROMPT_PATH
    return path.read_text(encoding="utf-8")


def build_prompt(paper_md: str, *, prompt_path: str | Path | None = None) -> str:
    """Assemble the extraction prompt: instructions + the paper markdown."""
    instructions = _load_prompt(prompt_path)
    return f"{instructions}\n\n---\n# PAPER (markdown)\n\n{paper_md}\n"


def _strip_code_fence(raw: str) -> str:
    """Drop a leading/trailing ```json ... ``` fence if the model added one."""
    text = raw.strip()
    fence = re.match(r"^```[a-zA-Z0-9]*\s*\n(.*)\n```$", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def parse_claims_json(raw: str) -> list[dict[str, Any]]:
    """Parse + structurally validate the LLM's JSON list of claims."""
    text = _strip_code_fence(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractError(f"LLM did not return valid JSON: {exc}") from exc

    if isinstance(data, dict) and "claims" in data:
        data = data["claims"]
    if not isinstance(data, list):
        raise ExtractError(f"expected a JSON list of claims, got {type(data).__name__}")
    if not data:
        raise ExtractError("LLM returned an empty claim list")

    allowed_types = set(enum_values("claim_type"))
    seen_ids: set[str] = set()
    claims: list[dict[str, Any]] = []
    for i, claim in enumerate(data):
        if not isinstance(claim, dict):
            raise ExtractError(f"claim[{i}] is not an object")
        missing = [k for k in REQUIRED_CLAIM_KEYS if not claim.get(k)]
        if missing:
            raise ExtractError(f"claim[{i}] missing required key(s): {', '.join(missing)}")
        cid = claim["claim_id"]
        if not _CLAIM_ID_RE.match(cid):
            raise ExtractError(f"claim[{i}] claim_id {cid!r} must match [a-z0-9_]+")
        if cid in seen_ids:
            raise ExtractError(f"duplicate claim_id {cid!r}")
        seen_ids.add(cid)
        if claim["claim_type"] not in allowed_types:
            raise ExtractError(
                f"claim[{i}] claim_type {claim['claim_type']!r} not in {sorted(allowed_types)}"
            )
        claims.append(claim)
    return claims


def extract_claims(
    paper_md: str | Path,
    *,
    client: LLMClient,
    prompt_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Extract reproducible claims from paper markdown using an LLM client.

    `paper_md` may be the markdown text itself or a path to a .md file.
    """
    candidate = Path(paper_md) if isinstance(paper_md, (str, Path)) else None
    if isinstance(paper_md, Path) or (
        isinstance(paper_md, str) and "\n" not in paper_md and candidate and candidate.is_file()
    ):
        text = candidate.read_text(encoding="utf-8")
    else:
        text = str(paper_md)

    prompt = build_prompt(text, prompt_path=prompt_path)
    raw = client.complete(prompt)
    if not isinstance(raw, str):
        raise ExtractError(f"client.complete must return str, got {type(raw).__name__}")
    return parse_claims_json(raw)
