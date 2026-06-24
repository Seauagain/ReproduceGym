"""Deterministic claim IDs.

LLM-generated IDs are useful hints but not stable enough for downstream task
directories. New builds use importance-ordered IDs of the form c001_short_slug.
"""

from __future__ import annotations

import re
from typing import Any

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, fallback: str = "claim", max_words: int = 5) -> str:
    words = [w for w in _SLUG_RE.sub(" ", text.lower()).split() if w]
    if not words:
        words = [fallback]
    return "_".join(words[:max_words])


def _sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
    idx, claim = item
    rank = claim.get("importance_rank")
    try:
        rank_int = int(rank)
    except (TypeError, ValueError):
        rank_int = idx + 1
    return rank_int, idx


def normalize_claim_ids(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copied claims with stable cNNN_slug IDs in importance order."""

    ordered = sorted(enumerate(claims), key=_sort_key)
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for new_idx, (_old_idx, claim) in enumerate(ordered, start=1):
        c = dict(claim)
        if c.get("claim_id") and not c.get("source_claim_id"):
            c["source_claim_id"] = c["claim_id"]
        title = str(c.get("display_title") or c.get("claim_id") or c.get("statement") or "claim")
        base = slugify(str(c.get("claim_slug") or title))
        slug = base
        suffix = 2
        while slug in used:
            slug = f"{base}_{suffix}"
            suffix += 1
        used.add(slug)
        c["claim_num"] = new_idx
        c["claim_slug"] = slug
        c["display_title"] = str(c.get("display_title") or title).strip()
        c["importance_rank"] = int(c.get("importance_rank") or new_idx)
        c["claim_id"] = f"c{new_idx:03d}_{slug}"
        out.append(c)
    return out


def claim_slug_from_id(claim_id: str) -> str:
    m = re.match(r"^c\d{3}_(.+)$", claim_id)
    return m.group(1) if m else slugify(claim_id)
