"""Canonical claim-spec hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

HASH_LEN = 12
# spec_hash identifies task-affecting content only. The multimodal figure reader
# emits provenance/quality metadata that varies between otherwise-identical builds
# -- confidence is non-deterministic and read_from is free text -- so they must be
# stripped before hashing, or rebuilding the same paper would mint a fresh
# task-version dir every time and orphan prior runs.
EXCLUDED_HASH_FIELDS = {"spec_hash", "confidence", "read_from"}


def _strip_generated(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_generated(v)
            for k, v in sorted(value.items())
            if k not in EXCLUDED_HASH_FIELDS
        }
    if isinstance(value, list):
        return [_strip_generated(v) for v in value]
    return value


def canonical_spec_payload(spec: dict[str, Any]) -> str:
    normalized = _strip_generated(copy.deepcopy(spec))
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_spec_hash(spec: dict[str, Any], *, length: int = HASH_LEN) -> str:
    digest = hashlib.sha256(canonical_spec_payload(spec).encode("utf-8")).hexdigest()
    return digest[:length]


def with_spec_hash(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    out["spec_hash"] = compute_spec_hash(out)
    return out
