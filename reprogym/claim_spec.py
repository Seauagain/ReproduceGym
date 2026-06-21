"""Canonical claim-spec loading + validation (single source of truth).

A claim spec is the human-reviewed YAML/JSON record for ONE reproducible claim
(`sandboxes/<paper>/claims/<claim_id>.yaml`). Everything else in a task is
rendered from it. This module is the gate that keeps a spec well-formed before
anything downstream (render_task, validate_task) trusts it.

Validation is driven by `schema/claim_spec.schema.json` (draft-07) so the schema
stays the single definition of the field set; helpers here only read it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

SCHEMA_PATH = Path(__file__).parent / "schema" / "claim_spec.schema.json"


class ClaimSpecError(ValueError):
    """Raised when a claim spec fails schema validation."""


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def enum_values(field: str, *, schema: dict | None = None) -> list[str]:
    """Return the allowed enum values for a top-level field (DRY with the schema)."""
    schema = schema if schema is not None else load_schema()
    prop = schema.get("properties", {}).get(field, {})
    values = prop.get("enum")
    if values is None:
        raise KeyError(f"{field!r} is not an enum field in the claim spec schema")
    return list(values)


def iter_validation_errors(spec: Any, *, schema: dict | None = None) -> list[str]:
    """Return human-readable validation errors (empty list = valid)."""
    schema = schema if schema is not None else load_schema()
    validator = jsonschema.Draft7Validator(schema)
    messages: list[str] = []
    for err in sorted(validator.iter_errors(spec), key=lambda e: list(e.absolute_path)):
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        messages.append(f"{location}: {err.message}")
    return messages


def validate_claim_spec(spec: Any, *, schema: dict | None = None) -> None:
    """Raise ClaimSpecError if the spec is invalid; otherwise return None."""
    errors = iter_validation_errors(spec, schema=schema)
    if errors:
        raise ClaimSpecError(
            "invalid claim spec (" + str(len(errors)) + " error(s)): " + "; ".join(errors)
        )


def load_claim_spec(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Load a claim spec from YAML or JSON and (optionally) validate it."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ClaimSpecError(f"claim spec must be a mapping, got {type(data).__name__}")
    if validate:
        validate_claim_spec(data)
    return data


def dump_claim_spec(spec: dict[str, Any], path: str | Path, *, validate: bool = True) -> Path:
    """Validate then write a claim spec as YAML. Returns the written path."""
    if validate:
        validate_claim_spec(spec)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out
