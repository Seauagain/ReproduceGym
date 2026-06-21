"""Scrub secrets from a recorded trajectory before it is persisted / 回流训练.

Credentials (agent API keys, MetaX/Bohrium access keys) are injected into the
sandbox env so the in-sandbox agent can authenticate -- which means they can
surface in captured stdout / tool calls / observations. Because trajectories feed
model training, secret VALUES must be masked at the recording boundary. Never
rely on the agent not echoing them.

Matching is literal value replacement (longest-first to handle overlaps); values
shorter than MIN_SECRET_LEN are ignored so we don't shred ordinary text.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

DEFAULT_MASK = "\u00abREDACTED\u00bb"
MIN_SECRET_LEN = 6


def collect_secrets(env: Mapping[str, str], keys: Iterable[str]) -> list[str]:
    """Pick maskable secret values out of an env mapping for the given keys."""
    out: list[str] = []
    for k in keys:
        v = env.get(k)
        if v and len(v) >= MIN_SECRET_LEN:
            out.append(v)
    return out


def _maskable(secrets: Iterable[str]) -> list[str]:
    uniq = {s for s in secrets if s and len(s) >= MIN_SECRET_LEN}
    return sorted(uniq, key=len, reverse=True)


def redact_text(text: Any, secrets: Iterable[str], *, mask: str = DEFAULT_MASK) -> Any:
    if not isinstance(text, str):
        return text
    for s in _maskable(secrets):
        if s in text:
            text = text.replace(s, mask)
    return text


def redact_obj(obj: Any, secrets: Iterable[str], *, mask: str = DEFAULT_MASK) -> Any:
    secrets = list(secrets)
    if isinstance(obj, str):
        return redact_text(obj, secrets, mask=mask)
    if isinstance(obj, dict):
        return {k: redact_obj(v, secrets, mask=mask) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, secrets, mask=mask) for v in obj]
    return obj


def redact_trajectory(traj, secrets: Iterable[str], *, mask: str = DEFAULT_MASK):
    """Mask secret values across all trajectory events + meta, in place."""
    secrets = _maskable(secrets)
    if not secrets:
        return traj
    traj.events = [redact_obj(e, secrets, mask=mask) for e in traj.events]
    traj.meta = redact_obj(traj.meta, secrets, mask=mask)
    return traj
