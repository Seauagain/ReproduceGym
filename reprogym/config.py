"""Lightweight .env loading + client construction (no external deps for parsing).

Secrets live in the repo-root .env (gitignored). We parse it ourselves so tests
and tooling don't need python-dotenv, and so we never accidentally require the
network just to import a module.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_env_text(text: str) -> dict[str, str]:
    """Parse .env text into a dict, resolving ${VAR} references within the file."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        # strip trailing inline comment for unquoted values
        if raw and raw[0] not in {'"', "'"}:
            raw = raw.split(" #", 1)[0].rstrip()
        value = _unquote(raw)
        value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda g: out.get(g.group(1), os.environ.get(g.group(1), "")), value)
        out[key] = value
    return out


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load .env into os.environ (without overriding existing vars by default).

    Returns the parsed mapping. Missing file -> empty dict (no error).
    """
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not env_path.is_file():
        return {}
    parsed = parse_env_text(env_path.read_text(encoding="utf-8"))
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"required env var {key!r} is not set (check .env)")
    return value
