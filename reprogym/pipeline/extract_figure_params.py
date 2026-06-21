"""Step 2b: read experimental params / curve targets off figures, via Qwen-VL.

Many RL papers report target numbers ONLY in figures. This runs once over a
paper's figures (prompts/extract_figure_params.md) and returns a reusable mapping
of {name: {value, source: 'Fig. N', status, exposure}} that claim specs reference
through merge_claim_spec. The FIGURE is public, but the numbers read off it are
answer-key material, so entries are tagged exposure:hidden (routed into reward/).

The VL client is injected (a .read_figure(image_path, prompt) -> str), so parsing
and merging are unit-tested without the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_PATH = PROMPTS_DIR / "extract_figure_params.md"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class VLClient(Protocol):
    def read_figure(self, image_path: Any, prompt: str) -> str: ...


class FigureParamError(ValueError):
    pass


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    m = re.match(r"^```[a-zA-Z0-9]*\s*\n(.*)\n```$", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def _figure_ref(filename: str) -> str:
    m = re.search(r"(\d+)", Path(filename).stem)
    return f"Fig. {m.group(1)}" if m else f"Fig. {Path(filename).stem}"


def _load_prompt(prompt_path: str | Path | None) -> str:
    return Path(prompt_path or PROMPT_PATH).read_text(encoding="utf-8")


def parse_figure_params_json(raw: str, *, default_source: str) -> list[dict[str, Any]]:
    """Parse one figure's VL JSON into normalized, hidden-by-default param entries."""
    data = json.loads(_strip_fence(raw))
    if isinstance(data, dict):
        data = data.get("params") or data.get("entries") or [data]
    if not isinstance(data, list):
        raise FigureParamError(f"expected a JSON list, got {type(data).__name__}")

    entries: list[dict[str, Any]] = []
    for e in data:
        if not isinstance(e, dict) or not e.get("name"):
            continue
        entry: dict[str, Any] = {
            "name": e["name"],
            "value": e.get("value"),
            "source": e.get("source") or default_source,
            "status": "paper_specified",
            "exposure": "hidden",
        }
        for k in ("unit", "confidence", "read_from"):
            if e.get(k) is not None:
                entry[k] = e[k]
        entries.append(entry)
    return entries


def extract_figure_params(
    figures_dir: str | Path,
    *,
    client: VLClient,
    prompt_path: str | Path | None = None,
    min_confidence: float = 0.0,
    strict: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read params off every figure in figures_dir; merge into {name: entry}."""
    figures_dir = Path(figures_dir)
    images = (
        sorted(p for p in figures_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if figures_dir.is_dir()
        else []
    )
    prompt = _load_prompt(prompt_path)

    merged: dict[str, dict[str, Any]] = {}
    for img in images:
        ref = _figure_ref(img.name)
        raw = client.read_figure(img, f"{prompt}\n\n(figure file: {img.name}, ref: {ref})")
        try:
            entries = parse_figure_params_json(raw, default_source=ref)
        except (FigureParamError, json.JSONDecodeError):
            if strict:
                raise
            continue
        for e in entries:
            conf = e.get("confidence")
            if conf is not None and conf < min_confidence:
                continue
            name = e["name"]
            if name in merged and (e.get("confidence") or 0) <= (merged[name].get("confidence") or 0):
                continue
            merged[name] = e

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return merged
