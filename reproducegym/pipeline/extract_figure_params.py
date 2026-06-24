"""Step 2b: read experimental params / curve targets off figures, via a multimodal model.

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


def _normalize_param(
    e: dict[str, Any],
    *,
    default_source: str,
    default_exposure: str = "hidden",
    default_use: str = "reproduction_param",
) -> dict[str, Any] | None:
    if not isinstance(e, dict) or not e.get("name"):
        return None
    exposure = e.get("exposure") or e.get("visibility") or default_exposure
    if exposure not in {"visible", "hidden"}:
        exposure = default_exposure
    entry: dict[str, Any] = {
        "name": e["name"],
        "value": e.get("value"),
        "source": e.get("source") or default_source,
        "status": e.get("status") or "paper_specified",
        "exposure": exposure,
        "use": e.get("use") or default_use,
    }
    for k in ("unit", "confidence", "read_from", "applies_to_claim"):
        if e.get(k) is not None:
            entry[k] = e[k]
    return entry


def _coerce_json(raw: str) -> Any:
    return json.loads(_strip_fence(raw))


def parse_figure_params_json(raw: str, *, default_source: str) -> list[dict[str, Any]]:
    """Parse one figure's VL JSON into normalized, hidden-by-default param entries."""
    data = _coerce_json(raw)
    if isinstance(data, dict):
        data = data.get("params") or data.get("entries") or [data]
    if not isinstance(data, list):
        raise FigureParamError(f"expected a JSON list, got {type(data).__name__}")

    entries: list[dict[str, Any]] = []
    for e in data:
        entry = _normalize_param(e, default_source=default_source)
        if entry is None:
            continue
        entries.append(entry)
    return entries


def parse_figure_evidence_json(
    raw: str,
    *,
    figure: dict[str, Any],
) -> dict[str, Any]:
    """Parse one figure's VL JSON into a structured figure evidence record."""
    data = _coerce_json(raw)
    if isinstance(data, list):
        data = {"params": data}
    if not isinstance(data, dict):
        raise FigureParamError(f"expected a JSON object/list, got {type(data).__name__}")

    default_source = figure.get("figure_ref") or _figure_ref(str(figure.get("image_file", "")))
    # Numbers read off a figure are answer-key material by default (routed into
    # reward/); a param is only exposed to the agent when the model explicitly tags
    # it visibility:visible, which _normalize_param honours over this default.
    params = [
        p for p in (
            _normalize_param(e, default_source=default_source, default_exposure="hidden")
            for e in data.get("params", []) or data.get("entries", []) or []
        )
        if p is not None
    ]
    targets = [
        p for p in (
            _normalize_param(e, default_source=default_source, default_exposure="hidden", default_use="target")
            for e in data.get("targets", []) or []
        )
        if p is not None
    ]
    return {
        "figure_ref": data.get("figure_ref") or figure.get("figure_ref") or default_source,
        "image_file": data.get("image_file") or figure.get("image_file"),
        "source_path": figure.get("source_path"),
        "caption": figure.get("caption", ""),
        "context": figure.get("context", ""),
        "claim_relevance": data.get("claim_relevance", ""),
        "params": params,
        "targets": targets,
        "axis_ranges": data.get("axis_ranges") or {},
        "conditions": data.get("conditions") or [],
        "confidence": data.get("confidence"),
        "read_from": data.get("read_from", ""),
    }


def _load_figures_index(
    figures_dir: Path,
    figures_index: str | Path | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if isinstance(figures_index, list):
        return figures_index
    if figures_index is not None:
        data = json.loads(Path(figures_index).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise FigureParamError("figures_index must contain a JSON list")
        return data
    images = (
        sorted(p for p in figures_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if figures_dir.is_dir()
        else []
    )
    return [
        {
            "figure_ref": _figure_ref(img.name),
            "image_file": img.name,
            "caption": "",
            "context": "",
        }
        for img in images
    ]


def _resolve_image_path(figures_dir: Path, figure: dict[str, Any]) -> Path:
    image_file = figure.get("image_file")
    candidates = []
    if image_file:
        p = Path(str(image_file))
        candidates.append(p if p.is_absolute() else figures_dir / p)
        candidates.append(figures_dir / p.name)
    if figure.get("source_path"):
        candidates.append(Path(str(figure["source_path"])))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FigureParamError(f"figure image not found for {figure.get('figure_ref')}: {image_file}")


def _figure_prompt(prompt: str, figure: dict[str, Any]) -> str:
    return (
        f"{prompt}\n\n"
        f"Figure ref: {figure.get('figure_ref')}\n"
        f"Figure file: {figure.get('image_file')}\n"
        f"Caption: {figure.get('caption', '')}\n"
        f"Nearby markdown context:\n{figure.get('context', '')}\n"
    )


def _norm_ref(ref: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", ref.lower())


def _ref_number(ref: str) -> str | None:
    m = re.search(r"(?:fig(?:ure)?|table)?\s*\.?\s*(\d+)", ref, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _figure_refs_for_claim(claim: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for anchor in claim.get("anchors", []) or []:
        if not isinstance(anchor, dict):
            continue
        if anchor.get("kind") in {"figure", "table"} and anchor.get("ref"):
            refs.add(str(anchor["ref"]))
    return refs


def _figure_matches_ref(figure: dict[str, Any], ref: str) -> bool:
    fig_ref = str(figure.get("figure_ref") or "")
    if not fig_ref:
        return False
    if _norm_ref(fig_ref) == _norm_ref(ref):
        return True
    # Treat panel refs like Fig. 5(a) as matching Fig. 5, without matching Fig. 15.
    return bool(_ref_number(fig_ref) and _ref_number(fig_ref) == _ref_number(ref))


def figures_for_claim(
    claim: dict[str, Any],
    figures_index: str | Path | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only indexed figures explicitly anchored by a claim."""
    figures = _load_figures_index(Path("."), figures_index)
    refs = _figure_refs_for_claim(claim)
    if not refs:
        return []
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fig in figures:
        if any(_figure_matches_ref(fig, ref) for ref in refs):
            key = str(fig.get("image_file") or fig.get("source_path") or fig.get("figure_ref"))
            if key not in seen:
                selected.append(fig)
                seen.add(key)
    return selected


def _claim_figure_prompt(prompt: str, figure: dict[str, Any], claim: dict[str, Any]) -> str:
    claim_focus = {
        "claim_id": claim.get("claim_id"),
        "statement": claim.get("statement"),
        "required_experiments": claim.get("required_experiments"),
        "metrics": claim.get("metrics"),
        "params": claim.get("params"),
        "notes": claim.get("notes"),
    }
    return (
        f"{prompt}\n\n"
        "# Claim focus\n"
        f"{json.dumps(claim_focus, ensure_ascii=False, indent=2)}\n\n"
        "Read this figure only for the claim above. Extract parameters, targets, "
        "conditions, and plot structure relevant to reproducing that claim.\n\n"
        f"Figure ref: {figure.get('figure_ref')}\n"
        f"Figure file: {figure.get('image_file')}\n"
        f"Caption: {figure.get('caption', '')}\n"
        f"Nearby markdown context:\n{figure.get('context', '')}\n"
    )


def extract_figure_evidence(
    figures_dir: str | Path,
    *,
    client: VLClient,
    figures_index: str | Path | list[dict[str, Any]] | None = None,
    prompt_path: str | Path | None = None,
    min_confidence: float = 0.0,
    strict: bool = False,
    out_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read structured figure evidence from every indexed figure."""
    figures_dir = Path(figures_dir)
    prompt = _load_prompt(prompt_path)
    figures = _load_figures_index(figures_dir, figures_index)

    raw_root = Path(raw_dir) if raw_dir is not None else None
    if raw_root is not None:
        raw_root.mkdir(parents=True, exist_ok=True)

    evidence: list[dict[str, Any]] = []
    for figure in figures:
        try:
            img = _resolve_image_path(figures_dir, figure)
            raw = client.read_figure(img, _figure_prompt(prompt, figure))
            if raw_root is not None:
                (raw_root / f"{Path(img).stem}.raw.json").write_text(raw, encoding="utf-8")
            entry = parse_figure_evidence_json(raw, figure=figure)
        except (FigureParamError, json.JSONDecodeError):
            if strict:
                raise
            continue
        conf = entry.get("confidence")
        if conf is not None and conf < min_confidence:
            continue
        evidence.append(entry)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(evidence, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return evidence


def extract_claim_figure_evidence(
    claim: dict[str, Any],
    figures_dir: str | Path,
    *,
    client: VLClient,
    figures_index: str | Path | list[dict[str, Any]] | None = None,
    prompt_path: str | Path | None = None,
    min_confidence: float = 0.0,
    strict: bool = False,
    out_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read evidence only from figures anchored by one claim."""
    figures_dir = Path(figures_dir)
    prompt = _load_prompt(prompt_path)
    figures = figures_for_claim(claim, figures_index or _load_figures_index(figures_dir, None))

    raw_root = Path(raw_dir) if raw_dir is not None else None
    if raw_root is not None:
        raw_root.mkdir(parents=True, exist_ok=True)

    evidence: list[dict[str, Any]] = []
    for figure in figures:
        try:
            img = _resolve_image_path(figures_dir, figure)
            raw = client.read_figure(img, _claim_figure_prompt(prompt, figure, claim))
            if raw_root is not None:
                claim_id = str(claim.get("claim_id") or "claim")
                (raw_root / f"{claim_id}.{Path(img).stem}.raw.json").write_text(raw, encoding="utf-8")
            entry = parse_figure_evidence_json(raw, figure=figure)
        except (FigureParamError, json.JSONDecodeError):
            if strict:
                raise
            continue
        conf = entry.get("confidence")
        if conf is not None and conf < min_confidence:
            continue
        evidence.append(entry)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(evidence, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return evidence


def extract_figure_params(
    figures_dir: str | Path,
    *,
    client: VLClient,
    figures_index: str | Path | list[dict[str, Any]] | None = None,
    prompt_path: str | Path | None = None,
    min_confidence: float = 0.0,
    strict: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read params off every figure in figures_dir; merge into {name: entry}."""
    figures_dir = Path(figures_dir)
    evidence = extract_figure_evidence(
        figures_dir,
        client=client,
        figures_index=figures_index,
        prompt_path=prompt_path,
        min_confidence=min_confidence,
        strict=strict,
    )
    merged: dict[str, dict[str, Any]] = {}
    for fig in evidence:
        entries = list(fig.get("params") or []) + list(fig.get("targets") or [])
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
