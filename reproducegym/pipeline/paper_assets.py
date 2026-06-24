"""Paper asset indexing: Markdown -> local figure inventory.

Claim extraction is only useful if the model can see the same figure evidence a
human would use. This module makes image availability explicit and auditable:
Markdown image references are resolved to local files, copied into a stable
figures/ directory, and indexed with nearby caption/context for multimodal parsing.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
_FIG_REF_RE = re.compile(r"\b(?:fig(?:ure)?\.?\s*)([A-Za-z]?\d+[A-Za-z]?)\b", re.IGNORECASE)


class PaperAssetError(ValueError):
    """Raised when a paper cannot provide local figure assets."""


@dataclass(frozen=True)
class FigureAsset:
    figure_ref: str
    image_file: str
    source_path: str
    alt_text: str = ""
    caption: str = ""
    context: str = ""
    markdown_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_ref": self.figure_ref,
            "image_file": self.image_file,
            "source_path": self.source_path,
            "alt_text": self.alt_text,
            "caption": self.caption,
            "context": self.context,
            "markdown_path": self.markdown_path,
        }


def _clean_uri(raw: str) -> str:
    uri = raw.strip().strip("<>").strip("\"'")
    return unquote(uri.split("#", 1)[0].split("?", 1)[0])


def _is_remote(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme in {"http", "https", "data"}


def _iter_image_refs(markdown: str) -> list[dict[str, str | int]]:
    refs: list[dict[str, str | int]] = []
    for m in _MD_IMAGE_RE.finditer(markdown):
        refs.append({"alt": m.group(1).strip(), "uri": _clean_uri(m.group(2)), "pos": m.start()})
    for m in _HTML_IMAGE_RE.finditer(markdown):
        refs.append({"alt": "", "uri": _clean_uri(m.group(1)), "pos": m.start()})
    return sorted(refs, key=lambda x: int(x["pos"]))


def _line_window(markdown: str, pos: int, *, before: int = 2, after: int = 4) -> list[str]:
    prefix = markdown[:pos]
    line_no = prefix.count("\n")
    lines = markdown.splitlines()
    lo = max(0, line_no - before)
    hi = min(len(lines), line_no + after + 1)
    return lines[lo:hi]


def _caption_from_window(lines: list[str], alt: str) -> str:
    candidates = [ln.strip() for ln in lines if ln.strip()]
    for line in candidates:
        if _FIG_REF_RE.search(line):
            return line
    return alt.strip()


def _infer_figure_ref(*parts: str, fallback: str) -> str:
    haystack = " ".join(p for p in parts if p)
    match = _FIG_REF_RE.search(haystack)
    if match:
        return f"Fig. {match.group(1)}"
    stem_match = re.search(r"(\d+[A-Za-z]?)", Path(fallback).stem)
    if stem_match:
        return f"Fig. {stem_match.group(1)}"
    return f"Fig. {Path(fallback).stem}"


def _resolve_image(uri: str, *, paper_dir: Path, figures_dir: Path | None) -> Path | None:
    if _is_remote(uri):
        return None
    raw = Path(uri)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(paper_dir / raw)
        if figures_dir is not None:
            candidates.append(figures_dir / raw)
            candidates.append(figures_dir / raw.name)
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
            return candidate.resolve()
    return None


def _copy_unique(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = src.name
    dst = dst_dir / base
    if dst.exists() and dst.resolve() != src.resolve():
        stem, suffix = src.stem, src.suffix
        i = 2
        while True:
            candidate = dst_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1
    if not dst.exists() or dst.resolve() != src.resolve():
        shutil.copy2(src, dst)
    return dst


def collect_markdown_figures(
    paper_md: str | Path,
    *,
    figures_dir: str | Path | None = None,
    copy_to: str | Path | None = None,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Return a normalized local figure inventory for one Markdown paper.

    ``figures_dir`` is searched as an additional root, useful for MinerU outputs
    where Markdown references may point at ``images/foo.png`` but the parsed
    bundle stores files in ``figures/``. ``copy_to`` creates a stable figures
    directory and makes ``image_file`` relative to it.
    """

    paper_md = Path(paper_md)
    text = paper_md.read_text(encoding="utf-8")
    paper_dir = paper_md.parent
    figures_root = Path(figures_dir) if figures_dir is not None else None
    copy_root = Path(copy_to) if copy_to is not None else None

    refs = _iter_image_refs(text)
    if strict and not refs:
        raise PaperAssetError(f"{paper_md} contains no Markdown/HTML image references")

    assets: list[FigureAsset] = []
    missing: list[str] = []
    for ref in refs:
        uri = str(ref["uri"])
        alt = str(ref["alt"])
        src = _resolve_image(uri, paper_dir=paper_dir, figures_dir=figures_root)
        if src is None:
            missing.append(uri)
            continue
        stored = _copy_unique(src, copy_root) if copy_root is not None else src
        window = _line_window(text, int(ref["pos"]))
        caption = _caption_from_window(window, alt)
        context = "\n".join(window).strip()
        figure_ref = _infer_figure_ref(alt, caption, uri, fallback=stored.name)
        image_file = stored.name if copy_root is not None else str(stored)
        assets.append(
            FigureAsset(
                figure_ref=figure_ref,
                image_file=image_file,
                source_path=str(src),
                alt_text=alt,
                caption=caption,
                context=context,
                markdown_path=uri,
            )
        )

    if strict and missing:
        raise PaperAssetError(
            f"{paper_md} references missing/non-local image files: {', '.join(missing)}"
        )
    if strict and not assets:
        raise PaperAssetError(f"{paper_md} has image references but none resolved to local files")
    return [asset.to_dict() for asset in assets]


def count_image_refs(paper_md: str | Path) -> int:
    """Number of Markdown/HTML image references in a paper (resolved or not)."""
    return len(_iter_image_refs(Path(paper_md).read_text(encoding="utf-8")))


def write_figure_index(figures: list[dict[str, Any]], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(figures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def figure_inventory_text(figures: list[dict[str, Any]]) -> str:
    """Compact, prompt-friendly figure inventory."""

    lines = []
    for f in figures:
        caption = (f.get("caption") or f.get("alt_text") or "").strip()
        lines.append(f"- {f.get('figure_ref')}: {f.get('image_file')} -- {caption}")
    return "\n".join(lines)
