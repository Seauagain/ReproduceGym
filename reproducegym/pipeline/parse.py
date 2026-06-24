"""Step 1: PDF -> Markdown + figures, via MinerU.

Implemented over the `mineru-open-api` CLI (pip install mineru-open-api),
authenticated by MINERU_TOKEN/MINERU_API_KEY from .env. NOT a raw REST base_url.

    mineru-open-api extract <pdf...> -o <out> --language en --model pipeline --timeout 900

Batch is preferred for dataset building (<=200 files/request, server-side
concurrent); >200 inputs are split into chunks here rather than looping singles.

MinerU emits a flat `<basename>.md` per input plus a shared `images/` folder; we
map that to `<dest>/paper.md` + `<dest>/figures/`. A `.md` input is passed through
unchanged (already markdown). The CLI invocation is injectable so the mapping
logic is unit-tested without the network.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from reproducegym.config import dotenv_values
from reproducegym.pipeline.paper_assets import collect_markdown_figures, write_figure_index
from reproducegym.pipeline.token_usage import TokenUsageRecorder
from reproducegym.runlayout import PaperLayout, write_index

MINERU_BIN = "mineru-open-api"
MAX_BATCH = 200

Runner = Callable[[Sequence[str]], Any]
Fetcher = Callable[[str, Path], Any]

# arXiv: bare id (2503.20783[v2]), optional arXiv: prefix, or abs/pdf URL.
_ARXIV_ID = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$", re.IGNORECASE)
_ARXIV_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?(?:[?#].*)?$", re.IGNORECASE
)


class ParseError(RuntimeError):
    pass


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess:
    file_env = dotenv_values()
    env = dict(os.environ)
    for key in ("MINERU_TOKEN", "MINERU_API_KEY"):
        if file_env.get(key):
            env[key] = file_env[key]
        else:
            env.pop(key, None)
    return subprocess.run(list(argv), capture_output=True, text=True, check=False, env=env)


def build_mineru_argv(
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    *,
    language: str = "en",
    model: str = "pipeline",
    timeout: int = 900,
    extra_args: Sequence[str] = (),
) -> list[str]:
    argv = [MINERU_BIN, "extract", *[str(p) for p in inputs], "-o", str(out_dir)]
    argv += ["--language", language, "--model", model, "--timeout", str(timeout)]
    argv += list(extra_args)
    return argv


def _check_runner_result(res: Any) -> None:
    code = getattr(res, "returncode", 0)
    if code:
        tail = (getattr(res, "stderr", "") or getattr(res, "stdout", "") or "")[-800:]
        raise ParseError(f"mineru-open-api exited {code}: {tail}")


def _find_md(raw_dir: Path, stem: str) -> Path:
    exact = list(raw_dir.rglob(f"{stem}.md"))
    if exact:
        return exact[0]
    any_md = list(raw_dir.rglob("*.md"))
    if not any_md:
        raise ParseError(f"mineru produced no .md under {raw_dir}")
    return any_md[0]


def _collect(raw_dir: Path, dest_dir: Path, stem: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paper_md = dest_dir / "paper.md"
    shutil.copy2(_find_md(raw_dir, stem), paper_md)

    images = next((d for d in raw_dir.rglob("images") if d.is_dir()), None)
    if images is not None:
        figures = dest_dir / "figures"
        if figures.exists():
            shutil.rmtree(figures)
        shutil.copytree(images, figures)
        figures_index = collect_markdown_figures(
            paper_md,
            figures_dir=figures,
            copy_to=figures,
            strict=True,
        )
        write_figure_index(figures_index, dest_dir / "figures.index.json")
    return paper_md


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def parse_pdf(
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    runner: Runner | None = None,
    language: str = "en",
    model: str = "pipeline",
    timeout: int = 900,
    extra_args: Sequence[str] = (),
) -> Path:
    """Parse one PDF (or pass through a .md) into <out_dir>/paper.md (+ figures/)."""
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)

    if pdf_path.suffix.lower() in {".md", ".markdown"}:
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / "paper.md"
        shutil.copy2(pdf_path, dst)
        figures = collect_markdown_figures(
            pdf_path,
            copy_to=out_dir / "figures",
            strict=True,
        )
        write_figure_index(figures, out_dir / "figures.index.json")
        return dst

    runner = runner or _default_runner
    raw = out_dir / "_mineru_raw"
    raw.mkdir(parents=True, exist_ok=True)
    argv = build_mineru_argv(
        [pdf_path], raw, language=language, model=model, timeout=timeout, extra_args=extra_args
    )
    _check_runner_result(runner(argv))
    paper_md = _collect(raw, out_dir, pdf_path.stem)
    shutil.rmtree(raw, ignore_errors=True)
    return paper_md


def parse_pdfs_batch(
    pdf_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    runner: Runner | None = None,
    language: str = "en",
    model: str = "pipeline",
    timeout: int = 1800,
    extra_args: Sequence[str] = (),
) -> list[Path]:
    """Parse many PDFs in <=200-file batches -> <out_dir>/<stem>/paper.md each."""
    runner = runner or _default_runner
    out_dir = Path(out_dir)
    pdfs = [Path(p) for p in pdf_paths]

    results: list[Path] = []
    for chunk in _chunks(pdfs, MAX_BATCH):
        raw = out_dir / "_mineru_raw"
        raw.mkdir(parents=True, exist_ok=True)
        argv = build_mineru_argv(
            chunk, raw, language=language, model=model, timeout=timeout, extra_args=extra_args
        )
        _check_runner_result(runner(argv))
        for p in chunk:
            results.append(_collect(raw, out_dir / p.stem, p.stem))
        shutil.rmtree(raw, ignore_errors=True)
    return results


# --------------------------------------------------------------------------- #
# Stage 0 (parse): source resolution + download + canonical 00-parse/ bundle.
# --------------------------------------------------------------------------- #
def arxiv_pdf_url(src: str) -> str | None:
    """Map an arXiv id / abs / pdf reference to its canonical pdf URL, else None."""
    s = src.strip()
    m = _ARXIV_URL.search(s)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    m = _ARXIV_ID.match(s)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    return None


def resolve_pdf_source(src: str | Path) -> tuple[str, str]:
    """Classify a source into ("url"|"pdf"|"md", resolved). arXiv-aware."""
    s = str(src).strip()
    ax = arxiv_pdf_url(s)
    if ax:
        return ("url", ax)
    if urlparse(s).scheme in {"http", "https"}:
        return ("url", s)
    suffix = Path(s).suffix.lower()
    if suffix in {".md", ".markdown"}:
        return ("md", s)
    return ("pdf", s)


def _default_fetcher(url: str, dest: Path) -> None:
    import urllib.request  # lazy: keep module import network-free

    req = urllib.request.Request(url, headers={"User-Agent": "reproducegym-parse/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def download_pdf(url: str, dest: str | Path, *, fetcher: Fetcher | None = None) -> Path:
    """Download a PDF to dest (injectable fetcher for offline tests)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    (fetcher or _default_fetcher)(url, dest)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise ParseError(f"download produced no file for {url}")
    return dest


def _derive_paper_id(kind: str, resolved: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if kind == "url":
        m = re.search(r"/(?:pdf|abs)/([^/?#]+?)(?:\.pdf)?$", resolved)
        if m:
            return m.group(1).replace("/", "-")
        stem = Path(urlparse(resolved).path).stem
        return stem or "paper"
    return Path(resolved).stem


def _normalize_source(
    src: str | Path | None,
    url: str | Path | None,
    pdf: str | Path | None,
    md: str | Path | None,
) -> tuple[str, str]:
    provided = [x for x in (src, url, pdf, md) if x]
    if len(provided) != 1:
        raise ParseError("parse_paper requires exactly one of src/url/pdf/md")
    if md:
        return ("md", str(md))
    if pdf:
        return resolve_pdf_source(pdf)
    if url:
        return resolve_pdf_source(url)
    return resolve_pdf_source(src)  # type: ignore[arg-type]


def parse_paper(
    *,
    src: str | Path | None = None,
    url: str | Path | None = None,
    pdf: str | Path | None = None,
    md: str | Path | None = None,
    out: str | Path,
    paper_id: str | None = None,
    runner: Runner | None = None,
    fetcher: Fetcher | None = None,
    language: str = "en",
    model: str = "pipeline",
    timeout: int = 900,
    extra_args: Sequence[str] = (),
) -> dict[str, Any]:
    """Stage 0: url/pdf/md -> runs/<paper_id>/00-parse/{paper.md,figures/,figures.index.json}.

    Exactly one of src/url/pdf/md is required. URLs are arXiv-aware; PDFs go through
    MinerU; .md inputs pass through (figures collected from their sibling images).
    """
    kind, resolved = _normalize_source(src, url, pdf, md)
    paper_id = _derive_paper_id(kind, resolved, paper_id)

    layout = PaperLayout.for_paper(out, paper_id)
    dest = layout.parse_dir
    dest.mkdir(parents=True, exist_ok=True)
    token_recorder = TokenUsageRecorder(layout.root, paper_id=paper_id)
    source_meta = {
        "kind": kind,
        "resolved": resolved,
        "input": str(src or url or pdf or md),
    }

    if kind == "url":
        start = time.perf_counter()
        pdf_path = download_pdf(resolved, dest / "source.pdf", fetcher=fetcher)
        token_recorder.record_event(
            stage="parse",
            step="download_pdf",
            elapsed_ms=int((time.perf_counter() - start) * 1000),
            metadata={
                "source": source_meta,
                "bytes": pdf_path.stat().st_size if pdf_path.is_file() else None,
            },
        )
        start = time.perf_counter()
        paper_md = parse_pdf(
            pdf_path, dest, runner=runner, language=language, model=model,
            timeout=timeout, extra_args=extra_args,
        )
        token_recorder.record_event(
            stage="parse",
            step="mineru_extract",
            elapsed_ms=int((time.perf_counter() - start) * 1000),
            metadata={"source": source_meta, "mineru_model": model, "timeout": timeout},
        )
    elif kind == "pdf":
        start = time.perf_counter()
        paper_md = parse_pdf(
            resolved, dest, runner=runner, language=language, model=model,
            timeout=timeout, extra_args=extra_args,
        )
        token_recorder.record_event(
            stage="parse",
            step="mineru_extract",
            elapsed_ms=int((time.perf_counter() - start) * 1000),
            metadata={"source": source_meta, "mineru_model": model, "timeout": timeout},
        )
    else:  # md passthrough (figures collected from sibling images/)
        start = time.perf_counter()
        paper_md = parse_pdf(resolved, dest, runner=runner)
        token_recorder.record_event(
            stage="parse",
            step="md_passthrough",
            elapsed_ms=int((time.perf_counter() - start) * 1000),
            metadata={"source": source_meta},
        )

    figures: list[dict[str, Any]] = []
    if layout.figure_index_path.is_file():
        figures = json.loads(layout.figure_index_path.read_text(encoding="utf-8"))

    (dest / "parse.json").write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "source": source_meta,
                "mineru_model": model if kind in {"url", "pdf"} else None,
                "n_figures": len(figures),
                "paper_md": str(paper_md),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    token_recorder.record_event(
        stage="parse",
        step="bundle_index",
        metadata={"n_figures": len(figures), "paper_chars": paper_md.stat().st_size},
    )
    write_index(layout, paper_id=paper_id)
    token_summary = token_recorder.write_summary()
    return {
        "paper_id": paper_id,
        "paper_md": str(paper_md),
        "parse_dir": str(dest),
        "figures": figures,
        "n_figures": len(figures),
        "source": source_meta,
        "token_usage": str(token_recorder.jsonl_path),
        "token_usage_summary": str(token_summary),
    }
