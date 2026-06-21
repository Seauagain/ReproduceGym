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

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from reprogym.config import load_dotenv

MINERU_BIN = "mineru-open-api"
MAX_BATCH = 200

Runner = Callable[[Sequence[str]], Any]


class ParseError(RuntimeError):
    pass


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True, check=False)


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
    load_dotenv()
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)

    if pdf_path.suffix.lower() in {".md", ".markdown"}:
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / "paper.md"
        shutil.copy2(pdf_path, dst)
        return dst

    runner = runner or _default_runner
    raw = out_dir / "_mineru_raw"
    raw.mkdir(parents=True, exist_ok=True)
    argv = build_mineru_argv(
        [pdf_path], raw, language=language, model=model, timeout=timeout, extra_args=extra_args
    )
    _check_runner_result(runner(argv))
    return _collect(raw, out_dir, pdf_path.stem)


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
    load_dotenv()
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
    return results
