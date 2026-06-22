"""R1: PDF -> Markdown via the mineru-open-api CLI (injectable runner)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from reproducegym.pipeline.parse import (
    ParseError,
    build_mineru_argv,
    parse_pdf,
    parse_pdfs_batch,
)


class FakeRunner:
    """Simulates mineru: writes <stem>.md + images/ into the -o dir per input."""

    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        out = Path(argv[argv.index("-o") + 1])
        inputs = argv[argv.index("extract") + 1 : argv.index("-o")]
        out.mkdir(parents=True, exist_ok=True)
        (out / "images").mkdir(exist_ok=True)
        (out / "images" / "fig1.jpg").write_bytes(b"jpeg")
        for inp in inputs:
            stem = Path(inp).stem
            (out / f"{stem}.md").write_text(f"# {stem}\n![](images/fig1.jpg)\n", encoding="utf-8")
        return SimpleNamespace(returncode=self.returncode, stdout="", stderr="")


def test_build_mineru_argv():
    argv = build_mineru_argv(["a.pdf", "b.pdf"], "/out", language="en", timeout=900)
    assert argv[:2] == ["mineru-open-api", "extract"]
    assert "a.pdf" in argv and "b.pdf" in argv
    assert argv[argv.index("-o") + 1] == "/out"
    assert argv[argv.index("--timeout") + 1] == "900"


def test_parse_pdf_maps_to_paper_md_and_figures(tmp_path):
    pdf = tmp_path / "mypaper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    runner = FakeRunner()
    out = parse_pdf(pdf, tmp_path / "dest", runner=runner)
    assert out == tmp_path / "dest" / "paper.md"
    assert out.read_text().startswith("# mypaper")
    assert (tmp_path / "dest" / "figures" / "fig1.jpg").is_file()
    assert len(runner.calls) == 1


def test_md_passthrough_no_runner(tmp_path):
    md = tmp_path / "paper_in.md"
    md.write_text("# already markdown", encoding="utf-8")

    def explode(argv):  # must not be called
        raise AssertionError("runner should not run for .md input")

    out = parse_pdf(md, tmp_path / "dest", runner=explode)
    assert out.read_text() == "# already markdown"


def test_parse_pdf_missing_md_raises(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"x")

    def noop(argv):
        Path(argv[argv.index("-o") + 1]).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(ParseError):
        parse_pdf(pdf, tmp_path / "dest", runner=noop)


def test_parse_pdf_runner_nonzero_raises(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"x")
    with pytest.raises(ParseError):
        parse_pdf(pdf, tmp_path / "dest", runner=FakeRunner(returncode=2))


def test_batch_chunks_over_200(tmp_path):
    pdfs = []
    for i in range(250):
        p = tmp_path / f"p{i}.pdf"
        p.write_bytes(b"x")
        pdfs.append(p)
    runner = FakeRunner()
    results = parse_pdfs_batch(pdfs, tmp_path / "dest", runner=runner)
    assert len(results) == 250
    assert len(runner.calls) == 2  # 200 + 50
    assert (tmp_path / "dest" / "p0" / "paper.md").is_file()
    assert (tmp_path / "dest" / "p249" / "paper.md").is_file()
