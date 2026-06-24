"""R1: PDF -> Markdown via the mineru-open-api CLI (injectable runner)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import json

from reproducegym.pipeline.parse import (
    ParseError,
    arxiv_pdf_url,
    build_mineru_argv,
    download_pdf,
    parse_paper,
    parse_pdf,
    parse_pdfs_batch,
    resolve_pdf_source,
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
    assert not (tmp_path / "dest" / "_mineru_raw").exists()
    assert len(runner.calls) == 1


def test_md_passthrough_no_runner(tmp_path):
    md = tmp_path / "paper_in.md"
    (tmp_path / "fig1.png").write_bytes(b"img")
    md.write_text("# already markdown\n![Fig. 1](fig1.png)", encoding="utf-8")

    def explode(argv):  # must not be called
        raise AssertionError("runner should not run for .md input")

    out = parse_pdf(md, tmp_path / "dest", runner=explode)
    assert out.read_text() == "# already markdown\n![Fig. 1](fig1.png)"
    assert (tmp_path / "dest" / "figures" / "fig1.png").is_file()
    assert (tmp_path / "dest" / "figures.index.json").is_file()


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


# --- stage 0 (parse): source resolution + download + 00-parse/ bundle ---------
@pytest.mark.parametrize(
    "src,expected",
    [
        ("2503.20783", "https://arxiv.org/pdf/2503.20783"),
        ("arXiv:2503.20783v2", "https://arxiv.org/pdf/2503.20783v2"),
        ("https://arxiv.org/abs/2503.20783", "https://arxiv.org/pdf/2503.20783"),
        ("https://arxiv.org/pdf/2503.20783v3.pdf", "https://arxiv.org/pdf/2503.20783v3"),
    ],
)
def test_arxiv_pdf_url(src, expected):
    assert arxiv_pdf_url(src) == expected


def test_arxiv_pdf_url_none_for_non_arxiv():
    assert arxiv_pdf_url("https://example.com/p.pdf") is None
    assert arxiv_pdf_url("paper.pdf") is None


def test_resolve_pdf_source_kinds(tmp_path):
    assert resolve_pdf_source("2503.20783")[0] == "url"
    assert resolve_pdf_source("https://example.com/x.pdf") == ("url", "https://example.com/x.pdf")
    assert resolve_pdf_source("a/b/paper.pdf") == ("pdf", "a/b/paper.pdf")
    assert resolve_pdf_source("a/b/paper.md")[0] == "md"


def test_download_pdf_uses_injected_fetcher(tmp_path):
    dest = tmp_path / "d" / "source.pdf"

    def fetcher(url, d):
        Path(d).write_bytes(b"%PDF fake " + url.encode())

    out = download_pdf("https://example.com/x.pdf", dest, fetcher=fetcher)
    assert out.is_file() and out.read_bytes().startswith(b"%PDF")


def test_download_pdf_empty_raises(tmp_path):
    with pytest.raises(ParseError):
        download_pdf("u", tmp_path / "e.pdf", fetcher=lambda u, d: Path(d).write_bytes(b""))


def test_parse_paper_url_builds_bundle(tmp_path):
    runs = tmp_path / "runs"
    runner = FakeRunner()
    res = parse_paper(
        url="https://arxiv.org/abs/2503.20783",
        out=runs,
        runner=runner,
        fetcher=lambda u, d: Path(d).write_bytes(b"%PDF-1.4 fake"),
    )
    assert res["paper_id"] == "2503.20783"
    parse_dir = runs / "2503.20783" / "00-parse"
    assert (parse_dir / "paper.md").is_file()
    assert (parse_dir / "source.pdf").is_file()
    assert (parse_dir / "figures" / "fig1.jpg").is_file()
    assert not (parse_dir / "_mineru_raw").exists()
    idx = json.loads((parse_dir / "figures.index.json").read_text())
    assert idx and idx[0]["image_file"] == "fig1.jpg"
    meta = json.loads((parse_dir / "parse.json").read_text())
    assert meta["source"]["resolved"] == "https://arxiv.org/pdf/2503.20783"
    assert res["n_figures"] == 1
    assert Path(res["token_usage"]).is_file()
    summary = json.loads(Path(res["token_usage_summary"]).read_text())
    assert summary["totals"]["usage_records"] == 0
    assert summary["totals"]["usage_unavailable_records"] >= 2
    # canonical parsed paper lives only in 00-parse/
    assert not (runs / "2503.20783" / "paper.md").exists()


def test_parse_paper_requires_exactly_one_source(tmp_path):
    with pytest.raises(ParseError):
        parse_paper(out=tmp_path)
    with pytest.raises(ParseError):
        parse_paper(url="2503.1", pdf="x.pdf", out=tmp_path)
