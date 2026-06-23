from __future__ import annotations

import pytest

from reproducegym.pipeline.paper_assets import (
    PaperAssetError,
    collect_markdown_figures,
    figure_inventory_text,
)


def test_markdown_without_images_fails(tmp_path):
    md = tmp_path / "paper.md"
    md.write_text("# Paper\nNo figures here.\n", encoding="utf-8")

    with pytest.raises(PaperAssetError):
        collect_markdown_figures(md, strict=True)


def test_markdown_image_refs_resolve_and_copy(tmp_path):
    (tmp_path / "fig5.png").write_bytes(b"img")
    md = tmp_path / "paper.md"
    md.write_text(
        "# Paper\n\n![Figure 5: policy iteration step](fig5.png)\n",
        encoding="utf-8",
    )

    figures = collect_markdown_figures(md, copy_to=tmp_path / "out_figures", strict=True)

    assert figures[0]["figure_ref"] == "Fig. 5"
    assert figures[0]["image_file"] == "fig5.png"
    assert (tmp_path / "out_figures" / "fig5.png").is_file()
    assert "Fig. 5" in figure_inventory_text(figures)


def test_missing_local_image_fails(tmp_path):
    md = tmp_path / "paper.md"
    md.write_text("![Fig. 1](missing.png)\n", encoding="utf-8")

    with pytest.raises(PaperAssetError):
        collect_markdown_figures(md, strict=True)
