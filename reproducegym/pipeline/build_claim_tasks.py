"""Build paper-derived claim/task bundles.

This is the single implementation of stage 1:

    paper -> figure index/evidence -> ranked claims -> spec -> task bundle

It never launches a sandbox or a GPU job.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from reproducegym.claim_spec import dump_claim_spec
from reproducegym.models import ClaudeClient, MultimodalFigureClient, multimodal_figure_configured
from reproducegym.pipeline.extract_claims import (
    dedup_claim_candidates,
    extract_claim_candidates,
    finalize_claims,
)
from reproducegym.pipeline.claim_selection import (
    DEFAULT_MAX_CLAIMS,
    rank_claims,
    selection_table,
    select_top_claims,
)
from reproducegym.pipeline.extract_figure_params import extract_claim_figure_evidence
from reproducegym.pipeline.merge_claim_spec import merge_claim_spec
from reproducegym.pipeline.paper_assets import (
    collect_markdown_figures,
    count_image_refs,
    write_figure_index,
)
from reproducegym.pipeline.render_check import write_baseline_check
from reproducegym.pipeline.render_task import render_task
from reproducegym.pipeline.validate_task import validate_task
from reproducegym.runlayout import PARSE, PaperLayout, write_index


def _resolve_paper_input(
    paper: str | Path, paper_id: str | None
) -> tuple[Path, Path | None, Path | None, str]:
    """Locate the markdown + its figures from a parse bundle or a raw paper file.

    Returns (paper_md, figures_dir, prebuilt_index, paper_id). A directory is read
    as a parse bundle (its 00-parse/ stage, or itself if it is one); a file is read
    as a raw paper.md whose figures live in a sibling figures/ dir.
    """
    paper = Path(paper)
    if paper.is_dir():
        pdir = paper / PARSE if (paper / PARSE).is_dir() else paper
        paper_md = pdir / "paper.md"
        if not paper_md.is_file():
            raise ValueError(
                f"no paper.md under {pdir}; run parse_paper.py to build the parse bundle first"
            )
        derived = paper.name if (paper / PARSE).is_dir() else paper.parent.name
        figures_dir = pdir / "figures"
        index = pdir / "figures.index.json"
        return paper_md, figures_dir if figures_dir.is_dir() else None, (
            index if index.is_file() else None
        ), (paper_id or derived)
    figures_dir = paper.parent / "figures"
    return paper, figures_dir if figures_dir.is_dir() else None, None, (paper_id or paper.stem)


def select_claims(claims: list[dict[str, Any]], wanted: list[str]) -> list[dict[str, Any]]:
    if not wanted:
        return claims
    out = []
    for cid in wanted:
        match = next(
            (
                c for c in claims
                if c.get("claim_id") == cid or c.get("source_claim_id") == cid
            ),
            None,
        )
        if match is None:
            raise ValueError(f"claim {cid!r} not found; have {[c['claim_id'] for c in claims]}")
        out.append(match)
    return out


def should_parse_images(mode: str, *, has_figures: bool, configured: bool) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        if not has_figures:
            raise ValueError("--parse-images=always requires local Markdown image references")
        if not configured:
            raise ValueError(
                "--parse-images=always requires a configured multimodal model "
                "(MULTIMODAL_* / VISION_* / QWEN_* keys in .env)"
            )
        return True
    if mode != "auto":
        raise ValueError(f"unknown parse_images mode: {mode!r}")
    return has_figures and configured


def build_claim_tasks(
    *,
    paper: str | Path,
    paper_id: str | None = None,
    out: str | Path,
    claim_ids: list[str] | None = None,
    parse_images: str = "auto",
    vl_min_confidence: float = 0.0,
    strict_vl: bool = True,
    baseline_check: bool = True,
    max_claims: int | None = DEFAULT_MAX_CLAIMS,
    refresh_claims: bool = False,
    claude_client: Any | None = None,
    multimodal_client: Any | None = None,
) -> dict[str, Any]:
    """Build hash-versioned task bundles from a paper."""

    paper_md, src_figures_dir, prebuilt_index, paper_id = _resolve_paper_input(paper, paper_id)
    layout = PaperLayout.for_paper(out, paper_id)
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.extract_dir.mkdir(parents=True, exist_ok=True)

    build_figures_dir = layout.extract_dir / "figures"
    if prebuilt_index is not None:
        # Reuse the parse-stage figure index directly. Image bytes live only in
        # 00-parse/figures; 01-extract stores metadata/evidence, not another copy.
        shutil.rmtree(build_figures_dir, ignore_errors=True)
        figures = json.loads(prebuilt_index.read_text(encoding="utf-8"))
        figures_dir_for_vl = src_figures_dir or build_figures_dir
    else:
        # Raw paper.md: resolve refs against a sibling figures/ dir (the core fix --
        # markdown that writes images/x.jpg now resolves to the parsed figures/).
        figures = collect_markdown_figures(
            paper_md,
            figures_dir=src_figures_dir,
            copy_to=build_figures_dir,
            strict=parse_images == "always",
        )
        figures_dir_for_vl = build_figures_dir
    write_figure_index(figures, layout.extract_dir / "figures.index.json")
    if not figures and parse_images == "auto" and count_image_refs(paper_md) > 0:
        print(
            f"[build] WARNING: {paper_md} has image references but none resolved to local "
            "files; image-enhanced parsing skipped. Run parse_paper.py to fetch figures.",
            flush=True,
        )

    claude = claude_client or ClaudeClient()
    raw_candidates_path = layout.extract_dir / "claim_candidates.raw.json"
    dedup_candidates_path = layout.extract_dir / "claim_candidates.dedup.json"
    if raw_candidates_path.is_file() and not refresh_claims:
        candidates = json.loads(raw_candidates_path.read_text(encoding="utf-8"))
    else:
        candidates = extract_claim_candidates(
            paper_md,
            client=claude,
            figures=figures,
            # Keep each text call below the full-paper prompt that times out on gpugeek,
            # but large enough to avoid turning one paper into dozens of model calls.
            max_chunk_chars=18_000,
        )
        raw_candidates_path.write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if dedup_candidates_path.is_file() and not refresh_claims:
        deduped_candidates = json.loads(dedup_candidates_path.read_text(encoding="utf-8"))
    else:
        deduped_candidates = dedup_claim_candidates(candidates, client=claude)
        dedup_candidates_path.write_text(
            json.dumps(deduped_candidates, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Candidates were already LLM-deduped before the expensive VL pass. Avoid a
    # second all-claims prompt here; candidate claims can be large and would
    # recreate the timeout/overflow mode this staged pipeline is meant to avoid.
    candidate_claims = rank_claims(finalize_claims(deduped_candidates, client=None))
    (layout.extract_dir / "candidate_claims.json").write_text(
        json.dumps(candidate_claims, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (layout.extract_dir / "claim_selection.json").write_text(
        json.dumps(selection_table(candidate_claims), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    selected_claims = select_claims(candidate_claims, claim_ids or [])
    if not claim_ids:
        selected_claims = select_top_claims(selected_claims, max_claims=max_claims)
    (layout.extract_dir / "selected_claims.json").write_text(
        json.dumps(selected_claims, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Compatibility: claims.json is the active queue, not the entire candidate pool.
    claims = selected_claims
    (layout.extract_dir / "claims.json").write_text(
        json.dumps(claims, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    configured = multimodal_client is not None or multimodal_figure_configured()
    do_parse_images = should_parse_images(
        parse_images,
        has_figures=bool(figures),
        configured=configured,
    )
    vl = (multimodal_client or MultimodalFigureClient()) if do_parse_images else None
    evidence_by_claim: dict[str, list[dict[str, Any]]] = {}
    aggregate_evidence: list[dict[str, Any]] = []
    claim_evidence_dir = layout.extract_dir / "claim_figure_evidence"
    shutil.rmtree(layout.extract_dir / "figure_vl_raw", ignore_errors=True)

    if do_parse_images:
        # Only the active queue needs expensive VL reads. The full candidate pool
        # remains available in candidate_claims.json for review or re-selection.
        for claim in selected_claims:
            claim_id = str(claim.get("claim_id") or "claim")
            evidence_path = claim_evidence_dir / f"{claim_id}.yaml"
            if evidence_path.is_file():
                claim_evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8")) or []
            else:
                claim_evidence = extract_claim_figure_evidence(
                    claim,
                    figures_dir_for_vl,
                    client=vl,
                    figures_index=figures,
                    min_confidence=vl_min_confidence,
                    strict=strict_vl,
                    out_path=evidence_path,
                )
            evidence_by_claim[claim_id] = claim_evidence
            aggregate_evidence.extend(claim_evidence)
        (layout.extract_dir / "figure_evidence.yaml").write_text(
            yaml.safe_dump(aggregate_evidence, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:
        reason = "disabled" if parse_images == "never" else (
            "no local figures" if not figures else "multimodal model not configured"
        )
        (layout.extract_dir / "figure_evidence.yaml").write_text(
            f"[]\n# image-enhanced parsing skipped: {reason}\n",
            encoding="utf-8",
        )
    (layout.extract_dir / "claim_figure_evidence.index.json").write_text(
        json.dumps(evidence_by_claim, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    built = []
    for claim in selected_claims:
        spec = merge_claim_spec(claim, paper_id=paper_id, figure_evidence=aggregate_evidence)
        spec_path = layout.spec_path(spec["claim_id"], spec["spec_hash"])
        dump_claim_spec(spec, spec_path)
        task_dir = layout.task_dir(spec["claim_id"], spec["spec_hash"])
        render_task(spec, task_dir)
        if baseline_check:
            write_baseline_check(spec, task_dir / "reward")
        problems = validate_task(task_dir, spec)
        if problems:
            raise ValueError("task failed validation: " + "; ".join(problems))
        built.append(
            {
                "claim_id": spec["claim_id"],
                "spec_hash": spec["spec_hash"],
                "spec_path": str(spec_path),
                "task_dir": str(task_dir),
            }
        )

    manifest = write_index(layout, paper_id=paper_id)
    return {
        "paper_id": paper_id,
        "built": built,
        "manifest": str(layout.manifest_path),
        "n_claims": len(manifest["claims"]),
        "n_candidate_claims": len(candidate_claims),
        "n_selected_claims": len(selected_claims),
        "max_claims": max_claims if not claim_ids else None,
        "selection": str(layout.extract_dir / "claim_selection.json"),
        "parse_images": parse_images,
        "image_evidence": bool(aggregate_evidence),
    }
