"""Main control agent (host-side). Drives the end-to-end reproduction.

Steps (see docs/background.md):
  1. parse            pipeline.parse           PDF -> paper.md + figures/
  2. extract claims   pipeline.extract_claims  Claude -> claim text + anchors
     figure params    pipeline.extract_figure_params  Qwen-VL -> params/targets
     merge            pipeline.merge_claim_spec       -> canonical claim spec
  3. build task       pipeline.render_task + build-task skill -> sandbox task
                      pipeline.validate_task   consistency gate
  4. launch sandbox   sandbox.launcher         host sandbox, inject .env key
  5. run + trace      sandbox.runner           issue user_query, record trajectory
  6. retry            sandbox.retry            resume on interruption
  7. score            verify                   hidden reward.sh -> scalar reward

Everything here is light/host-side. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def reproduce(paper: Path, claim_id: str | None = None) -> None:
    raise NotImplementedError("scaffold: orchestrate steps 1-7")
