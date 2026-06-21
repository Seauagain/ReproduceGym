"""Step 2b: read experimental params / curve targets off figures, via Qwen-VL.

Many RL papers report target numbers ONLY in figures. Runs once over the whole
paper's figures (prompts/extract_figure_params.md) and emits a reusable
sandboxes/<paper>/claims/figure_params.yaml of {value, source: 'Fig. N',
status} entries that claim specs reference. The FIGURE is public; the numbers
read off it are answer-key material and get exposure:hidden downstream. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def extract_figure_params(figures_dir: Path) -> dict:
    raise NotImplementedError("scaffold: Qwen-VL + prompts/extract_figure_params.md")
