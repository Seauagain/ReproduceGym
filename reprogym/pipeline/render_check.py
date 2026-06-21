"""Generate a reward/check.py that scores ONLY by recomputing metrics.

The generated verifier reads the agent's raw artifact (output/metrics.csv),
recomputes each claim metric via its formula, compares the recomputed values to
the hidden thresholds, and derives the verdict itself. It never reads an
agent-declared verdict, score, or reward -- self-reported judgement is ignored by
construction.

The recompute logic is the single engine in reprogym/verifier/engine.py, embedded
verbatim (between its ENGINE BEGIN/END markers) so the rendered check.py is stdlib
-only and self-contained at scoring time. The generated file also declares the
module-level CONTRACT dict literal that validate_task reads, plus a SPEC dict the
engine consumes. The build-task skill may overwrite check.py with a bespoke
recompute verifier for claims whose metrics need richer extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import reprogym.verifier.engine as _engine
from reprogym.pipeline.render_task import derive_contract

DEFAULT_REWARD_BY_VERDICT = {
    "reproduced": 0.8,
    "failed": 0.35,
    "inconclusive": 0.2,
    "invalid": 0.0,
}

_ENGINE_BEGIN = "# === REPROGYM ENGINE BEGIN ==="
_ENGINE_END = "# === REPROGYM ENGINE END ==="

_HEADER = (
    "#!/usr/bin/env python3\n"
    '"""ReproGym recompute verifier (auto-generated).\n'
    "\n"
    "Scores ONLY by recomputing metrics from the reproduction artifacts; it never\n"
    "reads any agent-declared verdict, score, or reward.\n"
    '"""\n'
    "from __future__ import annotations\n\n"
    "import argparse\n"
    "import ast\n"
    "import csv\n"
    "import json\n"
    "import math\n"
    "import statistics\n"
    "from pathlib import Path\n\n"
)

_MAIN = '''

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--reward-only", action="store_true")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    report = recompute(workspace, SPEC)
    try:
        (workspace / "output").mkdir(exist_ok=True)
        (workspace / "output" / "verification_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass
    print(report["reward"] if args.reward_only else json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _engine_source() -> str:
    src = Path(_engine.__file__).read_text(encoding="utf-8")
    i = src.index(_ENGINE_BEGIN)
    j = src.index(_ENGINE_END) + len(_ENGINE_END)
    return src[i:j]


def _reward_by_verdict(spec: dict[str, Any]) -> dict[str, float]:
    reward_by_verdict = dict(DEFAULT_REWARD_BY_VERDICT)
    base = (spec.get("reward") or {}).get("base_by_verdict")
    if isinstance(base, dict):
        reward_by_verdict.update(base)
    return reward_by_verdict


def _metrics_csv_name(required_files: list[str]) -> str:
    for f in required_files:
        if f.endswith("metrics.csv"):
            return f
    return "output/metrics.csv"


def render_check_py(spec: dict[str, Any]) -> str:
    c = derive_contract(spec)
    contract = {
        "claim_id": c["claim_id"],
        "metrics": c["metric_names"],
        "thresholds": c["thresholds"],
        "required_files": c["required_files"],
        "verdicts": c["verdicts"],
    }
    spec_dict = {
        "claim_id": c["claim_id"],
        "metrics": c["metric_names"],
        "formulas": c["metric_formula"],
        "directions": c["metric_direction"],
        "windows": c["metric_window"],
        "thresholds": c["thresholds"],
        "required_files": c["required_files"],
        "metrics_csv": _metrics_csv_name(c["required_files"]),
        "metrics_csv_columns": c["metrics_csv_columns"],
        "condition_col": "condition",
        "conditions": c["conditions"],
        "min_rows_per_condition": c["min_rows_per_condition"],
        "verdicts": c["verdicts"],
        "reward_by_verdict": _reward_by_verdict(spec),
    }
    body = _HEADER
    body += "CONTRACT = %r\n" % (contract,)
    body += "SPEC = %r\n\n\n" % (spec_dict,)
    body += _engine_source()
    body += _MAIN
    return body


def write_check(spec: dict[str, Any], reward_dir: str | Path) -> Path:
    reward_dir = Path(reward_dir)
    reward_dir.mkdir(parents=True, exist_ok=True)
    out = reward_dir / "check.py"
    out.write_text(render_check_py(spec), encoding="utf-8")
    return out


# Backwards-compatible aliases (the auto-generated verifier is now recompute-based,
# not a verdict-trust baseline).
render_baseline_check_py = render_check_py
write_baseline_check = write_check
