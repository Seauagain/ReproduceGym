"""Step 3a: deterministically render a claim spec into a ClawGym-pure sandbox task.

claim spec -> tasks/<claim_id>/{data_entry.json, input_files/(task.md, params.yaml,
protocol.yaml, expected.json [+ copied assets]), reward/(reward.sh, targets.yaml)}.

ClawGym-pure: there is NO private/ dir. Verifier-only data (hidden thresholds /
params) lives under reward/ because the rollout copies only reward/ at scoring.
Exposure routing (per-leaf `exposure`, default from the schema) decides whether a
leaf lands in input_files/ (visible) or reward/ (hidden); a hidden value is never
written into any input_files/ artifact.

reward/check.py is intentionally NOT written here -- it is authored by the
build-task skill. The renderer only emits the constants check.py must agree with,
and validate_task is the gate that proves they stayed consistent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from reprogym.claim_spec import load_claim_spec, validate_claim_spec

# ClawGym-compatible mount path (the rollout reads this from data_entry.json).
INPUT_MOUNT_DIR = "/root/.openclaw/workspace"
STANDARD_VERDICTS = ["reproduced", "failed", "inconclusive", "invalid"]
BASE_VISIBLE_FILES = ["task.md", "params.yaml", "protocol.yaml", "expected.json"]


# --------------------------------------------------------------------------- #
# Canonical contract: the single set of facts every artifact must agree on.
# Shared with validate_task so render and validation read the same source.
# --------------------------------------------------------------------------- #
def _threshold_is_visible(threshold: dict) -> bool:
    return threshold.get("exposure", "hidden") == "visible"


def _param_is_visible(param: dict) -> bool:
    return param.get("exposure", "visible") == "visible"


def derive_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """The facts task.md / params / protocol / expected / targets / check.py share."""
    metrics = spec["metrics"]
    thresholds = spec.get("thresholds", [])
    required = spec.get("required_outputs", {})
    return {
        "claim_id": spec["claim_id"],
        "paper_id": spec["paper_id"],
        "metric_names": [m["name"] for m in metrics],
        "metric_direction": {m["name"]: m["direction"] for m in metrics},
        "metric_formula": {m["name"]: m.get("formula", "") for m in metrics},
        "metric_window": {m["name"]: m.get("window") for m in metrics},
        "thresholds": {t["metric"]: t["pass_threshold"] for t in thresholds},
        "visible_thresholds": {
            t["metric"]: t["pass_threshold"] for t in thresholds if _threshold_is_visible(t)
        },
        "hidden_thresholds": {
            t["metric"]: t["pass_threshold"] for t in thresholds if not _threshold_is_visible(t)
        },
        "threshold_rationale": {t["metric"]: t.get("rationale", "") for t in thresholds},
        "required_files": list(required.get("files", [])),
        "metrics_csv_columns": list(required.get("metrics_csv_columns", [])),
        "min_rows_per_condition": required.get("min_rows_per_condition"),
        "verdicts": list(STANDARD_VERDICTS),
        "verdict_rules": spec.get("verdict_rules", {}),
    }


def task_id_for(spec: dict[str, Any]) -> str:
    raw = f"{spec['paper_id']}-{spec['claim_id']}".lower()
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", raw)).strip("-")


def visible_files(spec: dict[str, Any]) -> list[str]:
    files = list(BASE_VISIBLE_FILES)
    for asset in spec.get("input_files", []):
        name = Path(asset).name
        if name not in files:
            files.append(name)
    return files


# --------------------------------------------------------------------------- #
# Individual artifact renderers (pure: spec -> str/obj)
# --------------------------------------------------------------------------- #
def render_data_entry(spec: dict[str, Any]) -> dict[str, Any]:
    c = derive_contract(spec)
    return {
        "task_id": task_id_for(spec),
        "user_query": (
            f"Reproduce claim '{spec['claim_id']}' from paper {spec['paper_id']}. "
            "Read input_files/task.md and the referenced parameter files, run the "
            "required experiment, and write all required outputs under output/."
        ),
        "metadata": {
            "paper_id": spec["paper_id"],
            "claim_id": spec["claim_id"],
            "claim_type": spec.get("claim_type"),
            "tier": spec["tier"],
            "exposure": spec["exposure_policy"],
            "requires_training": spec.get("requires_training"),
            "cost": spec.get("cost"),
            "verifiability": spec.get("verifiability"),
            "grading_type": "automated",
            "public_inputs": visible_files(spec),
            "private_targets_hidden": bool(c["hidden_thresholds"]),
        },
        "input_mount_dir": INPUT_MOUNT_DIR,
    }


def render_task_md(spec: dict[str, Any]) -> str:
    c = derive_contract(spec)
    lines: list[str] = []
    add = lines.append

    add(f"# Task: Reproduce claim `{spec['claim_id']}`\n")
    add(
        "You are an autonomous reproduction agent. Read this specification and the "
        "referenced parameter files, consult the paper when needed, then write whatever "
        "code is necessary to run the experiment and submit auditable results under "
        "`output/`.\n"
    )

    add("## 1. Claim To Reproduce\n")
    add(f"Paper: `{spec['paper_id']}`\n")
    add(f"> {spec['statement']}\n")

    add("## 2. Paper Evidence\n")
    anchors = spec.get("anchors", [])
    if anchors:
        for a in anchors:
            note = f" — {a['note']}" if a.get("note") else ""
            add(f"- {a['kind'].capitalize()} {a['ref']}{note}")
        add("")
    else:
        add("No structured anchors were provided.\n")

    add("## 3. Experimental Contract\n")
    conditions = spec.get("conditions", [])
    if conditions:
        add("Run the following conditions and label outputs exactly as shown:\n")
        for cond in conditions:
            add(f"### Condition `{cond['label']}`")
            add(cond["description"])
            for k, v in (cond.get("switches") or {}).items():
                add(f"- {k}: `{v}`")
            add("")
    if spec.get("matched_variables"):
        add("Hold these variables identical across all conditions:\n")
        for var in spec["matched_variables"]:
            add(f"- {var}")
        add("")

    add("## 4. Parameter Contract\n")
    add(
        "Read `params.yaml` before writing code. `paper_specified` values must not be "
        "changed silently; `author_repo_config` must record source/commit; "
        "`paper_unspecified` values may be substituted only if recorded as deviations "
        "(which makes the run approximate, not strict).\n"
    )

    add("## 5. Required Output Files\n")
    add("Create exactly these files under `output/`:\n")
    add("```text")
    for f in c["required_files"]:
        add(f)
    add("```\n")

    if c["metrics_csv_columns"]:
        add("## 6. Metrics CSV Contract\n")
        add("`output/metrics.csv` must contain at least these columns:\n")
        add("```text")
        add(",".join(c["metrics_csv_columns"]))
        add("```")
        if c["min_rows_per_condition"]:
            add(f"\nAt least {c['min_rows_per_condition']} rows per condition.")
        add("")

    add("## 7. Verifier Contract\n")
    add("The hidden verifier recomputes these metrics from your outputs:\n")
    for name in c["metric_names"]:
        formula = c["metric_formula"].get(name) or "(see protocol.yaml)"
        window = c["metric_window"].get(name)
        win = f" over {window}" if window else ""
        add(f"- `{name}` = {formula}{win} ({c['metric_direction'][name]})")
    add("")
    if c["visible_thresholds"]:
        add("Pass criteria:\n")
        for metric, val in c["visible_thresholds"].items():
            op = ">=" if c["metric_direction"][metric] == "higher_is_better" else "<="
            add(f"- `{metric} {op} {val}`")
        add("")
    if c["hidden_thresholds"]:
        add(
            "Some pass thresholds are withheld and applied only by the hidden verifier. "
            "Report honestly computed metrics; do not guess the thresholds.\n"
        )

    add("## 8. Verdicts\n")
    add("The verifier emits one of: " + ", ".join(f"`{v}`" for v in c["verdicts"]) + ".\n")
    add(
        "Do not copy paper claims or target values into your outputs. Reported metrics "
        "must be recomputable from your submitted artifacts.\n"
    )
    return "\n".join(lines).rstrip() + "\n"


def render_params_yaml(spec: dict[str, Any]) -> str:
    doc: dict[str, Any] = {
        "claim_id": spec["claim_id"],
        "parameter_policy": {
            "paper_specified": "Must not be changed silently.",
            "author_repo_config": "Use if recovered from released code/config; record source.",
            "paper_unspecified": "Recover before strict run; local substitute makes run approximate.",
        },
    }
    buckets = {"paper_specified": {}, "author_repo_config": {}, "paper_unspecified": {}}
    for p in spec.get("params", []):
        if p.get("applies_to_claim") is False or not _param_is_visible(p):
            continue
        entry: dict[str, Any] = {}
        if "value" in p and p["value"] is not None:
            entry["value"] = p["value"]
        if p.get("unit"):
            entry["unit"] = p["unit"]
        if p.get("source"):
            entry["source"] = p["source"]
        if p["status"] == "paper_unspecified":
            entry["required_for_strict"] = bool(
                p.get("required_for_strict", p.get("affects_strict_reproduction", False))
            )
        buckets[p["status"]][p["name"]] = entry
    for status, entries in buckets.items():
        if entries:
            doc[status] = entries

    switches = {
        cond["label"]: cond["switches"]
        for cond in spec.get("conditions", [])
        if cond.get("switches")
    }
    if switches:
        doc["algorithm_switches"] = switches
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def render_protocol_yaml(spec: dict[str, Any]) -> str:
    c = derive_contract(spec)
    doc: dict[str, Any] = {
        "protocol_version": 0.1,
        "task_id": task_id_for(spec),
        "claim_id": spec["claim_id"],
        "workspace_contract": {
            "exposure": spec["exposure_policy"],
            "agent_visible": visible_files(spec),
            "agent_hidden": ["reward/"],
            "agent_must_write": c["required_files"],
        },
        "episode": {
            "reset": ["Mount input_files/ into a clean workspace."],
            "interaction": [
                "Agent may inspect files, create scripts, run commands, and collect logs."
            ],
            "submit": ["Agent writes the required output files."],
            "verify": [
                "reward/check.py recomputes metrics from agent outputs.",
                "reward/reward.sh returns a scalar reward in [0, 1].",
            ],
        },
        "metric_computation": {
            name: {
                "formula": c["metric_formula"].get(name, ""),
                "direction": c["metric_direction"][name],
                **({"window": c["metric_window"][name]} if c["metric_window"].get(name) else {}),
            }
            for name in c["metric_names"]
        },
        "verdict_rules": c["verdict_rules"] or {v: [] for v in c["verdicts"]},
    }
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def render_expected_json(spec: dict[str, Any]) -> dict[str, Any]:
    """Agent-visible expectation. Hidden thresholds are NOT included here."""
    c = derive_contract(spec)
    primary = []
    for name in c["metric_names"]:
        entry = {"name": name, "direction": c["metric_direction"][name]}
        if name in c["visible_thresholds"]:
            entry["pass_threshold"] = c["visible_thresholds"][name]
        primary.append(entry)
    return {
        "claim_id": spec["claim_id"],
        "primary_metrics": primary,
        "allowed_verdicts": c["verdicts"],
        "thresholds_hidden": bool(c["hidden_thresholds"]),
    }


def render_reward_targets_yaml(spec: dict[str, Any]) -> str:
    """Hidden verifier targets (lives in reward/; never mounted for the agent)."""
    c = derive_contract(spec)
    doc: dict[str, Any] = {
        "claim_id": spec["claim_id"],
        "hidden_from_agent": True,
        "primary_thresholds": {
            metric: {
                "pass_threshold": value,
                "direction": c["metric_direction"].get(metric, ""),
                **(
                    {"rationale": c["threshold_rationale"][metric]}
                    if c["threshold_rationale"].get(metric)
                    else {}
                ),
            }
            for metric, value in c["thresholds"].items()
        },
        "verdicts": c["verdicts"],
    }
    hidden_params = {
        p["name"]: {k: p[k] for k in ("value", "unit", "source") if p.get(k) is not None}
        for p in spec.get("params", [])
        if not _param_is_visible(p)
    }
    if hidden_params:
        doc["hidden_params"] = hidden_params
    if spec.get("reward"):
        doc["reward"] = spec["reward"]
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


REWARD_SH = """#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/check.py" "${WORKSPACE}" --reward-only
"""


# --------------------------------------------------------------------------- #
# Top-level: render everything to disk.
# --------------------------------------------------------------------------- #
def render_task(
    claim_spec: str | Path | dict[str, Any],
    task_dir: str | Path,
    *,
    assets_dir: str | Path | None = None,
) -> Path:
    """Render a claim spec into a ClawGym-pure task directory. Returns task_dir."""
    spec = (
        load_claim_spec(claim_spec)
        if isinstance(claim_spec, (str, Path))
        else claim_spec
    )
    validate_claim_spec(spec)

    task_dir = Path(task_dir)
    input_dir = task_dir / "input_files"
    reward_dir = task_dir / "reward"
    input_dir.mkdir(parents=True, exist_ok=True)
    reward_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "data_entry.json").write_text(
        json.dumps(render_data_entry(spec), indent=2) + "\n", encoding="utf-8"
    )
    (input_dir / "task.md").write_text(render_task_md(spec), encoding="utf-8")
    (input_dir / "params.yaml").write_text(render_params_yaml(spec), encoding="utf-8")
    (input_dir / "protocol.yaml").write_text(render_protocol_yaml(spec), encoding="utf-8")
    (input_dir / "expected.json").write_text(
        json.dumps(render_expected_json(spec), indent=2) + "\n", encoding="utf-8"
    )

    reward_sh = reward_dir / "reward.sh"
    reward_sh.write_text(REWARD_SH, encoding="utf-8")
    reward_sh.chmod(0o755)
    (reward_dir / "targets.yaml").write_text(render_reward_targets_yaml(spec), encoding="utf-8")

    # Copy extra visible assets if their source dir was provided.
    if assets_dir is not None:
        assets_dir = Path(assets_dir)
        for asset in spec.get("input_files", []):
            src = assets_dir / asset
            if src.is_file():
                dst = input_dir / Path(asset).name
                dst.write_bytes(src.read_bytes())

    return task_dir
