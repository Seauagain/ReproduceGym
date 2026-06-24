"""Step 3b: consistency gate over a rendered task.

Re-derives the canonical contract from the claim spec and rejects the task unless
metric names, threshold values + bindings, required output files / metrics.csv
columns, and the verdict label set AGREE across data_entry.json, task.md,
params.yaml, protocol.yaml, expected.json, reward/targets.yaml and the
hand-authored reward/check.py. Also enforces the ClawGym structural contract and
the exposure rule (no hidden value may leak into input_files/).

Returns a list of human-readable problems; an empty list means the task is sound.
reward/check.py must declare a module-level CONTRACT dict literal so this gate can
read it without importing/executing arbitrary code:

    CONTRACT = {
        "claim_id": "...",
        "metrics": ["m1", "m2"],
        "thresholds": {"m1": 1.5, "m2": -0.02},
        "required_files": ["output/result.json", ...],
        "verdicts": ["reproduced", "failed", "inconclusive", "invalid"],
    }
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import yaml

from reproducegym.claim_spec import load_claim_spec
from reproducegym.pipeline.formula_contract import formula_problem
from reproducegym.pipeline.render_task import derive_contract, visible_files


def _num_strings(value: Any) -> list[str]:
    out = {str(value), repr(value)}
    return list(out)


def _has_number_literal(blob: str, text: str) -> bool:
    """Match a hidden numeric value as a standalone literal, not inside hashes/paths."""
    pattern = re.compile(r"(?<![A-Za-z0-9_.+-])" + re.escape(text) + r"(?![A-Za-z0-9_.+-])")
    return bool(pattern.search(blob))


def _visible_text_for_leak_scan(text: str) -> str:
    """Remove renderer metadata constants that are not paper/verifier answers."""

    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*(protocol_version|schema_version)\s*:", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_check_contract(text: str) -> dict[str, Any] | None:
    """Pull the module-level CONTRACT = {...} literal from check.py via AST."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "CONTRACT":
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return None
    return None


def validate_task(task_dir: str | Path, claim_spec: str | Path | dict) -> list[str]:
    task_dir = Path(task_dir)
    spec = load_claim_spec(claim_spec) if isinstance(claim_spec, (str, Path)) else claim_spec
    c = derive_contract(spec)
    problems: list[str] = []
    add = problems.append

    input_dir = task_dir / "input_files"
    reward_dir = task_dir / "reward"

    # --- ClawGym structural contract ------------------------------------- #
    de_path = task_dir / "data_entry.json"
    data_entry: dict[str, Any] = {}
    if not de_path.is_file():
        add("data_entry.json missing")
    else:
        try:
            data_entry = json.loads(de_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            add(f"data_entry.json is not valid JSON: {e}")
        if "metadata" not in data_entry:
            add("data_entry.json missing required 'metadata'")
        else:
            meta = data_entry.get("metadata") or {}
            if meta.get("claim_id") != spec["claim_id"]:
                add("data_entry.json metadata claim_id mismatch")
            if spec.get("claim_uid") and meta.get("claim_uid") != spec.get("claim_uid"):
                add("data_entry.json metadata claim_uid mismatch")
            if spec.get("contract_hash") and meta.get("contract_hash") != spec.get("contract_hash"):
                add("data_entry.json metadata contract_hash mismatch")
            if meta.get("spec_hash") != spec["spec_hash"]:
                add("data_entry.json metadata spec_hash mismatch")
            if meta.get("pool") != c["verification_pool"]:
                add("data_entry.json metadata pool mismatch")
    if not input_dir.is_dir():
        add("input_files/ missing")
    if not (reward_dir / "reward.sh").is_file():
        add("reward/reward.sh missing")

    # --- expected.json ---------------------------------------------------- #
    exp_path = input_dir / "expected.json"
    if not exp_path.is_file():
        add("input_files/expected.json missing")
    else:
        try:
            exp = json.loads(exp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            add(f"expected.json invalid JSON: {e}")
            exp = {}
        exp_metrics = {m["name"] for m in exp.get("primary_metrics", [])}
        if exp_metrics != set(c["metric_names"]):
            add(f"expected.json metrics {sorted(exp_metrics)} != spec {sorted(c['metric_names'])}")
        for m in exp.get("primary_metrics", []):
            if "pass_threshold" in m:
                want = c["visible_thresholds"].get(m["name"])
                if want is None:
                    add(f"expected.json exposes threshold for hidden/absent metric '{m['name']}'")
                elif m["pass_threshold"] != want:
                    add(
                        f"expected.json threshold {m['name']}={m['pass_threshold']} != spec {want}"
                    )
        if set(exp.get("allowed_verdicts", [])) != set(c["verdicts"]):
            add("expected.json allowed_verdicts disagree with verdict set")
        if exp.get("claim_id") != spec["claim_id"]:
            add("expected.json claim_id mismatch")
        if spec.get("claim_uid") and exp.get("claim_uid") != spec.get("claim_uid"):
            add("expected.json claim_uid mismatch")
        if spec.get("contract_hash") and exp.get("contract_hash") != spec.get("contract_hash"):
            add("expected.json contract_hash mismatch")
        if exp.get("spec_hash") != spec["spec_hash"]:
            add("expected.json spec_hash mismatch")

    # --- protocol.yaml ---------------------------------------------------- #
    proto_path = input_dir / "protocol.yaml"
    if not proto_path.is_file():
        add("input_files/protocol.yaml missing")
    else:
        proto = yaml.safe_load(proto_path.read_text(encoding="utf-8")) or {}
        proto_metrics = set((proto.get("metric_computation") or {}).keys())
        if proto_metrics != set(c["metric_names"]):
            add(f"protocol metric_computation {sorted(proto_metrics)} != spec metrics")
        must_write = (proto.get("workspace_contract") or {}).get("agent_must_write", [])
        if list(must_write) != c["required_files"]:
            add("protocol agent_must_write != spec required_outputs.files")
        bad_verdicts = set((proto.get("verdict_rules") or {}).keys()) - set(c["verdicts"])
        if bad_verdicts:
            add(f"protocol verdict_rules has unknown verdicts: {sorted(bad_verdicts)}")
        if proto.get("claim_id") != spec["claim_id"]:
            add("protocol claim_id mismatch")
        if spec.get("claim_uid") and proto.get("claim_uid") != spec.get("claim_uid"):
            add("protocol claim_uid mismatch")
        if spec.get("contract_hash") and proto.get("contract_hash") != spec.get("contract_hash"):
            add("protocol contract_hash mismatch")
        if proto.get("spec_hash") != spec["spec_hash"]:
            add("protocol spec_hash mismatch")

    # --- params.yaml ------------------------------------------------------ #
    params_path = input_dir / "params.yaml"
    if not params_path.is_file():
        add("input_files/params.yaml missing")
    else:
        params = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
        if params.get("claim_id") != spec["claim_id"]:
            add("params.yaml claim_id mismatch")
        if spec.get("claim_uid") and params.get("claim_uid") != spec.get("claim_uid"):
            add("params.yaml claim_uid mismatch")
        if spec.get("contract_hash") and params.get("contract_hash") != spec.get("contract_hash"):
            add("params.yaml contract_hash mismatch")
        if params.get("spec_hash") != spec["spec_hash"]:
            add("params.yaml spec_hash mismatch")

    # --- task.md presence checks ----------------------------------------- #
    task_md_path = input_dir / "task.md"
    if not task_md_path.is_file():
        add("input_files/task.md missing")
    else:
        task_md = task_md_path.read_text(encoding="utf-8")
        for name in c["metric_names"]:
            if name not in task_md:
                add(f"task.md does not mention metric '{name}'")
        for f in c["required_files"]:
            if f not in task_md:
                add(f"task.md does not mention required output '{f}'")

    # --- reward/targets.yaml --------------------------------------------- #
    targets_path = reward_dir / "targets.yaml"
    if not targets_path.is_file():
        add("reward/targets.yaml missing")
    else:
        targets = yaml.safe_load(targets_path.read_text(encoding="utf-8")) or {}
        tgt = {k: v.get("pass_threshold") for k, v in (targets.get("primary_thresholds") or {}).items()}
        if tgt != c["thresholds"]:
            add(f"reward/targets.yaml thresholds {tgt} != spec {c['thresholds']}")
        if targets.get("claim_id") != spec["claim_id"]:
            add("reward/targets.yaml claim_id mismatch")
        if spec.get("claim_uid") and targets.get("claim_uid") != spec.get("claim_uid"):
            add("reward/targets.yaml claim_uid mismatch")
        if spec.get("contract_hash") and targets.get("contract_hash") != spec.get("contract_hash"):
            add("reward/targets.yaml contract_hash mismatch")
        if targets.get("spec_hash") != spec["spec_hash"]:
            add("reward/targets.yaml spec_hash mismatch")
        verification = targets.get("verification") or {}
        if verification.get("pool") != c["verification_pool"]:
            add("reward/targets.yaml verification pool mismatch")

    missing_thresholds = sorted(set(c["metric_names"]) - set(c["thresholds"]))
    for metric in spec.get("metrics") or []:
        if not isinstance(metric, dict):
            continue
        problem = formula_problem(metric.get("formula"))
        if problem:
            add(f"metric formula for '{metric.get('name')}' is not executable by check.py: {problem}")
    if c["verification_pool"] == "rlvr":
        if c["verification_mode"] not in {"numeric_threshold", "directional", "structural"}:
            add(f"rlvr task has invalid verification mode: {c['verification_mode']}")
        if missing_thresholds:
            add(
                "rlvr task has no executable threshold for metric(s): "
                + ", ".join(missing_thresholds)
            )
        for metric in c["metric_names"]:
            details = c["threshold_details"].get(metric) or {}
            evidence = details.get("target_evidence") or {}
            if not (details.get("source") or evidence.get("source")):
                add(f"rlvr threshold for metric '{metric}' is missing target evidence source")
            if not (details.get("rationale") or evidence.get("read_from")):
                add(f"rlvr threshold for metric '{metric}' is missing target evidence rationale")

    # --- exposure: no hidden threshold value may appear in input_files/ --- #
    if input_dir.is_dir():
        visible_blob = ""
        for p in sorted(input_dir.rglob("*")):
            if p.is_file():
                try:
                    visible_blob += "\n" + _visible_text_for_leak_scan(p.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, OSError):
                    continue
        for metric, value in c["hidden_thresholds"].items():
            for s in _num_strings(value):
                if _has_number_literal(visible_blob, s):
                    add(f"exposure leak: hidden threshold {metric}={value} appears in input_files/")
                    break

    # --- reward/check.py CONTRACT ---------------------------------------- #
    check_path = reward_dir / "check.py"
    if not check_path.is_file():
        add("reward/check.py missing (author it via the build-task skill)")
    else:
        contract = _extract_check_contract(check_path.read_text(encoding="utf-8"))
        if contract is None:
            add("reward/check.py does not declare a literal CONTRACT dict")
        else:
            if set(contract.get("metrics", [])) != set(c["metric_names"]):
                add("check.py CONTRACT metrics disagree with spec")
            if spec.get("contract_hash") and contract.get("contract_hash") != spec.get("contract_hash"):
                add("check.py CONTRACT contract_hash disagrees with spec")
            if dict(contract.get("thresholds", {})) != c["thresholds"]:
                add("check.py CONTRACT thresholds disagree with spec")
            if list(contract.get("required_files", [])) != c["required_files"]:
                add("check.py CONTRACT required_files disagree with spec")
            if set(contract.get("verdicts", [])) != set(c["verdicts"]):
                add("check.py CONTRACT verdicts disagree with spec")

    return problems
