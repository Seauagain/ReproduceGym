"""Task factory: paper -> claim spec -> ClawGym-compatible sandbox task.

  parse                PDF -> paper.md + figures/          (MinerU)
  extract_claims       claim text + anchors + type/cost/verifiability (Claude)
  extract_figure_params figure -> params/targets           (Qwen-VL)
  triage               build[]/defer[]/v0 + resource_profile (Claude)
  merge_claim_spec     -> canonical claim spec (single source of truth)
  render_task          claim spec -> task.md/params/protocol/expected/data_entry
  validate_task        consistency gate across all rendered files + check.py

Authoring of reward/check.py itself is the build-task skill, not this package.
All steps are deterministic API orchestration except the skill. Stubs only.
"""
