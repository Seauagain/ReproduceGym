# Derived Target Contract Gaps

This note records a build-time gap exposed by the Dr-GRPO Fig. 8 ablation claim.
It is intentionally a backlog/design note, not an implementation plan.

## Problem

Some important paper claims are not verified by a single directly observed target
value. They require a deterministic derived target compiled from multiple
paper-grounded observations.

The current contract synthesizer is deterministic, but it mostly binds one
extracted target to one metric. It does not yet compile higher-level ablation
contrasts such as:

- "A is the primary causal factor, B is secondary."
- "Removing component X explains most of the observed effect."
- "Two variants both improve performance, but one specifically controls the
  length/efficiency mechanism."
- "Metric should remain near zero while another metric improves."

As a result, the pipeline can extract the evidence but still fail to produce an
RLVR task for the full scientific claim.

## Concrete Regression: Dr-GRPO Fig. 8

Paper claim:

> Removing the length normalization term is the primary factor responsible for
> reducing response length, while removing std normalization has a smaller effect
> on length but still improves accuracy over vanilla GRPO.

Paper experiment:

- RL-tune Qwen2.5-1.5B on a 3K mixed math set from ASDiv, MATH, and AIME
  pre-2023.
- Compare four variants:
  - vanilla GRPO
  - GRPO without length normalization
  - GRPO without std normalization
  - Dr. GRPO, with both terms removed
- Use Fig. 8 training reward, training length, and average benchmark score curves.

Paper-grounded observations extracted by the pipeline:

- `length_ratio_wo_length_norm_target ~= 0.5`
- `length_ratio_wo_std_norm_target ~= 0.8-0.9`
- `accuracy_difference ~= 2 pp` for the std-normalization ablation versus vanilla
  GRPO

What the contract should derive:

- `length_ratio_wo_length_norm_vs_grpo` should be `lower_is_better`, target about
  `0.5`.
- `length_ratio_wo_std_norm_vs_grpo` should also be `lower_is_better`, target
  about `0.8-0.9`.
- A causal contrast target should be compiled, for example:
  - `length_ratio_gap = r_std - r_len`, target about `0.3-0.4`; or
  - `length_norm_dominance_ratio = (1 - r_len) / (1 - r_std)`, target about
    `2.5-5.0` depending on visual read.

Current failure mode:

- The ratio metric was emitted with the wrong direction (`higher_is_better`
  instead of `lower_is_better`), so the direct target failed reward-curve
  validation.
- The synthesizer did not compile the two extracted ratios into a derived
  dominance/gap target, so the core "length norm is the primary factor" claim
  remained unverifiable as RLVR.

## Why This Matters

Strict RLVR gating is correct: reward should only depend on metrics with
paper-grounded targets and valid reward curves.

However, without derived target compilation, the gate is too lossy for many
scientific claims. Papers often prove mechanisms through controlled contrasts,
not isolated scalar targets. If we only bind direct targets, we risk selecting
easier table claims and missing important ablation/mechanism claims.

## Required Future Capability

Add a deterministic derived-target compilation layer after target binding and
before RLVR selection.

It should recognize at least these patterns:

- Ratio contrast:
  - Inputs: `ratio_variant_a_vs_baseline`, `ratio_variant_b_vs_baseline`
  - Derived metrics: `ratio_gap`, `dominance_ratio`
- Difference contrast:
  - Inputs: `delta_variant_a`, `delta_variant_b`
  - Derived metrics: `delta_gap`, `dominance_ratio`
- Near-zero constraint:
  - Inputs: "no notable improvement", "overlap", "within tolerance"
  - Derived target: `target_value = 0`, lower-is-better absolute difference
- Mechanism with performance guard:
  - Inputs: mechanism metric target plus accuracy/performance non-regression
  - Derived contract: reward over mechanism metric gated by performance metric

The layer must remain deterministic and auditable. Every derived target should
record:

- source input targets
- formula used to derive the target
- derived target value
- tolerance rule
- source figures/tables
- whether any visual estimate was used

## Non-Goals

- Do not let LLMs invent derived targets without source values.
- Do not accept directional-only dominance claims as strong RLVR.
- Do not hide unsupported metrics inside reward curves.
- Do not make the verifier depend on verdict labels.

## Acceptance Criteria

For the Dr-GRPO Fig. 8 claim, a future implementation should produce an accepted
RLVR task whose primary reward metrics include:

- response-length ratio for the length-normalization ablation
- response-length ratio for the std-normalization ablation, or a derived contrast
  between the two
- an accuracy/performance guard showing that removing bias terms does not simply
  reduce length by breaking learning

The task statement should be narrowed to exactly the verified aspects if some
claim components remain diagnostic.
