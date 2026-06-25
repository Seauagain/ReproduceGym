# Claim-First RLVR Task Pipeline Plan

## Goal

Build RLVR tasks from papers by following the paper's scientific logic:

```text
paper claim -> supporting experiment/table/figure -> measurable artifact -> reward
```

The pipeline must not start from images and invent claims. It should extract the
authors' claims from text, then use figures, tables, captions, and nearby
sections only to decide whether each claim can become a quantified reproduction
task.

The output of a successful build is not just a rendered task directory. It is a
task whose reward can be recomputed from submitted artifacts and scored against
paper-grounded target sources.

## Defects To Avoid

This plan is a generic refactor plan for the claim extraction and task build
pipeline. It uses recent Dr-GRPO and DeepSeek-R1 failures as regression cases,
but the design must work for any RL/ML paper with text claims, tables, figures,
or mixed evidence.

The refactor must avoid these failure modes:

- **Claim without target**: important prose claims were rendered as tasks even
  when no metric had a paper-grounded reward curve.
- **Directional fallback overuse**: the build used directional thresholds such
  as `metric > 0` even when the paper contained numeric target values in
  figures or tables.
- **Missed supporting panel**: Dr-GRPO Fig. 5 panel 2 was mentioned in text but
  not promoted into structured targets, even though it is central to the claim
  about output length runaway.
- **Wrong evidence granularity**: Fig. 5 panel 4/5 were treated as the whole
  Dr-GRPO verification target, while panel 2 should be the primary evidence for
  training-time length control and panel 4/5 should be auxiliary evaluation
  evidence.
- **Weak target binding**: targets such as percentages, ratios, token lengths,
  and pass rates can be attached to the wrong metric if only names are matched.
- **Reward tied to verdict**: reward must be computed from metrics and target
  curves. Verdict is reporting only and must never determine reward.
- **Partial metric reward**: every required metric must contribute to final
  reward. A claim with missing reward curves is not an RLVR task.
- **Expensive broad VL pass**: parsing all figures for all claims wastes time
  and still misses the panel that matters for a specific claim.
- **Ambiguous downstream task lookup**: hash-versioned task directories are fine
  for reproducibility, but the build must emit a manifest mapping `claim_id` to
  the exact task directory so downstream agents do not guess.

## Non-Negotiable Invariants

1. The claim comes from the paper's text, usually abstract, introduction,
   results, discussion, or figure-referenced paragraphs.
2. A claim enters `rlvr` only if it has at least one measurable artifact the
   reproduction agent can produce.
3. Every required metric has a target source and a reward curve.
4. Every target is bound to a metric, condition, unit, and source location.
5. `reward/check.py` recomputes metrics from workspace outputs. It does not
   trust agent-provided scores, verdicts, or claims.
6. Final reward is computed from metric rewards only. Verdict is derived after
   reward computation for human reporting.
7. Image/VL parsing is claim-scoped and anchor-scoped. The pipeline does not run
   an expensive global figure parse before it knows which claims need which
   figures.
8. A task is accepted only after synthetic selftests prove target-like,
   threshold-like, poor, and missing submissions receive ordered rewards.

## Artifact Model

The build should use one canonical artifact at each boundary:

```text
paper.md + figures.index.json
  -> paper_evidence_index.json
  -> candidate_claims.json
  -> triaged_claims.json
  -> claim_evidence/<claim_uid>.json
  -> target_points/<claim_uid>.json
  -> compiled_contracts.json
  -> selected_claims_for_build.json
  -> 02-spec/
  -> 03-task/
  -> build_validation.json
  -> task_manifest.json
```

### `paper_evidence_index.json`

Local deterministic index built once from `paper.md` and `figures.index.json`.
It is not an LLM artifact and is reused by all claim-scoped stages. It contains:

- section tree and heading spans;
- paragraph ids with character offsets;
- figure/table captions and their nearby paragraphs;
- markdown table blocks with stable ids;
- figure references and image paths;
- optional text embeddings or keyword index for bounded fallback retrieval.

Claim-scoped evidence binding must read from this index. It must not send the
whole paper back to a model for each candidate claim.

### `candidate_claims.json`

Text-only whole-paper extraction. Each item contains:

- `claim_uid`: deterministic hash from normalized statement and evidence
  anchors.
- `statement`: the author's claim in one sentence.
- `paper_claim_type`: e.g. method_effect, ablation_mechanism, benchmark_result,
  scaling_behavior, diagnostic_observation.
- `support_anchors`: figure/table/section references cited by the paper text.
- `observable_hints`: metrics the paper appears to measure, without thresholds.
- `cost_hint`: inference, evaluation, short training, long training.
- `risk_notes`: missing datasets, huge model, visual-only target, ambiguous
  protocol.

The prompt must not output thresholds, reward, verifier rules, or hidden target
values.

### `triaged_claims.json`

Deterministic ranking and gating before any VL call. Each candidate gets:

- `importance_score`: centrality to the paper's contribution.
- `quantifiability_score`: whether support anchors expose numeric tables,
  curves, bars, or explicit values.
- `reproducibility_score`: whether an agent can produce the required artifacts.
- `cost_score`: lower is better.
- `route`: `evidence_binding`, `exploration`, or `reject`.
- `route_reason`: short explanation.

Only the top `max_claims_for_evidence` candidates, default 3 to 5, proceed to
claim-scoped evidence binding.

### `claim_evidence/<claim_uid>.json`

Claim-scoped evidence bundle. It contains the paper text slices and media needed
for this claim only:

- claim statement and anchors;
- supporting paragraphs around each anchor;
- figure captions and table text;
- resolved image files and panel hints;
- parsed tables from markdown where available;
- any previous cached VL reads keyed by image hash and prompt version.

For a figure with multiple panels, the bundle must preserve panel identity. A
claim about Fig. 5 panel 2 must not be satisfied by values from panel 4.

### `target_points/<claim_uid>.json`

Structured target extraction from claim evidence. This is the first stage that
may read image pixels. Each item is a target point:

```json
{
  "observable": "output_length",
  "condition": "Dr-GRPO",
  "value": 520,
  "unit": "tokens",
  "source": {"kind": "figure", "ref": "Fig. 5", "panel": "2"},
  "read_from": "red curve endpoint around step 150",
  "step": 150,
  "confidence": 0.7,
  "tolerance": {"rel": 0.3}
}
```

Tables should be preferred over visual estimates when both exist. Visual curve
reads must include tolerance and confidence. Non-numeric values such as
`stable`, `minimal`, or `slight` are not RLVR targets unless the compiler turns
them into a documented numeric reward curve from nearby numeric evidence.

### `compiled_contracts.json`

Deterministic compiler output. It maps each claim to:

- `required_outputs`: files and schemas the reproduction agent must write.
- `metrics`: executable formulas over those outputs.
- `reward_curves`: one curve per required metric.
- `aggregation`: default `min`.
- `target_bindings`: metric-to-target-source mapping.
- `pool`: `rlvr` or `exploration`.
- `rejection_reasons`: if not `rlvr`.
- `contract_hash`: hash of the executable contract and accepted target sources.

The compiler, not the LLM, decides the final pool and reward contract.

Each reward curve must use the canonical form consumed by the verifier:

```json
{
  "metric": "final_length_ratio",
  "direction": "lower_is_better",
  "points": [
    {"value": 0.85, "reward": 0.0},
    {"value": 0.65, "reward": 0.5},
    {"value": 0.50, "reward": 1.0}
  ],
  "source": {"kind": "figure", "ref": "Fig. 5", "panel": "2"},
  "rationale": "target and tolerance derived from paper figure read"
}
```

The compiler may convert a target point plus tolerance into this curve, but it
must record the derived fail/pass/target values and rationale. A target point
without enough information to derive a curve cannot produce an `rlvr` metric.

### `selected_claims_for_build.json`

The only input to `merge_claim_spec`. Items must already include the compiled
contract. `merge_claim_spec` may validate and render the contract, but must not
invent new targets or silently change the pool.

Each selected item must contain the downstream-facing fields that already flow
through the spec renderer:

- `claim_uid`;
- final `claim_id`;
- `statement`;
- `reproduction_protocol`;
- `verification_contract`;
- `verification.pool`;
- `reward_curves`;
- `contract_hash`.

The renderer may reshape these fields into `02-spec`, `reward/targets.yaml`, and
embedded `check.py` constants, but the semantic contract must remain identical.

### `build_validation.json`

Task-level validation output written after render. It records, for every task:

- schema validation status;
- contract hash consistency status;
- leak scan status;
- formula executability status;
- synthetic selftest rewards;
- optional historical replay result;
- final accept/reject decision.

`task_manifest.json` may list only tasks accepted in `build_validation.json`.

### `task_manifest.json`

A downstream-facing manifest written after render:

```json
{
  "paper_id": "2503-dr-grpo",
  "tasks": [
    {
      "claim_id": "c001_output_length_runaway",
      "claim_uid": "clm_...",
      "contract_hash": "...",
      "spec_hash": "...",
      "task_dir": "03-task/c001_output_length_runaway/<spec_hash>",
      "pool": "rlvr"
    }
  ]
}
```

Downstream agents consume this manifest instead of guessing hash directories.

## Pipeline Stages

### 1. Reuse Parse Bundle

Use `runs/<paper>/00-parse` when it already exists. Do not re-run PDF parsing
for claim/task iteration. PDF parsing is a separate stage and should be cached by
paper id, PDF hash, and parser version.

Build `paper_evidence_index.json` locally from the parse bundle before the
LLM-based claim pass. This keeps later claim evidence binding bounded and avoids
per-claim whole-paper rereads.

### 2. Text-Only Claim Extraction

Run one whole-paper text pass over `paper.md`, captions, and markdown tables.
The prompt asks for the paper's claims and their supporting anchors, not final
verifier details.

This stage should prefer claims that the authors explicitly support with
experiments. It should not create claims by looking at figure values alone.

### 3. Deterministic Claim Triage

Before any image parsing:

- keep claims with strong paper centrality and concrete support anchors;
- downgrade claims that are purely theoretical, too broad, or unsupported by
  measurable artifacts;
- avoid claims that require reproducing a massive model unless the output can be
  replayed from an existing workspace or a small proxy is explicitly acceptable;
- cap the number of evidence-bound claims.

This stage is where speed is won. Most candidates should never reach VL.

### 4. Claim-Scoped Evidence Binding

For each triaged claim:

1. Collect nearby text and tables for the cited anchors from
   `paper_evidence_index.json`.
2. Resolve figure references to image files.
3. Ask a lightweight text model to identify which panel or table entries matter.
4. Run VL only on those image/panel regions when text/table evidence is
   insufficient.
5. Emit target points with source, condition, observable, value, unit, and
   tolerance.

If the original anchors are incomplete, the stage may run bounded local retrieval
over `paper_evidence_index.json` using claim keywords and observable hints. It
may not expand to a fresh whole-paper LLM read. If bounded retrieval cannot find
a target source, the claim routes to `exploration`.

The VL prompt must be claim-specific. For example, for Dr-GRPO Fig. 5 it should
ask separately about panel 1 reward, panel 2 output length, panel 4 incorrect
output length, and panel 5 benchmark score when those panels support the claim.

### 5. Contract Compilation

Compile target points into an executable RLVR contract.

Rules:

- Each required output column is named before metrics are generated.
- Metric formulas use only supported verifier grammar.
- Units must match: token length targets bind only to length metrics; percentages
  bind only to score/pass-rate metrics; ratios bind only to ratio metrics.
- Conditions must match: `Dr-GRPO`, `GRPO`, `w/o length norm`, and `w/o std`
  cannot be conflated.
- Panel/table sources must match the claim. Fig. 5 panel 2 cannot substitute for
  Fig. 5 panel 4, and vice versa.
- If a directional metric is kept, it still receives a reward curve with
  `fail_threshold`, `pass_threshold`, and `target_value` justified from paper
  evidence or documented tolerance.

Default final reward:

```text
final_reward = min(metric_rewards)
```

Weighted mean is allowed only when the contract explicitly says why a secondary
metric is an auxiliary guard rather than a core reproduction target.

### 6. Render, Validate, Selftest

Render specs and task directories only for compiled contracts. Then run:

- schema validation;
- contract hash consistency checks across spec, targets, and `check.py`;
- leakage checks for hidden target values;
- formula executability checks;
- synthetic selftests:
  - target-like workspace -> reward 1.0;
  - pass-threshold workspace -> reward around 0.5 or documented threshold reward;
  - poor workspace -> low reward;
  - missing output -> 0.0.

If historical workspaces exist for the same claim, replay them through the new
verifier before accepting the task as a replacement.

## Generic Claim Patterns

The pipeline should recognize claim patterns that often appear in RL/ML papers
and compile each into a standard task shape when target sources exist.

### Method effect claim

The paper claims a method changes an observable while preserving or improving a
quality metric.

Common anchors:

- training curves;
- evaluation summaries;
- ablation tables;
- result paragraphs that explicitly refer to figures or tables.

Required output shape:

```csv
method,step_or_split,primary_observable,guard_metric
method_a,...
baseline,...
```

Typical metrics:

- final observable ratio or difference;
- slope/growth-rate difference for curve claims;
- non-degradation gap for reward, accuracy, score, or pass rate.

Every metric must bind to a target source with matching unit and condition.

### Ablation mechanism claim

The paper claims one component is responsible for an observed effect.

Common anchors:

- ablation tables;
- multi-condition curves;
- "w/o component" or "only component" comparisons.

Required output shape:

```csv
condition,metric_value
full_method,...
baseline,...
ablation_a,...
ablation_b,...
```

Typical metrics:

- ablated variant closeness to full method;
- ablated variant closeness to baseline;
- dominance ratio between component effects.

The compiler must keep condition names distinct. It must not conflate
`without length normalization`, `without standard deviation normalization`, and
`full method` just because all appear in the same figure.

### Benchmark result claim

The paper claims a model reaches or exceeds a benchmark score.

Common anchors:

- result tables;
- benchmark leaderboards;
- explicit text values.

Required output shape:

```csv
model,benchmark,metric,value
target_model,AIME_2024,pass1,...
```

Typical metrics:

- per-benchmark score;
- mean score over a stated benchmark set;
- score gap against a named baseline.

Tables are preferred over visual reads. Baseline gaps must bind both the target
model value and baseline value to the same benchmark and metric.

### Scaling or dynamics claim

The paper claims an observable changes with training step, compute, problem
difficulty, dataset size, or another axis.

Common anchors:

- learning curves;
- compute-scaling plots;
- difficulty-stratified plots;
- text descriptions with endpoint values.

Required output shape:

```csv
condition,x,value
method,0,...
method,100,...
```

Typical metrics:

- endpoint value;
- slope over a declared window;
- monotonicity or correlation over bins;
- ratio between easy and hard / early and late / low and high settings.

The target source must include the axis, window, and condition. A curve endpoint
cannot substitute for a whole-curve claim unless the contract explicitly says
the endpoint is the chosen measurable proxy.

### Diagnostic observation claim

The paper claims a specific phenomenon explains or motivates a method.

Common anchors:

- exploratory figures;
- error analysis;
- prompt/template diagnostics;
- data distribution summaries.

These claims enter `rlvr` only when they have clear metrics and target sources.
Otherwise they remain `exploration`, even if scientifically important.

## Regression Cases

These are not special-case implementations. They are fixtures that prove the
generic pipeline avoids known mistakes.

### Dr-GRPO Fig. 5 Panel Binding

The generic method-effect and scaling/dynamics logic must extract the author
claim about output length runaway from text, then bind:

- training reward to Fig. 5 panel 1;
- training output length to Fig. 5 panel 2;
- incorrect evaluation length to Fig. 5 panel 4 only for an evaluation-side
  claim;
- average benchmark score to Fig. 5 panel 5 as a non-degradation guard.

Panel 4 must not satisfy a training-time length-runaway claim. Panel 2 must not
be dropped just because panel 4 has an easier visual endpoint.

### DeepSeek-R1 Table/Figure Separation

The generic benchmark-result logic must prefer table values for benchmark pass
rates when tables exist. Figure-derived difficulty or compute-scaling values can
support separate scaling/dynamics claims, but must not be merged into unrelated
benchmark contracts.

## Speed Budget

For an existing parse bundle, target build latency should be:

- text-only claim pass: 1 LLM call;
- deterministic triage: local only;
- evidence binding: at most 3 to 5 claims;
- VL: only anchored figures/panels, usually 1 to 3 images;
- contract compilation/render/validation: local plus selftests.

Expected build time from existing `paper.md`: 5 to 15 minutes. Any build over 20
minutes must report which stage consumed time and tokens.

Caching requirements:

- cache text claim extraction by `paper_hash + prompt_version + model`;
- cache target point extraction by `claim_uid + image_hash + prompt_version +
  model`;
- cache compiled contracts by `claim_uid + target_points_hash +
  compiler_version`;
- record cache hits in token/timing logs.

## Implementation Steps

1. Add `candidate_claims.json` and `triaged_claims.json` generation to
   `build_claim_tasks.py`.
2. Add a deterministic triage module for importance, quantifiability, cost, and
   route decisions.
3. Replace broad figure parsing with claim-scoped evidence binding and target
   point extraction.
4. Add a target-point schema and unit/condition/panel binding checks.
5. Add a contract compiler that requires reward curves for every RLVR metric.
6. Update `render_check.py` and verifier engine so reward is computed only from
   metric reward curves, with verdict derived afterward.
7. Update `validate_task.py` to reject RLVR tasks missing reward curves,
   synthetic selftests, or manifest entries.
8. Write `build_validation.json` after render and selftests.
9. Write `task_manifest.json` from accepted validation results.
10. Add historical replay checks for Dr-GRPO and DeepSeek-R1 where existing
   workspaces can be adapted.

## Acceptance Tests

Minimum tests before using the new path by default:

- A Dr-GRPO fixture with Fig. 5 panel 2 evidence produces a contract containing
  `final_length_ratio`; panel 4 alone is insufficient for the training-runaway
  claim.
- A claim with only directional prose and no target source routes to
  `exploration`.
- A target from Fig. 5 panel 4 cannot bind to a training output-length metric
  from panel 2.
- A percentage target cannot bind to a token-length metric.
- A metric without a reward curve makes the task invalid for `rlvr`.
- Changing `verdict` text does not change reward for identical metrics.
- Synthetic target, threshold, poor, and missing workspaces receive ordered
  rewards.
- `build_validation.json` records the synthetic selftest rewards and final
  accept/reject decision for each rendered task.
- `task_manifest.json` maps every rendered `claim_id` to exactly one active
  accepted task directory.
- For an existing parse bundle, a build with `max_claims=3` does not run VL on
  figures unrelated to selected claims.
- Claim-scoped evidence binding reads from `paper_evidence_index.json` and does
  not send the full paper to the model per claim.

## Definition Of Done

The pipeline is ready when a paper build can answer, for every selected RLVR
task:

- What is the author claim?
- Which experiment/table/figure supports it?
- What exact output files must the reproduction agent produce?
- Which metrics are recomputed by the verifier?
- What target source and reward curve scores each metric?
- Why is this task `rlvr` rather than `exploration`?
- How long and how many tokens did each stage cost?

If any answer is missing, the task should not be treated as RLVR.
