# Prompt: Read Experimental Evidence from Figures (Multimodal)

> Consumed by `pipeline/extract_figure_params.py`.

## Goal

Given a figure image plus its paper ref, caption, and nearby Markdown context,
read the experimental evidence that a reproduction task may depend on:

- visible reproduction parameters: axis-stated training steps, conditions,
  dataset names, model sizes, seeds, hyperparameter labels, window sizes
- hidden targets: curve endpoints, approximate ratios, pass/fail thresholds, or
  paper result values that should be used by a verifier but not shown to the
  reproduction agent
- plot structure: axis ranges, condition labels, panels/subplots, and what each
  line/bar represents

Do not guess. If a value is not legible, omit it or mark low confidence.

## Output

Strict JSON object. No prose outside JSON.

```json
{
  "figure_ref": "Fig. N",
  "claim_relevance": "what claims this figure can support",
  "params": [
    {
      "name": "policy_iteration_steps",
      "value": 150,
      "unit": "steps",
      "source": "Fig. 5",
      "visibility": "visible",
      "use": "reproduction_param",
      "confidence": 0.95,
      "read_from": "x-axis spans 0 to 150 Policy iteration step"
    }
  ],
  "targets": [
    {
      "name": "response_length_ratio",
      "value": 0.9,
      "source": "Fig. 5",
      "visibility": "hidden",
      "use": "target",
      "confidence": 0.7,
      "read_from": "visual endpoint ratio between DR-GRPO and GRPO"
    }
  ],
  "axis_ranges": {
    "x": {"label": "Policy iteration step", "min": 0, "max": 150},
    "y": {"label": "Response length", "min": null, "max": null}
  },
  "conditions": ["GRPO", "DR-GRPO"],
  "confidence": 0.9
}
```

## Notes

- Hidden targets are answer-key material: downstream they are routed into
  reward/, never input_files/.
- Visible reproduction parameters are allowed in params.yaml because the agent
  needs them to run the experiment faithfully.
- Prefer explicit axis labels/ticks over visual estimates. For visual estimates,
  state that in `read_from` and lower confidence.
