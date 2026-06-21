# Prompt: Read Experimental Params from Figures (Qwen-VL)

> Placeholder. Consumed by `pipeline/extract_figure_params.py`.

## Goal

Given a figure image (+ its caption/ref), read off experimental parameters and
quantitative targets that appear ONLY in the figure (curve endpoints, plateau
values, ratios, axis-stated hyperparameters, etc.).

## Output

Strict JSON list of entries:

```json
{ "name": "...", "value": 0.0, "unit": "", "source": "Fig. N", "confidence": 0.0,
  "read_from": "what in the figure this came from" }
```

## Notes

- These values are answer-key material: downstream they get `exposure: hidden`
  and are routed into reward/, never input_files/.
- If a value cannot be read confidently, omit it or set low confidence — do not
  guess. A missing target becomes `paper_unspecified`.
