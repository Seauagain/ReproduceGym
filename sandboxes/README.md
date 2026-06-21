# sandboxes/ — produced reproduction units

One directory per paper. Schema is compatible with `../../RL/sandboxes/<paper>/`,
so the same tasks feed both interactive reproduction and RL rollout.

```
<paper_id>/
├── paper.md  paper.json        # parsed paper (MinerU)
├── figures/                    # extracted figures (public; optional in input_files)
├── claims/                     # SOURCE OF TRUTH (git, human-reviewed)
│   ├── figure_params.yaml      #   whole-paper Qwen-VL param inventory
│   └── <claim_id>.yaml         #   canonical claim spec (schema/claim_spec.schema.json)
└── tasks/                      # RENDERED ClawGym-compatible tasks (git)
    └── <claim_id>/
        ├── data_entry.json     # task_id, user_query, metadata (REQUIRED), input_mount_dir
        ├── input_files/        # the ONLY thing mounted to the agent
        │   ├── task.md  params.yaml  protocol.yaml  expected.json
        │   ├── paper.md / paper_excerpt.md          # per exposure_policy
        │   ├── figures/  starter/
        └── reward/             # verifier-only; copied in ONLY at scoring
            ├── reward.sh       # entry; last stdout line = scalar reward
            ├── check.py        # authored by the build-task skill
            └── targets.*       # hidden thresholds/targets (exposure: hidden)
```

Flow: `claims/<id>.yaml` (source) → `render_task` → `tasks/<id>/` (deterministic)
→ `build-task` skill writes `reward/check.py` → `validate_task` consistency gate.

Authoring layout is nested; the rollout needs a flat dir, so `reprogym.dataset`
symlinks selected task dirs into `../datasets/<name>/`. See `schema/task_contract.md`.
