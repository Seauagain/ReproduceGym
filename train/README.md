# train/ — RL training rollout mode

The second use of a sandbox task: in RL training, after each training step the
system launches the sandbox, runs the task, and the hidden verifier scores it;
the rollout produces a trajectory and a reward used to update the policy.

A ReproduceGym task dir is byte-for-byte consumable by the existing rollout in
`../../RL/ClawGym-Agents/RL/clawgym_rl_rollout.py`. The only glue:

1. `reproducegym.dataset.build_dataset` / `rollout_adapter.as_rollout_source`
   flattens selected `runs/<paper>/03-task/<claim>/` into a flat
   `datasets/<name>/` of symlinks (the rollout scans only one level).
2. Point the rollout's `source_path` at `datasets/<name>/`.

No task changes between interactive reproduction and training rollout — same
task description, same verifier.
