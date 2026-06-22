"""ReproduceGym: automated RL-literature reproduction as sandbox tasks.

A paper is turned into ClawGym-compatible sandbox tasks, each runnable in two
modes from the same dir, unchanged:
  - interactive reproduction (host sandbox + reproduction agent, scored, traced)
  - RL training rollout (consumed by ClawGym-Agents/RL rollout)

Main control, sandbox, verifier and secrets all live on the HOST; compute nodes
are only reached by the in-sandbox agent when it needs GPUs.

Scaffold only — modules are stubs.
"""

__version__ = "0.0.0"
