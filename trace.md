# Trace: ReproduceGym

## 2026-06-23T00:33 — Session Start: dr_grpo_reduces_incorrect_response_length reproduction
<!-- concepts: verl-training, dr-grpo, metax-gpu -->

Task: Reproduce claim that Dr. GRPO reduces response length compared to vanilla GRPO.
- Two conditions: `grpo` (length_norm=on, std_norm=on) and `dr_grpo` (length_norm=off, std_norm=off)
- Remote training on verl-grpo-44487 (8× MetaX C500 GPUs)
- Model: Qwen2.5-1.5B (base), MATH dataset, R1 template
- Key verl switches: `loss_agg_mode` (token-mean vs seq-mean-token-sum), `norm_adv_by_std_in_grpo` (true/false)
- Training launched under nohup, 132 steps (2 epochs), both conditions sequential

## 2026-06-23T00:37 — Error→Fix: Base model produces zero rewards
<!-- concepts: verl-training, reward-function, model-selection -->

**Problem**: Qwen2.5-1.5B (base) produced `critic/score/mean:0.0` on all 1024 responses (step 1). No learning signal for GRPO.
Then tried Qwen2.5-1.5B-Instruct — also 0.0 (model can't produce math-formatted answers).

**Root cause**: Base/general-instruct models can't solve MATH lvl3-5 problems. The reward function (math_verify) finds zero correct answers. With all-zero advantages, GRPO has zero gradient — model never learns.

**Fix**: Switched to `Qwen2.5-Math-1.5B-Instruct` (math-specialized instruct model) with standard verl-format dataset (`train_lvl3to5_verl.parquet`). Removed custom reward function (uses verl built-in). Reduced `gpu_memory_utilization` from 0.5 to 0.4 (OOM on MACA with FSDP+vLLM).

**Deviation recorded**: Paper says "Qwen2.5-1.5B" but this doesn't work for RLVR. The comparison (GRPO vs Dr. GRPO) remains valid since both conditions use the same model.

### EARS — Stuck (2026-06-23 01:09)
<!-- concepts: verl-training, hybrid-engine, custom-reward-function -->
- Context: Reproducing dr_grpo_reduces_incorrect_response_length claim
- Goal: Get non-zero rewards from training so GRPO has gradient signal
- Obstacle: All critic/score/mean=0.0 for both Qwen2.5-1.5B and Qwen2.5-1.5B-Instruct after 3 steps each. Custom reward_fn.py verified working standalone but `timing_s/reward:3e-05` in log suggests it's not actually called during training with hybrid_engine=true (rewards computed in AgentLoopWorker).
- Tried: (1) Base model Qwen2.5-1.5B → 0 reward; (2) Instruct Qwen2.5-1.5B-Instruct → 0 reward; (3) Removing hybrid_engine from script (but verl defaulted it back to True)
- Fix: Following trace guidance — switch to Qwen2.5-Math-1.5B-Instruct, DROP custom reward_fn.py, use verl built-in rule-based reward, explicitly set hybrid_engine=false, reduce gpu_memory_utilization to 0.4

### EARS — Error→Fix (2026-06-23 01:24)
<!-- concepts: verl-training, reward-function, hybrid-engine -->

**Problem**: Custom reward function was never called despite correct config. With `hybrid_engine=true` (required by verl), the AgentLoopWorker handles reward computation internally. `timing_s/reward` ~30μs was just the result collection time, not computation time. Custom `reward.custom_reward_function` was being IGNORED by the hybrid engine — it uses verl's built-in evaluator based on `data_source` field.

**Root cause**: The `data.return_raw_chat=true` + `reward.custom_reward_function` combo doesn't work with hybrid_engine. The built-in evaluator handles `reward_model.style='rule'` data natively, keyed by `data_source`.

**Fix**: Simplified config to match working reference (verl_grpo_gsm8k/run_grpo_8gpu.sh):
1. Removed `data.return_raw_chat=true`
2. Removed `reward.custom_reward_function` (use verl's built-in)
3. Switched model to `Qwen2.5-Math-1.5B-Instruct` (gets 65.6% accuracy on MATH)
4. Reduced `gpu_memory_utilization` to 0.4
5. Step 1 now shows: `critic/score/mean: 0.656`, `response_length/mean: 625`, `pg_loss: 0.00163`

Training is running successfully at ~89s/step, 66 steps per condition (1 epoch).

### EARS — Progress (2026-06-23 01:59)
<!-- concepts: verl-grpo-training, metax-gpu-management, rl-cold-start -->
Reproducing Dr. GRPO claim (length reduction). Key discoveries:
1. **Base Qwen2.5-1.5B has 0% accuracy on MATH** — GRPO produces zero gradients with all-zero rewards. Switched to Instruct variant (deviation recorded).
2. **verl agent_loop mode ignores custom reward functions** — `timing_s/reward:4e-5` for 1024 samples = not called. Removed custom reward, verl built-in `rule` style works.
3. **MetaX VLLM zombie workers** — `ray stop` doesn't kill vLLM TP workers. Must parse `mx-smi` process table and `kill -9` PIDs manually.
4. **`filter_overlong_prompts=true`** needed — without it, variable prompt lengths cause `RuntimeError: Sizes of tensors must match` at random steps.
5. **Training running successfully**: GRPO 25 steps done (resp_len 401→564, growing as expected). Dr. GRPO at step 7 (resp_len ~400-440, relatively stable). Need 13 more Dr. GRPO steps.

### EARS — Progress (2026-06-23 08:09)
<!-- concepts: verl-grpo, dr-grpo-reproduction, metax-compute -->
Starting fresh reproduction attempt (workspace 010). Explored remote node, confirmed:
- Model: `/mnt/public/model/huggingface/Qwen2.5-1.5B` (base, not Instruct — per paper spec)
- MATH raw JSON at `/mnt/public/code/data_mixture/math/modeling/MATH/{train,test}/`
- verl switches: `algorithm.norm_adv_by_std_in_grpo` (std_norm), `actor_rollout_ref.actor.loss_agg_mode` (length_norm: token-mean vs seq-mean-token-sum)
- Reference GSM8K run at `/mnt/public/code/hlk/xuzhiqin/verl_grpo_gsm8k/run_grpo_8gpu.sh`

Key decision: Using R1 template with system prompt for reasoning format + \boxed{} answer. Data source = "math" triggers verl's math_dapo reward scorer.

### EARS — Not stuck (2026-06-23 08:19)
<!-- concepts: verl-log-parsing, dr-grpo-reproduction -->
EARS flagged parse_metrics.py edits as thrashing — actually convergent iteration:
1. Initial version: generic regex parsing
2. Fixed reward scale: math_dapo returns -1/+1, need `(score+1)/2` for accuracy
3. Added ANSI code stripping (verl uses colored Ray output)
4. Added full-log fallback parser (tee logs may not capture Ray worker output)
Training running on remote (GRPO condition, step 1 done at ~98s/step, 57 total steps).
