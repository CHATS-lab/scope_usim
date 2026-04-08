# 0403 Experiment Launch TODO

## Current Status (2026-04-03 ~02:55 EDT)

### Branch: `exp/0403-evolving-tau2`

### tau2 Ensemble (chatslab) — RUNNING
- App: `ap-oaelFWJg02ztQiYPS5iA9d`
- Status: Training in progress, rollouts completing (reward=0 early on is normal)
- Known issue: `--rollout-max-response-len 16384` should be 32768 (fix pushed but not in this run)
- WandB: `usim / qwen3-4B-Instruct-2507-tau2-ensemble-0403`
- **Action**: Monitor. If it finishes or crashes, relaunch with latest branch (has 32k fix).

### P4G Ensemble (for-uw) — FAILED (CUDA illegal instruction)
- App: `ap-PwPT4yp1XJAv0WJ1BNkz9n` — crashed
- Error: `CUDA error: an illegal instruction was encountered` during SGLang CUDA graph capture
- This happens on `slime.train` (sync) with `--colocate`. tau2 uses same config and works.
- Possible causes:
  1. Stale GPU state from previous failed runs on the same Modal container
  2. CUDA driver mismatch (CUDA 12.9 image vs driver 580.95/CUDA 13.0)
  3. Race condition in engine init
- **Action**: Just retry — CUDA illegal instruction is often transient. If it fails again, try without `--context-parallel-size 2` (set to 1).

## Fixes Already Pushed (in branch)
1. ✅ PROJECT_ROOT depth (8→6 levels)
2. ✅ MODEL_DIR separate from WORKSPACE_DIR
3. ✅ train_p4g_slime.py switched to slime.train (sync, supports colocate)
4. ✅ OpenRouter API key updated in modal .env
5. ✅ gpt-5-mini routed through OpenAI directly (litellm+OpenRouter compat)
6. ✅ Truncation fix: only truncate when total_len > limit (not total+max_gen > limit)
7. ✅ tau2 rollout-max-response-len restored to 32768
8. ✅ JSON serialization fix for trajectory recorder (tau2 Task objects)
9. ✅ Cooperative reward mode for tau2 cotrain
10. ✅ --no-agent-kl flag for per-role KL control

## Priority 1: Get Ensemble Running (both tasks)

### P4G Ensemble
```bash
cd /Users/simonyu/local/local_orby/modal-examples && \
conda run -n modal env MODAL_PROFILE=for-uw modal run --detach \
    simon-exps/spare/slime_train.py \
    --project usim \
    --script cmd/slime/qwen3_4b_instruct/0403/ensemble/p4g/train.sh \
    --branch exp/0403-evolving-tau2
```
If CUDA error persists, try setting `--context-parallel-size 1` in the script.

### tau2 Ensemble
Already running. If crashed/finished, relaunch:
```bash
cd /Users/simonyu/local/local_orby/modal-examples && \
conda run -n modal env MODAL_PROFILE=chatslab modal run --detach \
    simon-exps/spare/slime_train.py \
    --project usim \
    --script cmd/slime/qwen3_4b_instruct/0403/ensemble/tau2/train.sh \
    --branch exp/0403-evolving-tau2
```

## Priority 2: Launch Cotrain (after ensemble done)

### P4G Evolving (dual_cotrain) — needs slime patch
```bash
cd /Users/simonyu/local/local_orby/modal-examples && \
conda run -n modal env MODAL_PROFILE=for-uw modal run --detach \
    simon-exps/spare/slime_train.py \
    --project usim \
    --script cmd/slime/qwen3_4b_instruct/0403/evolving/p4g/train.sh \
    --branch exp/0403-evolving-tau2 \
    --patch /Users/simonyu/local/local_orby/user_simulator_rl/usim/patches/slime_per_server_engines.patch
```

### tau2 Evolving (dual_cotrain) — needs slime patch + tau2_cotrain_rollout.py
```bash
cd /Users/simonyu/local/local_orby/modal-examples && \
conda run -n modal env MODAL_PROFILE=chatslab modal run --detach \
    simon-exps/spare/slime_train.py \
    --project usim \
    --script cmd/slime/qwen3_4b_instruct/0403/evolving/tau2/train.sh \
    --branch exp/0403-evolving-tau2 \
    --patch /Users/simonyu/local/local_orby/user_simulator_rl/usim/patches/slime_per_server_engines.patch
```

## Priority 3: Still TODO
- [ ] Build t-SNE trajectory evolution analysis script (task #3)
- [ ] VS (single gemini) experiments (lower priority, can run after)
- [ ] Evolving+Checkpoint experiments (after evolving works)

## Monitor Commands
```bash
# Check app status
MODAL_PROFILE=for-uw /Users/simonyu/opt/anaconda3/envs/modal/bin/modal app list
MODAL_PROFILE=chatslab /Users/simonyu/opt/anaconda3/envs/modal/bin/modal app list

# Tail logs (streams forever, use timeout)
MODAL_PROFILE=for-uw timeout 12 /Users/simonyu/opt/anaconda3/envs/modal/bin/modal app logs <APP_ID>
MODAL_PROFILE=chatslab timeout 12 /Users/simonyu/opt/anaconda3/envs/modal/bin/modal app logs <APP_ID>

# Monitor script location
/tmp/monitor_0403_v6.log
```

## Key WandB Groups
- `usim / qwen3-4B-Instruct-2507-p4g-ensemble-0403`
- `usim / qwen3-4B-Instruct-2507-tau2-ensemble-0403`
- `usim / qwen3-4B-Instruct-2507-p4g-evolving-0403`
- `usim / qwen3-4B-Instruct-2507-tau2-evolving-0403`
