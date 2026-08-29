#!/bin/bash

# P4G adversarial co-training — Qwen3-4B-Instruct vs Qwen3-4B-Instruct
# Agent (persuader): Qwen3-4B-Instruct-2507, reward = donation / 2.0
# User (persuadee):  Qwen3-4B-Instruct-2507, reward = 1 - donation / 2.0
# Both trainable, small KL penalty on each side to prevent drift.
# Date: 2026-04-17

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3

set -e

export PYTHONUNBUFFERED=1
export WEAVE_PRINT_CALL_LINK=false
# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set in 0313 scripts)
# is incompatible with SGLang's TorchMemorySaver used by
# --offload-rollout / --offload-train. 0403 scripts don't set it either.
# Leave unset.

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/0417_p4g_cotrain/$(date +%Y%m%d_%H%M%S)}"
# Modal mounts both volumes: /mnt/spare-workspace (Qwen3-4B model checkpoints)
# and /mnt/usim-workspace (repo + outputs). The launcher overrides
# WORKSPACE_DIR to the usim mount; MODEL_DIR stays on spare since that's
# where the actual HF + torch_dist checkpoints live.
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"
MODEL_DIR="${MODEL_DIR:-/mnt/spare-workspace}"

mkdir -p "${OUTPUT_DIR}"

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B-Instruct-2507"
   --ref-load "${MODEL_DIR}/Qwen3-4B-Instruct-2507_torch_dist"
   --save "${OUTPUT_DIR}/qwen3-4B-Instruct-2507_p4g_cotrain/"
   --save-interval 32
)

ROLLOUT_ARGS=(
   --data-source-path usim.p4g.data_source.get_p4g_data_source
   --rollout-function-path usim.slime.cotrain_rollout.cotrain_generate_rollout
   --num-rollout 500
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 0.7
   --global-batch-size 128
   --balance-data
)

# Co-train settings: train both roles, adversarial reward, no --cooperative-reward.
COTRAIN_ARGS=(
   --training-mode dual_cotrain
   --trainable-role both
   --max-turns 10
)

# P4G environment
P4G_ARGS=(
   --p4g-corpus-path "${PROJECT_ROOT}/data/p4g/corpus"
   --p4g-dataset-dir "${PROJECT_ROOT}/data/p4g/train"
   --p4g-word-limit 50
   --p4g-num-turns 10
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --log-probs-chunk-size 2048
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

# GRPO + small KL on both sides (matches SPICE's mixed setup: std-norm off so
# curriculum-scale signals don't get washed; KL on to keep both policies
# anchored to the reference Qwen3-4B-Instruct base).
GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.005
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT:-scope}"
   --wandb-group qwen3-4b-p4g-cotrain-0417
)

# 5-model OpenRouter eval every 16 steps.
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_config.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/p4g_6model.yaml" > "${EVAL_CONFIG_FILE}"

EVAL_ARGS=(
   --eval-interval 16
   --skip-eval-before-train
   --eval-config "${EVAL_CONFIG_FILE}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --sglang-config "${PROJECT_ROOT}/configs/sglang/cotrain_4plus4.yaml"
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m train_cotrain_slime \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   --offload-rollout \
   --offload-train \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${COTRAIN_ARGS[@]} \
   ${P4G_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}
