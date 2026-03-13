#!/bin/bash

# P4G Self-Play Script — Qwen3-4B-Instruct (Mode 3: self-play + checkpoint pool)
# Single training group trains on both agent + opponent trajectory samples.
# Actor SGLang engine: NCCL weight sync. Opponent SGLang engine: swapped from pool.
# Pool of historical checkpoints provides opponent diversity during rollout.
# GPU layout: [training:2][actor_engines:3][opponent_engines:3] = 8 GPUs

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3

set -ex

export PYTHONUNBUFFERED=1
export WEAVE_PRINT_CALL_LINK=false

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/selfplay_p4g/$(date +%Y%m%d_%H%M%S)}"
# Model checkpoint lives on spare-workspace (shared across projects)
MODEL_DIR="${MODEL_DIR:-/mnt/spare-workspace}"

mkdir -p "${OUTPUT_DIR}"

# Source model configuration
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B-Instruct-2507"
   --ref-load "${MODEL_DIR}/Qwen3-4B-Instruct-2507_torch_dist"
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_selfplay_p4g/"
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

# Self-play arguments
SELFPLAY_ARGS=(
   --training-mode selfplay
   --trainable-role agent
   --max-turns 10
   --pool-dir "${OUTPUT_DIR}/checkpoint_pool"
   --pool-size 10
   --pool-save-interval 16
   --pool-selection random
)

# P4G-specific arguments
P4G_ARGS=(
   --p4g-corpus-path "${PROJECT_ROOT}/data/p4g/corpus"
   --p4g-dataset-dir "${PROJECT_ROOT}/data/p4g/train"
   --p4g-word-limit 50
   --p4g-num-turns 10
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project usim
   --wandb-team simon011130
   --wandb-group qwen3-4B-Instruct-2507-p4g-selfplay
   --wandb-key ${WANDB_API_KEY:-""}
)

# Eval config
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_config.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/p4g_6model.yaml" > "${EVAL_CONFIG_FILE}"

EVAL_ARGS=(
   --eval-interval 16
   --skip-eval-before-train
   --eval-config "${EVAL_CONFIG_FILE}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# sglang-config with actor + opponent named models
SGLANG_ARGS=(
   --sglang-config "${PROJECT_ROOT}/configs/sglang/cotrain_2plus2.yaml"
   --rollout-num-gpus 6
   --rollout-num-gpus-per-engine 1
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
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   --actor-num-gpus-per-node 2 \
   --save-hf "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_selfplay_p4g_hf/" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${SELFPLAY_ARGS[@]} \
   ${P4G_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}
