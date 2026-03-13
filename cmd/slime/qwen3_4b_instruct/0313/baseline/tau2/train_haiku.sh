#!/bin/bash

# USIM Training Script — Qwen3-4B-Instruct on tau2-bench (retail)
# Agent (trainable): Qwen3-4B-Instruct-2507 via SGLang
# User sim (fixed): anthropic/claude-haiku-4.5 via OpenRouter
# Date: 2026-03-13 (0313 baseline)

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3

set -ex

export PYTHONBUFFERED=1
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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/0313_tau2_haiku/$(date +%Y%m%d_%H%M%S)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"

mkdir -p "${OUTPUT_DIR}"

# Source model configuration (Instruct, rotary_base=5000000)
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint "${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507"
   --ref-load "${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507_torch_dist"
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_usim_tau2/"
   --save-interval 32
)

ROLLOUT_ARGS=(
   --data-source-path usim.slime.data_source.get_tau2_data_source
   --rollout-function-path usim.slime.rollout.usim_generate_rollout
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 0.7
   --global-batch-size 128
   --balance-data
)

# USIM-specific arguments
# Agent (trainable) = Qwen3-4B-Instruct via SGLang
# User sim (fixed opponent) = anthropic/claude-haiku-4.5 via OpenRouter
USIM_ARGS=(
   --trainable-role agent
   --max-turns 30
   --usim-domain retail
   --usim-fixed-opponent-model "anthropic/claude-haiku-4.5"
   --usim-fixed-opponent-base-url "https://openrouter.ai/api/v1"
   --usim-fixed-opponent-api-key-var "OPENROUTER_API_KEY"
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
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
   --wandb-group qwen3-4B-Instruct-2507-tau2-haiku-0313
   --wandb-key ${WANDB_API_KEY:-""}
)

# Eval config (template in eval_configs/, resolved at runtime)
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_config.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/tau2_retail_6model.yaml" > "${EVAL_CONFIG_FILE}"

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

SGLANG_ARGS=(
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
   -- python3 -m train_usim_slime \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${USIM_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}
