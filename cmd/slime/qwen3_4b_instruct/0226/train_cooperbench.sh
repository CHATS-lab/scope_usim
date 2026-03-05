#!/bin/bash

# CooperBench Training Script — Qwen3-4B-Instruct on CooperBench
# Agent (trainable): Qwen3-4B-Instruct-2507 via SGLang
# Partner (fixed, coop only): gpt-5-mini via CooperBench mini_swe_agent
# Settings: baseline, solo (default), coop
# Date: 2025-02-26

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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"
COOPERBENCH_DIR="${PROJECT_ROOT}/external/CooperBench"

# Setting: baseline, solo, or coop (override via COOPERBENCH_SETTING env var)
SETTING="${COOPERBENCH_SETTING:-solo}"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/0226_cooperbench_${SETTING}/$(date +%Y%m%d_%H%M%S)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"

mkdir -p "${OUTPUT_DIR}"

# === Dataset setup ===
# Pull CooperBench dataset from HuggingFace if not already present
if [ ! -d "${COOPERBENCH_DIR}/dataset" ]; then
    echo "Downloading CooperBench dataset..."
    pip install huggingface_hub 2>/dev/null
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='CodeConflict/cooperbench-dataset',
    repo_type='dataset',
    local_dir='${COOPERBENCH_DIR}/dataset',
)
print('Dataset downloaded successfully')
"
fi

# Source model configuration (Instruct, rotary_base=5000000)
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint "${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507"
   --ref-load "${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507_torch_dist"
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_cooperbench_${SETTING}/"
   --save-interval 32
)

ROLLOUT_ARGS=(
   --data-source-path usim.cooperbench.data_source.get_cooperbench_data_source
   --rollout-function-path usim.cooperbench.rollout.cooperbench_generate_rollout
   --num-rollout 500
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.7
   --global-batch-size 128
   --balance-data
)

# CooperBench-specific arguments
COOPERBENCH_ARGS=(
   --trainable-role agent
   --cooperbench-setting "${SETTING}"
   --cooperbench-data-dir "${PROJECT_ROOT}/data/cooperbench"
   --cooperbench-backend modal
   --cooperbench-max-steps 50
   --cooperbench-dataset-dir "${COOPERBENCH_DIR}"
)

# Add coop-specific args
if [ "${SETTING}" = "coop" ]; then
    COOPERBENCH_ARGS+=(
       --cooperbench-partner-model "gpt-5-mini"
       --cooperbench-redis-url "redis://localhost:6379"
       --cooperbench-partial-reward
    )
fi

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --grpo-std-normalization
   --rewards-normalization
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
   --wandb-group "qwen3-4B-Instruct-2507-cooperbench-${SETTING}-0226"
   --wandb-key ${WANDB_API_KEY:-""}
)

EVAL_ARGS=()

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

# === Redis server (needed for coop mode) ===
if [ "${SETTING}" = "coop" ]; then
    if ! redis-cli ping > /dev/null 2>&1; then
        echo "Starting Redis server..."
        redis-server --daemonize yes
        sleep 1
    fi
fi

# === Modal setup ===
modal setup 2>/dev/null || echo "Modal already configured or not available"

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
   -- python3 -m train_cooperbench_slime \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 6 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${COOPERBENCH_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}
