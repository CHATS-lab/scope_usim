#!/bin/bash

# CooperBench Baseline Training Script — Qwen3.5-27B
# Agent (trainable): Qwen3.5-27B via SGLang
# Setting: baseline (1 agent, 1 feature)
# Date: 2025-03-04

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

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/0304_cooperbench_baseline_27b/$(date +%Y%m%d_%H%M%S)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"

mkdir -p "${OUTPUT_DIR}"

# === Install CooperBench + usim dependencies ===
# Order matters: install cooperbench first (heavy deps), then reinstall slime last
# to fix any dependency clobbering (especially transformers version).
echo "Installing usim..."
pip install -e "${PROJECT_ROOT}" 2>&1 | tail -3
echo "Installing CooperBench..."
pip install -e "${COOPERBENCH_DIR}" 2>&1 | tail -5
echo "Re-installing slime + fixing deps..."
pip install -e "${SLIME_DIR}" 2>&1 | tail -3
# Qwen3.5 needs latest transformers; pin openai for sglang; restore cudnn 9.16; modal for sandboxes
pip install --upgrade transformers openai==2.6.1 nvidia-cudnn-cu12==9.16.0.29 modal 2>&1 | tail -3
# Verify slime is importable
python3 -c "from slime.train_async import parse_args, train; print('slime import OK')" || { echo "FATAL: slime import failed"; exit 1; }

# === Dataset setup ===
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

# Source Qwen3.5-27B model configuration
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${WORKSPACE_DIR}/Qwen3.5-27B"
   --ref-load "${WORKSPACE_DIR}/Qwen3.5-27B_torch_dist"
   --save "${OUTPUT_DIR}/Qwen3.5-27B_cooperbench_baseline/"
   --save-interval 32
)

ROLLOUT_ARGS=(
   --data-source-path usim.cooperbench.data_source.get_cooperbench_data_source
   --rollout-function-path usim.cooperbench.rollout.cooperbench_generate_rollout
   --num-rollout 500
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 8192
   --rollout-temperature 0.7
   --global-batch-size 32
   --balance-data
)

# CooperBench baseline: 1 agent, 1 feature, no partner
COOPERBENCH_ARGS=(
   --trainable-role agent
   --cooperbench-setting baseline
   --cooperbench-data-dir "${PROJECT_ROOT}/data/cooperbench"
   --cooperbench-backend modal
   --cooperbench-max-steps 50
   --cooperbench-dataset-dir "${COOPERBENCH_DIR}"
   --cooperbench-max-concurrent 16
   --cooperbench-max-context-length 32768
   --cooperbench-max-tokens-per-turn 4096
   --cooperbench-max-tool-output-chars 4000
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
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
   --wandb-group "qwen3.5-27B-cooperbench-baseline-0304"
   --wandb-key ${WANDB_API_KEY:-""}
)

EVAL_ARGS=()

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 16
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# === Modal auth ===
# Inside a Modal container, auth happens via unix:/run/modal.sock automatically.
# Just export tokens so they propagate to Ray workers for nested sandbox creation.
MODAL_TOKEN_ID="${MODAL_TOKEN_ID:-}"
MODAL_TOKEN_SECRET="${MODAL_TOKEN_SECRET:-}"
export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
echo "MODAL_TOKEN_ID set: $([ -n "$MODAL_TOKEN_ID" ] && echo yes || echo no)"
echo "MODAL_TOKEN_SECRET set: $([ -n "$MODAL_TOKEN_SECRET" ] && echo yes || echo no)"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MODAL_TOKEN_ID\": \"${MODAL_TOKEN_ID}\",
    \"MODAL_TOKEN_SECRET\": \"${MODAL_TOKEN_SECRET}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m train_cooperbench_slime \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
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
