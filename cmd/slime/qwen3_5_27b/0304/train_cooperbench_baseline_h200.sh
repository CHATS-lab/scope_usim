#!/bin/bash

# CooperBench Baseline Training Script — Qwen3.5-27B on H200 (140GB)
# Agent (trainable): Qwen3.5-27B via SGLang
# Setting: baseline (1 agent, 1 feature)
# Date: 2025-03-04
#
# H200 vs H100 key differences (140GB vs 80GB per GPU):
#   - rollout-max-context-len: 65536 (vs 16384) — agents can finish tasks
#   - max-tokens-per-gpu: 4096 (vs 1024) — better training GPU utilization
#   - pipeline-model-parallel-size: 1 (vs 2) — no pipeline bubble overhead
#   - sglang-mem-fraction-static: 0.65 (vs 0.40) — larger KV cache, faster rollouts
#   - speculative decoding: enabled (NEXTN) — faster token generation
#
# OOM Debugging Guide (if OOM occurs, apply fixes in this order):
#
#   1. Reduce --sglang-mem-fraction-static (0.65 -> 0.55 -> 0.45)
#      SGLang KV cache competes with Megatron for GPU memory during training.
#      torch_memory_saver releases KV cache but model weights (~6.5 GiB/GPU at TP=8) remain.
#
#   2. Reduce --max-tokens-per-gpu (4096 -> 2048 -> 1024)
#      Controls micro-batch size during training. Logits tensor = tokens * (vocab_size/TP) * 4 bytes.
#      With vocab_size=248320 and TP=4: 4096 tokens → ~1 GiB logits. At 65k ctx this matters.
#
#   3. Switch to --pipeline-model-parallel-size 2 --context-parallel-size 1
#      PP=2 halves per-GPU params (10.9B -> 5.4B), cutting gradient buffer from ~40 GiB to ~20 GiB.
#      Trade-off: introduces pipeline bubble overhead (~15% slower).
#
#   4. Reduce --rollout-max-context-len (65536 -> 32768 -> 16384)
#      LAST RESORT. This is the most impactful setting for task completion.
#      The training forward pass creates logits tensors proportional to sequence length:
#        - 16k ctx: ~4 GiB peak logits    (safe on H100 80GB with PP=2)
#        - 32k ctx: ~8 GiB peak logits    (safe on H200 140GB)
#        - 65k ctx: ~14 GiB peak logits   (needs H200 140GB with PP=1)
#      OOM typically manifests as "logits.clone()" or "vocab_parallel_logits - logits_max" failure.
#
#   5. Disable speculative decoding (remove NEXTN args)
#      Speculative decoding allocates extra memory for draft model heads.
#      If SGLang reports "Not enough memory", try removing speculative args first.
#
#   6. Nuclear option: --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --context-parallel-size 2
#      Only if all above fail. Significantly reduces parallelism efficiency.
#
# Known issue: Qwen3.5 chat template requires tool_call arguments as dict, not JSON string.
#   Fixed in usim/core/types.py Message.to_dict() — uses json.loads() for string arguments.

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3

set -e

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

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/0304_cooperbench_baseline_27b_h200/$(date +%Y%m%d_%H%M%S)}"
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
# === Patch sglang for Qwen3.5 tool calling fixes ===
SGLANG_DIR="/sgl-workspace/sglang"
if [ -d "${SGLANG_DIR}" ]; then
    echo "Patching sglang for Qwen3.5..."
    pushd "${SGLANG_DIR}"
    git pull
    git fetch https://github.com/sgl-project/sglang.git main
    git cherry-pick --no-commit d38c0e537 b3202fe6d bdc1e46e5
    popd
    echo "sglang patched OK"
else
    echo "WARNING: ${SGLANG_DIR} not found, skipping sglang patch"
fi

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
   --rollout-max-response-len 16384
   --rollout-temperature 0.7
   --rollout-max-context-len 65536
   --global-batch-size 32
   --balance-data
)

# CooperBench baseline: 1 agent, 1 feature, no partner
COOPERBENCH_ARGS=(
   --trainable-role agent
   --cooperbench-setting baseline
   --cooperbench-data-dir "${PROJECT_ROOT}/data/cooperbench"
   --cooperbench-backend modal
   --cooperbench-max-steps 100
   --cooperbench-dataset-dir "${COOPERBENCH_DIR}"
   --cooperbench-max-concurrent 32
   --cooperbench-max-tokens-per-turn 4096
   --cooperbench-max-tool-output-chars 4000
   --cooperbench-patch-bonus 0.1
   --cooperbench-tool-call-parser qwen3_coder
)

# H200: PP=2, no CPU optimizer offload. Keeps Adam on GPU (~47 GiB/GPU) with
# room for 65k logits (~16 GiB). Total ~82 GiB/GPU — comfortable on 140 GiB.
# PP=1 would push to ~129 GiB/GPU (only 11 GiB headroom — fragmentation risk).
# No CPU offload avoids the 256 GiB RAM OOM (PP=1 offload needs ~432 GiB).
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
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
   --wandb-project "${WANDB_PROJECT:-scope}"
   --wandb-group "qwen3.5-27B-cooperbench-baseline-h200-0304"
)

EVAL_ARGS=()

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
)

# H200: larger KV cache (0.65) + speculative decoding for faster rollouts
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.65
   --sglang-cuda-graph-bs 1 2 4 8 16

   --sglang-max-running-requests 512

   # Speculative decoding — remove these 4 lines if SGLang reports "Not enough memory"
   --sglang-speculative-algorithm NEXTN
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
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

# === Redis (needed for coop messaging) ===
if command -v redis-server &>/dev/null; then
    redis-server --daemonize yes --bind 127.0.0.1 --port 6379
    echo "Redis started on localhost:6379"
else
    echo "WARNING: redis-server not found, coop mode will not work"
fi

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
