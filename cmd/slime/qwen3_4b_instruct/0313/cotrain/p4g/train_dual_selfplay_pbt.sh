#!/bin/bash

# Population-Based Self-Play Training Script — Qwen3-4B-Instruct on Persuasion for Good
# Population-based dual self-play — large checkpoint pool for opponent diversity
# Agent (trainable): Qwen3-4B-Instruct-2507 via SGLang (persuader)
# Opponent (pool): Large population of historical checkpoints (20), saved every 8 rollouts (persuadee)
# Date: 2026-03-13 (0313 dual selfplay pbt)

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3

set -ex

export PYTHONUNBUFFERED=1
export WEAVE_PRINT_CALL_LINK=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/0313_p4g_dual_selfplay_pbt/$(date +%Y%m%d_%H%M%S)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"

mkdir -p "${OUTPUT_DIR}"

# Source model configuration (Instruct, rotary_base=5000000)
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint "${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507"
   --ref-load "${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507_torch_dist"
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_dual_selfplay_pbt_p4g/"
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

# Cotrain-specific arguments (dual selfplay with large checkpoint pool)
COTRAIN_ARGS=(
   --training-mode dual_selfplay
   --trainable-role agent
   --max-turns 10
   --pool-dir "${OUTPUT_DIR}/checkpoint_pool"
   --pool-size 20
   --pool-save-interval 8
   --pool-selection random
)

# P4G-specific arguments
P4G_ARGS=(
   --p4g-corpus-path "${PROJECT_ROOT}/data/p4g/corpus"
   --p4g-dataset-dir "${PROJECT_ROOT}/data/p4g/train"
   --p4g-word-limit 50
   --p4g-num-turns 10
)

# Colocated 4+4 perf config
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --log-probs-chunk-size 2048
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
   --wandb-group qwen3-4B-Instruct-2507-p4g-dual-selfplay-pbt-0313
   --wandb-key ${WANDB_API_KEY:-""}
)

# Eval config (template in eval_configs/, resolved at runtime)
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
   --actor-num-gpus-per-node 4 \
   --colocate \
   --offload-rollout \
   --offload-train \
   --save-hf "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_dual_selfplay_pbt_p4g_hf/" \
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
