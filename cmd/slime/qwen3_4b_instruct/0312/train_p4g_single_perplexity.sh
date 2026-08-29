#!/bin/bash

# P4G RL-Single with frequent HF checkpoint saving for perplexity experiment
# Agent (trainable): Qwen3-4B-Instruct-2507 via SGLang (persuader)
# User sim (fixed): gpt-5-mini via OpenAI API (persuadee)
# Purpose: Save HF-format checkpoints every 4 steps for perplexity analysis
# Date: 2026-03-12

pip install 'transformers>=4.51.0,<5.0.0' 2>/dev/null || true

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3

set -e

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

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/0312_p4g_single_ppl/$(date +%Y%m%d_%H%M%S)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spare-workspace}"
MODEL_DIR="${MODEL_DIR:-/mnt/spare-workspace}"
PPL_OUTPUT_DIR="/scratch/usim_slime/0312_perplexity_eval"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PPL_OUTPUT_DIR}"

# Source model configuration
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B-Instruct-2507"
   --ref-load "${MODEL_DIR}/Qwen3-4B-Instruct-2507_torch_dist"
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_usim_p4g/"
   --save-interval 4
   --save-hf "${OUTPUT_DIR}/hf_checkpoints/step_{rollout_id}"
)

ROLLOUT_ARGS=(
   --data-source-path usim.p4g.data_source.get_p4g_data_source
   --rollout-function-path usim.p4g.rollout.p4g_generate_rollout
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 0.7
   --global-batch-size 128
   --balance-data
)

P4G_ARGS=(
   --trainable-role agent
   --max-turns 10
   --usim-fixed-opponent-model "gpt-5-mini"
   --usim-fixed-opponent-base-url "https://api.openai.com/v1"
   --usim-fixed-opponent-api-key-var "OPENAI_API_KEY"
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
   --wandb-group qwen3-4B-Instruct-2507-p4g-single-ppl-0312
)

# Eval config required even if we won't run eval (assertion in code)
# Set interval to 9999 so eval never fires during 16-step training
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_config.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/p4g_6model.yaml" > "${EVAL_CONFIG_FILE}"

EVAL_ARGS=(
   --eval-interval 9999
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

# ====================================================================
# STEP 1: Train P4G RL-Single with frequent HF checkpoint saves
# ====================================================================
echo "======================================================================"
echo "Training P4G RL-Single with HF checkpoints every 4 steps..."
echo "======================================================================"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m train_p4g_slime \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 6 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${P4G_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}

echo "Training complete."

# ====================================================================
# STEP 2: Run perplexity eval on base + each HF checkpoint
# ====================================================================
echo "======================================================================"
echo "Running perplexity evaluation on checkpoints..."
echo "======================================================================"

# Clean up training processes
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 5

P4G_TEST_DIR="${PROJECT_ROOT}/data/p4g/test"
EVAL_SCRIPT="${PROJECT_ROOT}/cmd/slime/qwen3_4b_instruct/0312/eval_perplexity.py"

# Base model
echo "Evaluating base model perplexity..."
python3 "${EVAL_SCRIPT}" \
    --model-path "${MODEL_DIR}/Qwen3-4B-Instruct-2507" \
    --data-dir "${P4G_TEST_DIR}" \
    --max-conversations 200 \
    --output "${PPL_OUTPUT_DIR}/perplexity_base.json" \
    --label "base" \
    --device "cuda:0"

# Each HF checkpoint
HF_CKPT_BASE="${OUTPUT_DIR}/hf_checkpoints"
if [ -d "${HF_CKPT_BASE}" ]; then
    for ckpt_dir in $(ls -d ${HF_CKPT_BASE}/step_* 2>/dev/null | sort -V); do
        step_name=$(basename "${ckpt_dir}")
        echo "Evaluating ${step_name} perplexity..."
        python3 "${EVAL_SCRIPT}" \
            --model-path "${ckpt_dir}" \
            --data-dir "${P4G_TEST_DIR}" \
            --max-conversations 200 \
            --output "${PPL_OUTPUT_DIR}/perplexity_rl_single_${step_name}.json" \
            --label "rl_single_${step_name}" \
            --device "cuda:0"
    done
else
    echo "No HF checkpoints found at ${HF_CKPT_BASE}"
fi

echo "======================================================================"
echo "Perplexity evaluation complete."
echo "======================================================================"

# Print summary
echo ""
echo "=== PERPLEXITY RESULTS ==="
for f in $(ls ${PPL_OUTPUT_DIR}/perplexity_*.json 2>/dev/null | sort); do
    if [ -f "$f" ]; then
        label=$(python3 -c "import json; print(json.load(open('$f'))['label'])")
        ppl=$(python3 -c "import json; print(f\"{json.load(open('$f'))['perplexity']:.2f}\")")
        tokens=$(python3 -c "import json; print(json.load(open('$f'))['total_tokens'])")
        echo "  ${label}: perplexity=${ppl} (${tokens} tokens)"
    fi
done
