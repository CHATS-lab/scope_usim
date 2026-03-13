#!/bin/bash

# Comprehensive perplexity evaluation on all available HF checkpoints
# Finds checkpoints from P4G single and SCOPE perplexity training runs
# Evaluates base model + all step checkpoints found
# Date: 2026-03-12

pip install 'transformers>=4.51.0,<5.0.0' 2>/dev/null || true

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-/mnt/spare-workspace}"
PPL_OUTPUT_DIR="/scratch/usim_slime/0312_perplexity_eval"
P4G_TEST_DIR="${PROJECT_ROOT}/data/p4g/test"
EVAL_SCRIPT="${PROJECT_ROOT}/cmd/slime/qwen3_4b_instruct/0312/eval_perplexity.py"

mkdir -p "${PPL_OUTPUT_DIR}"

# ====================================================================
# STEP 1: Base model perplexity
# ====================================================================
echo "======================================================================"
echo "Evaluating BASE model perplexity..."
echo "======================================================================"

python3 "${EVAL_SCRIPT}" \
    --model-path "${MODEL_DIR}/Qwen3-4B-Instruct-2507" \
    --data-dir "${P4G_TEST_DIR}" \
    --max-conversations 200 \
    --output "${PPL_OUTPUT_DIR}/perplexity_base.json" \
    --label "base" \
    --device "cuda:0"

# ====================================================================
# STEP 2: RL-Single HF checkpoints
# ====================================================================
echo "======================================================================"
echo "Searching for RL-Single HF checkpoints..."
echo "======================================================================"

# Find the P4G single perplexity training output dir
SINGLE_PPL_BASE=$(find /scratch/usim_slime/0312_p4g_single_ppl -name "hf_checkpoints" -type d 2>/dev/null | head -1)

if [ -n "$SINGLE_PPL_BASE" ] && [ -d "$SINGLE_PPL_BASE" ]; then
    echo "Found RL-Single HF checkpoints at: ${SINGLE_PPL_BASE}"
    for ckpt_dir in $(ls -d ${SINGLE_PPL_BASE}/step_* 2>/dev/null | sort -V); do
        step_name=$(basename "${ckpt_dir}")
        echo "Evaluating RL-Single ${step_name} perplexity..."
        python3 "${EVAL_SCRIPT}" \
            --model-path "${ckpt_dir}" \
            --data-dir "${P4G_TEST_DIR}" \
            --max-conversations 200 \
            --output "${PPL_OUTPUT_DIR}/perplexity_rl_single_${step_name}.json" \
            --label "rl_single_${step_name}" \
            --device "cuda:0"
    done
else
    echo "No RL-Single HF checkpoints found under /scratch/usim_slime/0312_p4g_single_ppl/"
fi

# ====================================================================
# STEP 3: SCOPE HF checkpoints (if available)
# ====================================================================
echo "======================================================================"
echo "Searching for SCOPE HF checkpoints..."
echo "======================================================================"

SCOPE_PPL_BASE=$(find /scratch/usim_slime/0312_p4g_scope_ppl -name "hf_checkpoints" -type d 2>/dev/null | head -1)

if [ -n "$SCOPE_PPL_BASE" ] && [ -d "$SCOPE_PPL_BASE" ]; then
    echo "Found SCOPE HF checkpoints at: ${SCOPE_PPL_BASE}"
    for ckpt_dir in $(ls -d ${SCOPE_PPL_BASE}/step_* 2>/dev/null | sort -V); do
        step_name=$(basename "${ckpt_dir}")
        echo "Evaluating SCOPE ${step_name} perplexity..."
        python3 "${EVAL_SCRIPT}" \
            --model-path "${ckpt_dir}" \
            --data-dir "${P4G_TEST_DIR}" \
            --max-conversations 200 \
            --output "${PPL_OUTPUT_DIR}/perplexity_scope_${step_name}.json" \
            --label "scope_${step_name}" \
            --device "cuda:0"
    done
else
    echo "No SCOPE HF checkpoints found under /scratch/usim_slime/0312_p4g_scope_ppl/"
fi

# ====================================================================
# SUMMARY
# ====================================================================
echo ""
echo "======================================================================"
echo "=== PERPLEXITY RESULTS ==="
echo "======================================================================"
for f in $(ls ${PPL_OUTPUT_DIR}/perplexity_*.json 2>/dev/null | sort); do
    if [ -f "$f" ]; then
        label=$(python3 -c "import json; print(json.load(open('$f'))['label'])")
        ppl=$(python3 -c "import json; print(f\"{json.load(open('$f'))['perplexity']:.2f}\")")
        tokens=$(python3 -c "import json; print(json.load(open('$f'))['total_tokens'])")
        echo "  ${label}: perplexity=${ppl} (${tokens} tokens)"
    fi
done
