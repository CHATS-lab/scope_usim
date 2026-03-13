#!/bin/bash

# USIM Perplexity Evaluation — Qwen3-4B-Instruct on P4G Human Test Data
# Evaluates base model and RL-trained checkpoints' perplexity on held-out human conversations.
# This measures sim-to-real alignment: does RL training make the policy diverge from human behavior?
# Date: 2026-03-12

pip install 'transformers>=4.51.0,<5.0.0' 2>/dev/null || true

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-/mnt/spare-workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/usim_slime/0312_perplexity_eval}"

mkdir -p "${OUTPUT_DIR}"

P4G_TEST_DIR="${PROJECT_ROOT}/data/p4g/test"
EVAL_SCRIPT="${PROJECT_ROOT}/cmd/slime/qwen3_4b_instruct/0312/eval_perplexity.py"

# ====================================================================
# STEP 1: Base model perplexity
# ====================================================================
echo "======================================================================"
echo "Evaluating BASE model perplexity on P4G human test data..."
echo "======================================================================"

python3 "${EVAL_SCRIPT}" \
    --model-path "${MODEL_DIR}/Qwen3-4B-Instruct-2507" \
    --data-dir "${P4G_TEST_DIR}" \
    --max-conversations 200 \
    --output "${OUTPUT_DIR}/perplexity_base.json" \
    --label "base" \
    --device "cuda:0"

echo "Base model perplexity eval complete."

# ====================================================================
# STEP 2: RL-Single checkpoint (if available)
# ====================================================================
# Look for P4G single training checkpoints
SINGLE_CKPT_DIR=$(find /scratch/usim_slime -name "Qwen3-4B-Instruct-2507_p4g_single*" -type d 2>/dev/null | head -1)
if [ -z "$SINGLE_CKPT_DIR" ]; then
    # Try alternate naming
    SINGLE_CKPT_DIR=$(find /scratch/usim_slime -path "*/p4g*single*" -name "step_*" -type d 2>/dev/null | sort -V | tail -1)
fi

if [ -n "$SINGLE_CKPT_DIR" ] && [ -d "$SINGLE_CKPT_DIR" ]; then
    echo "======================================================================"
    echo "Found RL-Single checkpoint: ${SINGLE_CKPT_DIR}"
    echo "Evaluating RL-Single perplexity..."
    echo "======================================================================"

    python3 "${EVAL_SCRIPT}" \
        --model-path "${SINGLE_CKPT_DIR}" \
        --data-dir "${P4G_TEST_DIR}" \
        --max-conversations 200 \
        --output "${OUTPUT_DIR}/perplexity_rl_single.json" \
        --label "rl_single" \
        --device "cuda:0"
else
    echo "No RL-Single checkpoint found. Skipping."
fi

# ====================================================================
# STEP 3: SCOPE checkpoint (if available)
# ====================================================================
SCOPE_CKPT_DIR=$(find /scratch/usim_slime -name "Qwen3-4B-Instruct-2507_p4g_scope*" -type d 2>/dev/null | head -1)
if [ -z "$SCOPE_CKPT_DIR" ]; then
    SCOPE_CKPT_DIR=$(find /scratch/usim_slime -path "*/p4g*scope*" -name "step_*" -type d 2>/dev/null | sort -V | tail -1)
fi

if [ -n "$SCOPE_CKPT_DIR" ] && [ -d "$SCOPE_CKPT_DIR" ]; then
    echo "======================================================================"
    echo "Found SCOPE checkpoint: ${SCOPE_CKPT_DIR}"
    echo "Evaluating SCOPE perplexity..."
    echo "======================================================================"

    python3 "${EVAL_SCRIPT}" \
        --model-path "${SCOPE_CKPT_DIR}" \
        --data-dir "${P4G_TEST_DIR}" \
        --max-conversations 200 \
        --output "${OUTPUT_DIR}/perplexity_scope.json" \
        --label "scope" \
        --device "cuda:0"
else
    echo "No SCOPE checkpoint found. Skipping."
fi

# ====================================================================
# STEP 4: RL-Configured checkpoint (if available)
# ====================================================================
CONFIGURED_CKPT_DIR=$(find /scratch/usim_slime -path "*/p4g*configured*" -name "step_*" -type d 2>/dev/null | sort -V | tail -1)

if [ -n "$CONFIGURED_CKPT_DIR" ] && [ -d "$CONFIGURED_CKPT_DIR" ]; then
    echo "======================================================================"
    echo "Found RL-Configured checkpoint: ${CONFIGURED_CKPT_DIR}"
    echo "Evaluating RL-Configured perplexity..."
    echo "======================================================================"

    python3 "${EVAL_SCRIPT}" \
        --model-path "${CONFIGURED_CKPT_DIR}" \
        --data-dir "${P4G_TEST_DIR}" \
        --max-conversations 200 \
        --output "${OUTPUT_DIR}/perplexity_configured.json" \
        --label "rl_configured" \
        --device "cuda:0"
else
    echo "No RL-Configured checkpoint found. Skipping."
fi

echo "======================================================================"
echo "Perplexity evaluation complete. Results in ${OUTPUT_DIR}/"
echo "======================================================================"

# Print summary
echo ""
echo "=== SUMMARY ==="
for f in "${OUTPUT_DIR}"/perplexity_*.json; do
    if [ -f "$f" ]; then
        label=$(python3 -c "import json; print(json.load(open('$f'))['label'])")
        ppl=$(python3 -c "import json; print(f\"{json.load(open('$f'))['perplexity']:.2f}\")")
        tokens=$(python3 -c "import json; print(json.load(open('$f'))['total_tokens'])")
        echo "  ${label}: perplexity=${ppl} (${tokens} tokens)"
    fi
done
