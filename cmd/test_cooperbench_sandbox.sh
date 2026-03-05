#!/bin/bash
# Quick test: verify Modal sandbox creation works for CooperBench
# Run via Modal launcher:
#   modal run simon-exps/spare/slime_train.py --project usim --script cmd/test_cooperbench_sandbox.sh --branch main

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COOPERBENCH_DIR="${PROJECT_ROOT}/external/CooperBench"

echo "=== Installing dependencies ==="
pip install -e "${PROJECT_ROOT}" 2>&1 | tail -3
pip install -e "${COOPERBENCH_DIR}" 2>&1 | tail -5
pip install modal 2>&1 | tail -3

echo "=== Modal auth check ==="
echo "MODAL_TOKEN_ID set: $([ -n "$MODAL_TOKEN_ID" ] && echo yes || echo no)"
echo "MODAL_TOKEN_SECRET set: $([ -n "$MODAL_TOKEN_SECRET" ] && echo yes || echo no)"
modal setup 2>/dev/null || echo "Modal setup skipped"

echo "=== Running single task test ==="
cd "${PROJECT_ROOT}"
python scripts/test_cooperbench_single.py

echo "=== Done ==="
