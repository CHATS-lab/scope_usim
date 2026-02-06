### Download the models

```bash
WORKSPACE_DIR=/mnt/spare-workspace
# mcore checkpoint
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir ${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507

cd ${WORKSPACE_DIR}/usim_rl/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507 \
    --save ${WORKSPACE_DIR}/Qwen3-4B-Instruct-2507_torch_dist
```