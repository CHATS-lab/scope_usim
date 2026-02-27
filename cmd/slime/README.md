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

---

## GLM-4.7-30B-A3B (Tillicum / SLURM + Apptainer)

GLM-4.7-30B-A3B is a MoE model (30B total, ~3B active) using multi-latent attention. Run all
steps below **inside the container** on a compute node.

### 1. Get an interactive node

```bash
# From login node — allocates 8x H200 for up to 24h
sbatch tillicum/sbatch/glm4.7_30b_a3b/sleep.sbatch

# SSH into the allocated node once the job starts
ssh $(squeue -j <JOBID> -h -o "%N")
```

### 2. Enter the container

```bash
SCRATCH=/gpfs/scrubbed/beneecs
PROJECT_ROOT=/gpfs/projects/socialrl/bo/work/projects/spare/github_repo/usim_rl

apptainer shell --nv \
  --bind ${SCRATCH}:/scratch \
  --bind ${PROJECT_ROOT}:/workspace \
  ${SCRATCH}/containers/slime_latest.sif
```

Inside the container, set up paths:

```bash
WORKSPACE_DIR=/scratch/spare-workspace   # ${SCRATCH} maps to /scratch inside container
SLIME_DIR=/workspace/slime
```

### 3. Download the dataset (tau2-bench)

tau2-bench data is provided via the `tau2` package — no separate download needed.
Install with:

```bash
cd /workspace
pip install -e ".[slime,tau2]" --no-deps
pip install openai
```

### 4. Download the model

```bash
# HF model → local directory
hf download THUDM/GLM-4.7-30B-A3B --local-dir ${WORKSPACE_DIR}/GLM-4.7-30B-A3B
```

> **Note**: verify the HF repo name at https://huggingface.co/THUDM — it may include a date
> suffix (e.g. `GLM-4.7-30B-A3B-0716`). Update the paths below accordingly.

### 5. Convert to Megatron torch_dist format

The model script is already in slime at `scripts/models/glm4.7-30B-A3B.sh`.

```bash
cd ${SLIME_DIR}
source scripts/models/glm4.7-30B-A3B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${WORKSPACE_DIR}/GLM-4.7-30B-A3B \
    --save ${WORKSPACE_DIR}/GLM-4.7-30B-A3B_torch_dist
```

For faster conversion with multiple GPUs:

```bash
PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${WORKSPACE_DIR}/GLM-4.7-30B-A3B \
    --save ${WORKSPACE_DIR}/GLM-4.7-30B-A3B_torch_dist
```

The converted checkpoint will be at `${WORKSPACE_DIR}/GLM-4.7-30B-A3B_torch_dist`.