# SCOPE experiment launchers

This directory contains the Slime launchers used to train and evaluate SCOPE. The table below is the camera-ready entry point; older dated configurations remain available as an archival record of development runs.

> [!WARNING]
> The training launchers are intended for a dedicated GPU node. They stop existing Ray and SGLang processes before starting a run and, unless edited, reserve all eight visible GPUs.

## Prerequisites

- Linux with eight CUDA GPUs for the released configurations
- Python 3.10 or newer
- Megatron-LM at `/root/Megatron-LM`, or an equivalent path reflected in the launcher's `RUNTIME_ENV_JSON`
- the initialized `slime` and benchmark submodules
- Qwen Hugging Face and Megatron `torch_dist` checkpoints
- API keys for the fixed simulators and held-out evaluation panel

From the repository root:

```bash
git submodule update --init --recursive
pip install -e ".[slime,tau2]"
pip install -e ./slime
pip install -e ./external/tau2-bench
pip install openai convokit
```

The launchers source model arguments from `slime/scripts/models/`. For Qwen3-4B-Instruct-2507, place these directories under `MODEL_DIR`:

```text
Qwen3-4B-Instruct-2507/
Qwen3-4B-Instruct-2507_torch_dist/
```

Use Slime's conversion tools if only the Hugging Face checkpoint is available.

## Environment

Set paths and credentials explicitly before launching:

```bash
export MODEL_DIR="/path/to/model-checkpoints"
export WORKSPACE_DIR="/path/to/workspace"
export OUTPUT_DIR="/path/to/run-output"

export OPENAI_API_KEY="<your OpenAI API key>"
export OPENROUTER_API_KEY="<your OpenRouter API key>"

export WANDB_PROJECT="scope"
export WANDB_ENTITY="<optional W&B team or username>"
# Use this instead when logging should remain local:
# export WANDB_MODE="offline"
```

| Variable | Purpose | Launcher default |
| --- | --- | --- |
| `MODEL_DIR` | Hugging Face and `torch_dist` model checkpoints | `/mnt/spare-workspace` |
| `WORKSPACE_DIR` | shared working storage used by some launchers | `/mnt/spare-workspace` |
| `OUTPUT_DIR` | checkpoints, generated eval configs, and run logs | a timestamped `/scratch/usim_slime/...` directory |
| `OPENAI_API_KEY` | GPT simulator calls | none |
| `OPENROUTER_API_KEY` | non-OpenAI simulators and held-out evaluation | none |
| `WANDB_PROJECT` | W&B project name | `scope` |
| `WANDB_ENTITY` | optional W&B account or team | W&B client default |

## Camera-ready launcher index

### P4G and tau2-bench

| Method | P4G | tau2-bench retail |
| --- | --- | --- |
| RL (single GPT-5-mini simulator) | [`qwen3_4b_instruct/0313/baseline/p4g/train_gpt5mini.sh`](qwen3_4b_instruct/0313/baseline/p4g/train_gpt5mini.sh) | [`qwen3_4b_instruct/0313/baseline/tau2/train_gpt5mini.sh`](qwen3_4b_instruct/0313/baseline/tau2/train_gpt5mini.sh) |
| Persona-guided | P4G scenarios already contain personas | [`qwen3_4b_instruct/0312/train_tau2_configured.sh`](qwen3_4b_instruct/0312/train_tau2_configured.sh) |
| Verbalized Sampling | [`qwen3_4b_instruct/0403/verbalized_gpt5mini/p4g/train.sh`](qwen3_4b_instruct/0403/verbalized_gpt5mini/p4g/train.sh) | [`qwen3_4b_instruct/0403/verbalized_gpt5mini/tau2/train.sh`](qwen3_4b_instruct/0403/verbalized_gpt5mini/tau2/train.sh) |
| Frozen ensemble | [`qwen3_4b_instruct/0403/ensemble/p4g/train.sh`](qwen3_4b_instruct/0403/ensemble/p4g/train.sh) | [`qwen3_4b_instruct/0403/ensemble/tau2/train.sh`](qwen3_4b_instruct/0403/ensemble/tau2/train.sh) |
| Co-Training | [`qwen3_4b_instruct/0417/train_p4g_cotrain.sh`](qwen3_4b_instruct/0417/train_p4g_cotrain.sh) | [`qwen3_4b_instruct/0417/train_tau2_cotrain_curriculum.sh`](qwen3_4b_instruct/0417/train_tau2_cotrain_curriculum.sh) |
| Population Co-Training | [`qwen3_4b_instruct/0403/evolving_checkpoint/p4g/train.sh`](qwen3_4b_instruct/0403/evolving_checkpoint/p4g/train.sh) | [`qwen3_4b_instruct/0403/evolving_checkpoint/tau2/train.sh`](qwen3_4b_instruct/0403/evolving_checkpoint/tau2/train.sh) |

For example:

```bash
bash cmd/slime/qwen3_4b_instruct/0403/verbalized_gpt5mini/tau2/train.sh
```

### CooperBench

The released Qwen3.5-27B configuration is:

```bash
pip install -e ./external/CooperBench
COOPERBENCH_SETTING=coop \
  bash cmd/slime/qwen3_5_27b/0304/train_cooperbench.sh
```

`COOPERBENCH_SETTING` accepts `baseline`, `solo`, or `coop`. The cooperative configuration also needs a running Redis service and a configured sandbox backend.

## Co-Training compatibility patch

Dual-model Co-Training needs the per-server engine routing in [`patches/slime_cotrain_combined.patch`](../../patches/slime_cotrain_combined.patch). Apply it once to the pinned Slime submodule:

```bash
git -C slime apply --check ../patches/slime_cotrain_combined.patch
git -C slime apply ../patches/slime_cotrain_combined.patch
```

To check whether it is already applied:

```bash
git -C slime apply --reverse --check ../patches/slime_cotrain_combined.patch
```

Do not commit the patched Slime worktree into this repository; the superproject continues to pin the upstream submodule commit and records the compatibility change as a reviewable patch.

## What each run writes

Each launcher creates a unique `OUTPUT_DIR` containing:

- model checkpoints
- a resolved evaluation-panel YAML generated from `eval_configs/`
- Slime/Ray logs and tracker metadata

Generated results are intentionally ignored by Git. Move only final aggregate artifacts into a separate archival location; do not commit API responses, W&B caches, or human-study records.

## Static validation

Validate launchers without allocating GPUs or starting services:

```bash
git ls-files -z 'cmd/slime/**/*.sh' | xargs -0 -n1 bash -n
```

Then run the Python test suite from the repository root:

```bash
pytest -q
```

## Historical configurations

Directories named `0206`, `0220`, `0226`, and the non-indexed variants under later dates capture earlier ablations, model providers, and infrastructure assumptions. They are kept for provenance, but the camera-ready table above is the supported reproduction surface. In particular, historical scripts may require different checkpoint locations, GPU partitions, or provider-specific API routing.
