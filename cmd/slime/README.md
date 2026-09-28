# SCOPE experiment launchers

Each script in this directory starts one Slime training or evaluation run for
an experiment in the paper. Run them from anywhere; they locate the repository
root from their own path.

> [!WARNING]
> The launchers assume a dedicated 8-GPU Linux node. Before starting, each one
> runs `pkill -9 sglang`, `ray stop --force` and `pkill -9 ray`, which kills
> every SGLang and Ray process on the machine, and then starts a new Ray head
> that claims all eight GPUs. Set `SKIP_PROCESS_CLEANUP=1` to skip the kill
> step when you manage Ray yourself.

## Layout

```text
cmd/slime/
├── p4g/            # Persuasion for Good, Qwen3-4B-Instruct-2507
├── tau2/           # tau2-bench retail, Qwen3-4B-Instruct-2507
├── ablations/      # appendix ablations (tau2-bench retail)
├── cooperbench/    # CooperBench, Qwen3.5-27B
└── eval_base.sh    # untrained-policy evaluation on tau2-bench, then P4G
```

In the code, Co-Training is `train_cotrain_slime.py --training-mode dual_cotrain`,
and Population Co-Training is `--training-mode dual_selfplay` with the `--pool-*`
flags. Module names such as `usim.slime.cotrain_rollout` use the same `cotrain`
prefix.

## Prerequisites

- Linux with eight CUDA GPUs
- Python 3.10 or newer
- Megatron-LM at `/root/Megatron-LM` (the Slime Docker image puts it there),
  or edit `PYTHONPATH` in the launcher's `RUNTIME_ENV_JSON`
- the `slime` and benchmark submodules
- Hugging Face and Megatron `torch_dist` checkpoints of the policy model
- API keys for the frozen simulators and the evaluation panel

From the repository root:

```bash
git submodule update --init --recursive
pip install -e ".[slime,p4g,tau2]"
pip install -e ./slime
pip install -e ./external/tau2-bench
```

For CooperBench, add `pip install -e ".[cooperbench]" -e ./external/CooperBench`.

Put the checkpoints under `MODEL_DIR`. The Qwen3-4B launchers expect:

```text
$MODEL_DIR/Qwen3-4B-Instruct-2507/
$MODEL_DIR/Qwen3-4B-Instruct-2507_torch_dist/
```

The CooperBench launcher expects `Qwen3.5-27B/` and `Qwen3.5-27B_torch_dist/`.
Slime's conversion tools produce the `torch_dist` copy from the Hugging Face
checkpoint. Model architecture flags come from `slime/scripts/models/`.

## Environment

```bash
export MODEL_DIR="/path/to/model-checkpoints"
export OUTPUT_DIR="/path/to/run-output"      # optional

export OPENAI_API_KEY="<your OpenAI API key>"
export OPENROUTER_API_KEY="<your OpenRouter API key>"

export WANDB_PROJECT="scope"
# export WANDB_MODE="offline"                # keep W&B logging local
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `MODEL_DIR` | Hugging Face and `torch_dist` checkpoints | none; the launcher exits if unset |
| `OUTPUT_DIR` | checkpoints, the resolved eval config, trajectories | `results/<run>/<timestamp>` in the repository |
| `OPENAI_API_KEY` | GPT-5-mini and GPT-4o simulators and evaluators | none |
| `OPENROUTER_API_KEY` | Claude Haiku 4.5, Gemini 3 Flash and the other panel models | none |
| `ANTHROPIC_API_KEY` | `ablations/vs_haiku_tau2.sh` only (Anthropic API directly) | none |
| `WANDB_PROJECT` | W&B project | `scope` |
| `SKIP_PROCESS_CLEANUP` | set to `1` to skip the SGLang/Ray kill step | `0` |
| `COOPERBENCH_SETTING` | `baseline`, `solo` or `coop` for the CooperBench launcher | `solo` |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal sandboxes for CooperBench | none |

Launchers that came from the April runs (`ensemble`, `vs_*`,
`population_co_training`, the ablations) also source `.env` from the repository
root when it exists.

## Paper experiments and their launchers

Paths are relative to `cmd/slime/`. Qwen3-4B-Instruct-2507 is the policy unless
a row says otherwise. "None" means no launcher for that setting exists in this
release or in the development history it was cut from.

### Main results (Persuasion for Good and tau2-bench)

| Paper result | P4G | tau2-bench retail |
| --- | --- | --- |
| Base (untrained) | `p4g/eval_base.sh` | `eval_base.sh` (tau2, then P4G) |
| RL (Single), GPT-5-mini simulator | `p4g/rl_single_gpt5mini.sh` | `tau2/rl_single_gpt5mini.sh` |
| Persona-Guided | not applicable (P4G tasks already carry personas) | `tau2/persona_guided.sh` |
| Ensemble, K=3 | `p4g/ensemble.sh` | `tau2/ensemble.sh` |
| Verbalized Sampling, GPT-5-mini | `p4g/vs_gpt5mini.sh` | `tau2/vs_gpt5mini.sh` |
| Co-Training | `p4g/co_training.sh` | `tau2/co_training.sh` |
| Population Co-Training | `p4g/population_co_training.sh` | `tau2/population_co_training.sh` |
| tau2-bench airline column | | None: every tau2 launcher sets `--usim-domain retail` |
| Qwen3-8B rows and the Qwen3-8B tau2 figure | None | None |

The training curves in the main figure come from the same runs.

### Single-simulator collapse

| Simulator | P4G | tau2-bench retail |
| --- | --- | --- |
| GPT-5-mini | `p4g/rl_single_gpt5mini.sh` | `tau2/rl_single_gpt5mini.sh` |
| Claude Haiku 4.5 | `p4g/rl_single_haiku.sh` | `tau2/rl_single_haiku.sh` |
| Gemini 3 Flash | `p4g/rl_single_gemini.sh` | `tau2/rl_single_gemini.sh` |

### CooperBench

| Paper result | Launcher |
| --- | --- |
| Cross-play with a frozen partner, Qwen3.5-27B | `COOPERBENCH_SETTING=coop cooperbench/train_qwen3_5_27b.sh` (the partner is `gemini-3-flash-preview`) |
| Cross-play with Claude Haiku 4.5 as partner | None; change `--cooperbench-partner-model` |
| Cross-play ensemble (K=3), Co-Training, Population Co-Training | None; `train_cooperbench_slime.py` takes a single API partner and has no trainable-partner or pool mode |
| Any Qwen3.5-9B run | None |
| Qwen3.5-27B with Tinker and LoRA adapters | None; the 27B launcher trains all weights with Slime and Megatron |

The cooperative setting also needs a Redis server (the launcher starts one if
none answers) and Modal credentials for the sandboxes.

### Appendix ablations

| Paper result | Launcher |
| --- | --- |
| Verbalized Sampling against RL (Single), GPT-5-mini | `tau2/vs_gpt5mini.sh` and `tau2/rl_single_gpt5mini.sh` (P4G: the `p4g/` pair) |
| Verbalized Sampling with Claude Haiku 4.5 | `ablations/vs_haiku_tau2.sh` |
| Verbalized Sampling with GPT-4o | `ablations/vs_gpt4o_tau2.sh` |
| Verbalized Sampling with GPT-5 | None |
| Simulator reward: curriculum | `tau2/co_training.sh` (`--tau2-user-reward-mode curriculum`) |
| Simulator reward: cooperative | `ablations/sim_reward_cooperative_tau2.sh` |
| Simulator reward: adversarial | None; the tau2 co-training rollout has no adversarial reward mode |
| Pool size K in {1, 3, 5, 10} | None; both pool launchers use `--pool-size 10 --pool-save-interval 16` |
| Reward-quadrant swaps | None |
| OLMo-3-7B-Instruct | None |

The Haiku and GPT-4o Verbalized Sampling runs evaluate on
`eval_configs/tau2_retail_3model_direct.yaml` (Claude Haiku 4.5, GPT-4o and
GPT-5-mini through their own APIs) instead of the 6-model panel. The
cooperative-reward run is not a one-flag change from the curriculum run: it
also uses `--trainable-role agent`, `--no-agent-kl`, 100 rollouts and a KL
coefficient of 0.01.

### Human study

The study app lives in [`../../human_study/`](../../human_study/README.md). It
serves trained checkpoints through OpenAI-compatible endpoints and defines the
conditions `base`, `rl_single` and `cotraining` (Co-Training); it has no
Verbalized Sampling condition.

## Launcher settings

The paper lists shared settings of 250 training steps, learning rate 1e-6 and a
KL coefficient of 0.005. The launchers were not all run with those values:

| Launchers | `--lr` | `--num-rollout` | `--kl-loss-coef` | Notes |
| --- | --- | --- | --- | --- |
| `*/rl_single_*.sh` | 5e-7 | 1000 | 0.01 | |
| `tau2/persona_guided.sh` | 1e-6 | 1000 | 0.01 | max response length 16384 |
| `*/ensemble.sh`, `*/vs_gpt5mini.sh` | 1e-6 | 100 | 0.01 | P4G max response length 16384 |
| `*/co_training.sh` | 1e-6 | 500 | 0.005 | both roles trained |
| `*/population_co_training.sh` | 1e-6 | 100 | 0.01 | `--no-agent-kl`, `--pool-size 10`, `--pool-save-interval 16` |
| `ablations/vs_*_tau2.sh` | 1e-6 | 100 | 0.01 | `--usim-vs-method prob` |
| `cooperbench/train_qwen3_5_27b.sh` | 1e-6 | 500 | 0.01 | no evaluation during training |

All Qwen3-4B launchers use 16 prompts per rollout batch, 8 samples per prompt,
a global batch of 128, rollout temperature 0.7 and evaluation every 16 steps
(the CooperBench launcher uses 8 prompts and a global batch of 64).

Verbalized Sampling is controlled by `--usim-verbalized-sampling`,
`--usim-vs-num-samples` and `--usim-vs-method`. With `prob` the simulator lists
candidates with verbalized probabilities and one is drawn in proportion to them;
with `random` the prompt asks for candidates without probabilities and one is
drawn uniformly. The two `vs_gpt5mini.sh` launchers use `random`.

The base-evaluation scripts measure the untrained policy at rollout 0
(`--eval-interval 1` without `--skip-eval-before-train`). The job then keeps
training; stop it with `ray job stop` once the rollout-0 evaluation is logged.
`eval_base.sh` starts its P4G job only after the tau2 job ends.

## Co-Training compatibility patch

Co-Training and Population Co-Training (`dual_cotrain`, `dual_selfplay`) need the
per-server engine routing in
[`patches/slime_cotrain_combined.patch`](../../patches/slime_cotrain_combined.patch).
Apply it once to the pinned Slime submodule:

```bash
git -C slime apply --check ../patches/slime_cotrain_combined.patch
git -C slime apply ../patches/slime_cotrain_combined.patch
```

To check whether it is already applied:

```bash
git -C slime apply --reverse --check ../patches/slime_cotrain_combined.patch
```

Do not commit the patched Slime worktree. The superproject pins the upstream
submodule commit and keeps the change as a reviewable patch.

## What each run writes

Each launcher creates `OUTPUT_DIR` with:

- model checkpoints
- the evaluation-panel YAML resolved from `eval_configs/`
- Slime and Ray logs, trajectory JSONL files where `--trajectory-output-dir`
  is set, and tracker metadata

`results/` is ignored by Git. Keep API responses, W&B caches and human-study
records out of the repository.

## Static validation

Check the launchers without allocating GPUs:

```bash
git ls-files -z 'cmd/*.sh' | xargs -0 -n1 bash -n
```

Then run the Python tests from the repository root:

```bash
pytest -q
```
