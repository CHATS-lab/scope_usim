# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

USIM is a framework-agnostic Python package for training agents through two-agent RL rollouts. An **Agent** (trainable, served by SGLang) and a **User Simulator** (fixed opponent via API, e.g., gpt-5-mini) exchange messages in turn-based conversations, with an optional Environment for tool execution. The system tracks tokens, loss masks, and logprobs for RL training. Supports both **Slime** (SGLang HTTP) and **Tinker** (SamplingClient) backends.

**Key roles:**
- **Agent** = trainable policy (SGLang), `trainable_role=agent`, `loss_mask=1` on assistant tokens
- **User Simulator** = fixed opponent (OpenAI API), `loss_mask=0` — supports model rotation (comma-separated)

## Commands

### Installation
```bash
pip install -e .              # Core only
pip install -e ".[slime]"     # With Slime backend
pip install -e ".[tinker]"    # With Tinker backend
pip install -e ".[tau2]"      # With tau2-bench
pip install -e ".[full]"      # All backends + dev tools
```

### Testing
```bash
pytest tests/                              # All tests
pytest tests/test_orchestrator.py -v       # Single test file
pytest tests/test_cooperbench.py -v        # CooperBench smoke tests (needs slime + cooperbench)
pytest tests/ --cov=usim --cov-report=html # With coverage
```
Async tests use `asyncio_mode = "auto"` (needs `pytest-asyncio`).

**CooperBench smoke tests** (`tests/test_cooperbench.py`): 32 tests covering data source loading (entry counts per setting, sample grouping, cycling), agent prompts (setting-aware collaboration blocks, solo combined prompts, stop signal), and reward functions (all 3 settings, partial reward, error handling, async wrappers). Requires `slime` and `cooperbench` packages — run on remote machine. No Modal/Redis/Docker needed.

### Code Quality
```bash
black --line-length 100 usim/ tests/
isort --profile black usim/ tests/
```
Line length is 100 characters. Target Python is 3.10+.

**Import convention:** Always place imports at the top of the file. Do not use lazy imports inside functions unless absolutely necessary (e.g., circular import resolution). Backend-specific modules (`usim/slime/`, `usim/tinker/`) may import their backend dependencies (`slime`, `tau2`, `sglang`) at the top level since they are only loaded when the backend is installed.

### Training (Slime) — 0226 Experiments
```bash
# tau2-bench: Qwen3-4B-Instruct + gpt-5-mini user sim
bash cmd/slime/qwen3_4b_instruct/0226/train_tau2_gpt5mini.sh

# tau2-bench: Qwen3-4B-Instruct + haiku user sim
bash cmd/slime/qwen3_4b_instruct/0226/train_tau2_haiku.sh

# tau2-bench: Qwen3-4B-Instruct + gemini user sim
bash cmd/slime/qwen3_4b_instruct/0226/train_tau2_gemini.sh

# tau2-bench: Qwen3-4B-Instruct + multi-model rotation
bash cmd/slime/qwen3_4b_instruct/0226/train_tau2_multi.sh

# Persuasion for Good: Qwen3-4B-Instruct (persuader) + gpt-5-mini (persuadee)
bash cmd/slime/qwen3_4b_instruct/0226/train_p4g.sh

# CooperBench: Qwen3-4B-Instruct coding agent (solo setting by default)
bash cmd/slime/qwen3_4b_instruct/0226/train_cooperbench.sh
# Override setting: COOPERBENCH_SETTING=baseline|solo|coop
COOPERBENCH_SETTING=coop bash cmd/slime/qwen3_4b_instruct/0226/train_cooperbench.sh
```

## Architecture

### Core Design: Protocol-Based, Framework-Agnostic

The `usim/core/` package defines all interfaces as `@runtime_checkable` Protocol classes.

**Key protocols:**
- `ModelAdapter` (`core/model_adapter.py`) — LLM generation interface. `apply_template()` returns `List[int]`. Methods: `tokenizer`, `apply_template()`, `generate()`, `generate_async()`.
- `OpenAIModelAdapter` (`core/api_model_adapter.py`) — Wraps AsyncOpenAI for fixed opponent. Shares the local tokenizer with the trainable model for token tracking. Logprobs = 0.0 (not trained).
- `BaseAgent` (`core/agent/base.py`) — Agent with `build_messages()`, `parse_response()`.
- `BaseUserSimulator` (`core/user_simulator/base.py`) — User sim with `build_messages()`, `parse_response()`.

### Environment Directory Convention

Environment implementations go under `usim/core/environment/{env_name}/` alongside the protocol at `usim/core/environment/base.py`. Each environment package has its own `__init__.py`, sandbox wrapper, and any connectors.

### Environment Protocol (`core/environment/base.py`)

All environments implement `BaseEnvironment` — an async Gym-based protocol:
- `async reset()` → `(initial_messages, tools_schema, task_info)`
- `async step(action)` → `(obs, reward, terminated, truncated, info)`
- `parse_response(text)` → `{"normal_text", "calls"}` or None
- `prompt_postprocess_fn` → optional text transform (e.g. tool instruction reformulation)

Env-specific logic (observation conversion, tool parsing, prompt postprocessing) lives in the environment implementation, NOT in the orchestrator or rollout.

### Orchestration Flow (`core/orchestrator.py`)

`UserSimOrchestrator.rollout(env, generate_fn, sampling_params)` runs a complete Gym rollout:

1. `env.reset()` → get initial messages, tools schema, task info
2. Tokenize initial prompt (with `env.prompt_postprocess_fn`)
3. Loop:
   - Generate via `generate_fn(input_ids, sampling_params)` — TITO with `input_ids`
   - Parse via `env.parse_response(text)` — tool calls or plain text
   - Track assistant tokens from model output DIRECTLY (`token_ids`, `logprobs`)
   - Step: `env.step(action)` → `(obs, reward, terminated, truncated, info)`
   - Track env tokens via `_get_token_delta` (`loss_mask=0`, `logprobs=0.0`)
4. Return Trajectory

**Token Tracking (spare convention):**
- `all_tokens`: starts with prompt, grows each turn
- `all_masks`: 1 for assistant, 0 for env — `len(all_masks) == len(all_logprobs)`
- `all_logprobs`: model logprobs for assistant, 0.0 for env
- `base_offset = len(all_tokens) - len(all_masks)` = prompt length
- Assistant tokens: `token_ids` and `logprobs` from model output (not re-tokenized)
- This is critical for GRPO: `importance_ratio = exp(current_logprob - rollout_logprob)`

**generate_fn interface:** `async (input_ids: List[int], sampling_params: Dict) -> {"text", "token_ids", "logprobs", "meta_info"}`

### Fixed Opponent via Model Rotation

When `--usim-fixed-opponent-model` is set, the user simulator is passed to the Gym environment (e.g. `AgentGymEnv(user_llm=model_name, ...)`). Model rotation is supported: pass comma-separated models and each sample picks `models[sample.index % len(models)]`. Per-model base URLs and API keys are also comma-separated.

### Coding Orchestrator (`core/coding_orchestrator.py`)

`CodingAgentOrchestrator` manages agent ↔ environment coding loops (no user simulator). Used for CooperBench.

Key differences from `UserSimOrchestrator`:
- No user simulator — only agent + environment (bash sandbox)
- Agent responses parsed for structured tool calls (Qwen3 `<tool_call>` format, mini-swe-agent-v2)
- `bash` tool calls execute commands; `send_message` tool calls deliver partner messages
- Tool results returned as `role="tool"` messages with JSON content
- `apply_chat_template` called with `tools=` for proper Qwen3 tool-call formatting
- Stop condition: `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` in bash command

### Backend Integration Pattern

Each backend provides:
1. **ModelAdapter** — wraps inference (`SlimeModelAdapter` / `TinkerModelAdapter`)
2. **Trajectory converter** — `Trajectory` → backend format
3. **Rollout function** — training entry point
4. **Data source** — loads tasks

## Key Data Types (`core/types.py`)

- `TrajectoryStatus` — `PENDING`, `RUNNING`, `COMPLETED`, `TRUNCATED`, `TIMEOUT`, `FAILED`, `ABORTED`
- `TrainableRole` — `AGENT`, `USER`, `BOTH`
- `Trajectory` — `index`, `tokens`, `loss_mask`, `rollout_log_probs`, `response_length`, `reward`, `status`, `messages`, `turn_count`, `metadata`
- `UserSimConfig` — `temperature`, `max_tokens`, `max_turns`, `max_context_length`, `trainable_role`, `stop_tokens`
- `compute_token_delta(tokenizer, messages)` — Incremental token tracking

## CLI Arguments (USIM-specific, `train_usim_slime.py`)

| Arg | Default | Description |
|---|---|---|
| `--trainable-role` | `agent` | Which role to train: `agent`, `user`, `both` |
| `--max-turns` | `30` | Max conversation turns |
| `--usim-domain` | `retail` | tau2-bench domain: `retail`, `airline`, `telecom` |
| `--usim-fixed-opponent-model` | `None` | Fixed user sim via API. Comma-separated for rotation |
| `--usim-fixed-opponent-base-url` | `https://api.openai.com/v1` | API base URL |
| `--usim-fixed-opponent-api-key-var` | `OPENAI_API_KEY` | Env var for API key |

## CLI Arguments (CooperBench, `train_cooperbench_slime.py`)

| Arg | Default | Description |
|---|---|---|
| `--trainable-role` | `agent` | Which role to train |
| `--cooperbench-setting` | `solo` | Setting: `baseline`, `solo`, `coop` |
| `--cooperbench-data-dir` | `data/cooperbench` | Path to JSON files (train_pairs.json, etc.) |
| `--cooperbench-backend` | `modal` | Sandbox backend: `modal`, `docker` |
| `--cooperbench-partner-model` | `gpt-5-mini` | Partner agent's LLM model (coop only) |
| `--cooperbench-max-steps` | `50` | Max steps per agent |
| `--cooperbench-redis-url` | `redis://localhost:6379` | Redis for messaging (coop only) |
| `--cooperbench-dataset-dir` | `None` | Path to CooperBench repo (with dataset/) for eval |
| `--cooperbench-partial-reward` | `False` | 0.5 reward when agent passes but merge fails (coop only) |

## CLI Arguments (P4G, `train_p4g_slime.py`)

| Arg | Default | Description |
|---|---|---|
| `--trainable-role` | `agent` | Which role to train (agent = persuader) |
| `--max-turns` | `10` | Max conversation turns |
| `--usim-fixed-opponent-model` | `None` | Fixed persuadee model via API |
| `--p4g-corpus-path` | `data/p4g/corpus` | Path to convokit Corpus |
| `--p4g-dataset-dir` | `data/p4g/train` | Path to dialogue JSONL dir |
| `--p4g-word-limit` | `50` | Max words per response |
| `--p4g-num-turns` | `10` | Number of conversation turns |

## Experiment Tracker (0226)

All experiments use GRPO with within-batch normalization (`--grpo-std-normalization --rewards-normalization`), n_samples_per_prompt=8, and 6-model eval configs in `eval_configs/`.

### Exp 1a: tau2-bench + gpt-5-mini user sim
- **Status**: READY TO RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0226/train_tau2_gpt5mini.sh`
- **Agent**: Qwen3-4B-Instruct-2507 (SGLang)
- **User sim**: gpt-5-mini via OpenAI API
- **Dataset**: tau2-bench retail (1000 tasks)
- **Eval**: 6-model eval every 16 steps (`eval_configs/tau2_retail_6model.yaml`)
- **Wandb**: `usim / qwen3-4B-Instruct-2507-tau2-gpt5mini-0226`

### Exp 1b: tau2-bench + haiku user sim
- **Status**: READY TO RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0226/train_tau2_haiku.sh`
- **User sim**: claude-haiku-4.5 via OpenRouter
- **Wandb**: `usim / qwen3-4B-Instruct-2507-tau2-haiku-0226`

### Exp 1c: tau2-bench + gemini user sim
- **Status**: READY TO RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0226/train_tau2_gemini.sh`
- **User sim**: gemini-3-flash-preview via OpenRouter
- **Wandb**: `usim / qwen3-4B-Instruct-2507-tau2-gemini-0226`

### Exp 1d: tau2-bench + multi-model rotation
- **Status**: READY TO RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0226/train_tau2_multi.sh`
- **User sim**: rotation of gpt-5-mini, haiku-4.5, gemini-3-flash
- **Purpose**: Test whether diverse user simulators improve agent robustness
- **Wandb**: `usim / qwen3-4B-Instruct-2507-tau2-multi-0226`

### Exp 2: Persuasion for Good
- **Status**: READY TO RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0226/train_p4g.sh`
- **Agent**: Qwen3-4B-Instruct-2507 (SGLang) — persuader role
- **User sim**: gpt-5-mini via OpenAI API — persuadee role
- **Dataset**: P4G corpus (739 train / 200 test), data in `data/p4g/`
- **Reward**: `donation_amount / 2.0` normalized to [0, 1]
- **Eval**: 6-model eval every 16 steps (`eval_configs/p4g_6model.yaml`)
- **Wandb**: `usim / qwen3-4B-Instruct-2507-p4g-0226`
- **Training entry**: `train_p4g_slime.py`

### Exp 3: CooperBench (0226)
- **Status**: READY TO RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0226/train_cooperbench.sh`
- **Agent**: Qwen3-4B-Instruct-2507 (SGLang) — coding agent
- **Settings**: baseline (170 features), solo (587 pairs), coop (1174 directed)
- **Data**: Pre-split JSON files in `data/cooperbench/` (train_baseline.json, train_pairs.json)
- **Sandbox**: Modal (via SwerexModalEnvironment for agent, cooperbench backends for eval)
- **Partner** (coop only): gpt-5-mini via mini_swe_agent + LiteLLM
- **Pre-reqs**: Modal configured, CooperBench dataset downloaded
- **Pre-reqs** (coop only): Redis running
- **Wandb**: `usim / qwen3-4B-Instruct-2507-cooperbench-{setting}-0226`
- **Override**: `COOPERBENCH_SETTING=baseline|solo|coop`

## Remote Machine: Quick Start

Use Claude Code on the remote machine to launch and debug experiments. Pull the latest code, then follow the checklist below.

### 1. Model checkpoints on WORKSPACE_DIR (default `/mnt/spare-workspace`)
```
Qwen3-4B-Instruct-2507/           # HF checkpoint
Qwen3-4B-Instruct-2507_torch_dist/ # Megatron torch_dist format
```
If only HF exists, convert via Slime's `scripts/convert_hf_to_torch_dist.sh`.

### 2. Packages
```bash
pip install -e ".[slime,tau2]"
pip install openai convokit  # For fixed opponent API calls + P4G persona loading
```

### 3. Environment variables
```bash
export WANDB_API_KEY="..."
export OPENAI_API_KEY="..."           # For gpt-5-mini user sim
export OPENROUTER_API_KEY="..."       # For haiku/gemini/glm5/qwen/deepseek eval
export WORKSPACE_DIR="/mnt/spare-workspace"
```

### 4. Slime directory
Script expects slime at `${PROJECT_ROOT}/slime` (submodule). Verify `scripts/models/qwen3-4B-Instruct-2507.sh` exists.

### 5. Eval configs
Fixed YAML templates in `eval_configs/`. Resolved via `envsubst` at script runtime — no manual prep needed.

### 6. Running experiments
```bash
# Recommended order: start with gpt5mini (cheapest), then branch out
bash cmd/slime/qwen3_4b_instruct/0226/train_tau2_gpt5mini.sh  # Exp 1a
bash cmd/slime/qwen3_4b_instruct/0226/train_p4g.sh            # Exp 2

# After verifying 1a works, try other user sims
bash cmd/slime/qwen3_4b_instruct/0226/train_tau2_haiku.sh     # Exp 1b
bash cmd/slime/qwen3_4b_instruct/0226/train_tau2_gemini.sh    # Exp 1c
bash cmd/slime/qwen3_4b_instruct/0226/train_tau2_multi.sh     # Exp 1d
```

### 7. Debugging tips
- Check ray dashboard at `http://localhost:8265` for job status
- Slime logs are in the ray job output
- If rollout fails, check the error in `metadata.error` field of returned Sample
- `pkill -9 sglang && ray stop --force` to clean up before re-running
- For P4G: verify `data/p4g/corpus/` exists (persona loading) and `convokit` is installed

## File Layout

```
usim/
├── cmd/slime/qwen3_4b_instruct/0226/   # 0226 experiment scripts
│   ├── train_tau2_gpt5mini.sh            # Exp 1a: tau2 + gpt-5-mini
│   ├── train_tau2_haiku.sh               # Exp 1b: tau2 + haiku
│   ├── train_tau2_gemini.sh              # Exp 1c: tau2 + gemini
│   ├── train_tau2_multi.sh               # Exp 1d: tau2 + multi-model rotation
│   ├── train_p4g.sh                      # Exp 2: P4G
│   └── train_cooperbench.sh             # Exp 3: CooperBench (baseline/solo/coop)
├── eval_configs/                         # Reusable eval YAML templates
│   ├── tau2_retail_6model.yaml           # 6-model eval for tau2
│   └── p4g_6model.yaml                   # 6-model eval for P4G
├── data/p4g/                             # P4G datasets (committed)
│   ├── corpus/                           # Convokit corpus for persona loading
│   ├── train/                            # 739 training dialogues
│   └── test/                             # 200 test dialogues
├── data/cooperbench/                     # CooperBench pre-split JSON files
│   ├── train_pairs.json                  # 587 training pairs (solo/coop)
│   ├── test_pairs.json                   # Test pairs
│   ├── train_baseline.json               # 170 training features (baseline)
│   └── test_baseline.json                # Test baseline features
├── train_usim_slime.py                   # Training entry point (tau2)
├── train_cooperbench_slime.py            # Training entry point (CooperBench)
├── train_p4g_slime.py                    # Training entry point (P4G)
├── usim/
│   ├── core/
│   │   ├── api_model_adapter.py          # OpenAI API adapter (fixed opponent)
│   │   ├── model_adapter.py              # ModelAdapter protocol
│   │   ├── orchestrator.py               # Gym-based rollout loop (env-agnostic)
│   │   ├── coding_orchestrator.py        # Coding agent orchestration loop
│   │   ├── types.py                      # Trajectory, TrainableRole, etc.
│   │   ├── agent/                        # Agent protocol + LLMAgent
│   │   ├── user_simulator/               # UserSim protocol + LLMUserSimulator
│   │   ├── environment/
│   │   │   ├── base.py                   # BaseEnvironment protocol (async Gym)
│   │   │   ├── tau2/                     # tau2-bench environment
│   │   │   │   └── environment.py        # Tau2Environment (wraps AgentGymEnv)
│   │   │   ├── p4g/                      # Persuasion for Good environment
│   │   │   │   └── environment.py        # P4gEnvironment (manages persuadee API)
│   │   │   └── cooperbench/              # CooperBench environment
│   │   │       ├── sandbox.py            # Modal sandbox wrapper
│   │   │       ├── environment.py        # USIM environment adapter
│   │   │       └── messaging.py          # Redis messaging connector
│   │   ├── prompts/                      # Prompt templates
│   │   └── utils/                        # Message & trajectory utils
│   ├── cooperbench/
│   │   ├── agent.py                      # CooperBenchAgent (system/instance prompts)
│   │   ├── data_source.py                # Task loading via discover_tasks()
│   │   ├── partner.py                    # Partner agent runner (thread)
│   │   ├── reward.py                     # Merge-test reward computation
│   │   └── rollout.py                    # Rollout function for Slime
│   ├── p4g/
│   │   ├── prompts.py                    # Persuader/persuadee prompt templates
│   │   ├── persona.py                    # PersonaLoader (convokit Corpus)
│   │   ├── agent.py                      # PersuaderAgent
│   │   ├── user_simulator.py             # PersuadeeUserSimulator
│   │   ├── reward.py                     # Donation-based reward
│   │   ├── data_source.py                # P4GDataSource (JSONL dialogues)
│   │   └── rollout.py                    # P4G rollout for Slime
│   ├── slime/
│   │   ├── model_adapter.py              # SlimeModelAdapter (SGLang)
│   │   ├── rollout.py                    # Slime rollout glue (env + generate_fn → orchestrator)
│   │   ├── trajectory_converter.py       # Trajectory → Slime Sample
│   │   └── data_source.py               # Tau2DataSource
│   └── tinker/
│       ├── model_adapter.py              # TinkerModelAdapter (SamplingClient)
│       ├── rollout.py                    # Tinker rollout entry
│       └── trajectory_converter.py       # Trajectory → Tinker format
└── tests/
```

## Related Repos

- `spare/` — Sister project with Slime/Tinker training infrastructure. usim's patterns ported from spare.
- `slime/` — SGLang-based RL training framework. Model configs at `slime/scripts/models/`.
- `slime/examples/persuasion/` — Reference implementation for Persuasion for Good (to be ported to usim for Exp 3).
- `CooperBench/` — Collaboration benchmark. Dataset: `CodeConflict/cooperbench-dataset` on HF.
