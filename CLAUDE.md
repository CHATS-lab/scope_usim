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
pytest tests/ --cov=usim --cov-report=html # With coverage
```
Async tests use `asyncio_mode = "auto"` (needs `pytest-asyncio`).

### Code Quality
```bash
black --line-length 100 usim/ tests/
isort --profile black usim/ tests/
```
Line length is 100 characters. Target Python is 3.10+.

**Import convention:** Always place imports at the top of the file. Do not use lazy imports inside functions unless absolutely necessary (e.g., circular import resolution). Backend-specific modules (`usim/slime/`, `usim/tinker/`) may import their backend dependencies (`slime`, `tau2`, `sglang`) at the top level since they are only loaded when the backend is installed.

### Training (Slime)
```bash
# 0206 experiment: Qwen3-4B-Instruct + gpt-5-mini user sim on tau2-bench retail
bash cmd/slime/qwen3_4b_instruct/0206/train_tau2.sh

# CooperBench: Qwen3-4B-Instruct + gpt-5-mini partner on CooperBench lite
bash cmd/slime/qwen3_4b_instruct/0206/train_cooperbench.sh

# Persuasion for Good: Qwen3-4B-Instruct (persuader) + gpt-5-mini (persuadee)
bash cmd/slime/qwen3_4b_instruct/0206/train_p4g.sh
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
- Agent responses parsed for bash code blocks via regex
- Environment executes command, returns observation as "user" message
- Incoming partner messages injected via messaging connector
- Stop condition: `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

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
| `--cooperbench-subset` | `None` | Subset: `lite`, `flash`, or all |
| `--cooperbench-repo` | `None` | Filter by repository name |
| `--cooperbench-partner-model` | `gpt-5-mini` | Partner agent's LLM model |
| `--cooperbench-max-steps` | `50` | Max steps per agent |
| `--cooperbench-redis-url` | `redis://localhost:6379` | Redis for messaging |
| `--cooperbench-dataset-dir` | `None` | Path to CooperBench directory |
| `--cooperbench-partial-reward` | `False` | 0.5 reward when agent passes but merge fails |

## CLI Arguments (P4G, `train_p4g_slime.py`)

| Arg | Default | Description |
|---|---|---|
| `--trainable-role` | `agent` | Which role to train (agent = persuader) |
| `--max-turns` | `10` | Max conversation turns |
| `--usim-fixed-opponent-model` | `None` | Fixed persuadee model via API |
| `--p4g-corpus-path` | `persuasion_simulation/.../persuasionforgood_corpus` | Path to convokit Corpus |
| `--p4g-dataset-dir` | `persuasion_simulation/.../persuasionforgood_train` | Path to dialogue JSONL dir |
| `--p4g-word-limit` | `50` | Max words per response |
| `--p4g-num-turns` | `10` | Number of conversation turns |

## Experiment Tracker (0206)

### Experiment 1: tau2-bench single model
- **Status**: Script ready, NOT YET RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0206/train_tau2.sh`
- **Agent (trainable)**: Qwen3-4B-Instruct-2507 (SGLang)
- **User sim (fixed)**: gpt-5-mini (single model)
- **Dataset**: tau2-bench retail
- **Key args**: `--trainable-role agent --usim-fixed-opponent-model "gpt-5-mini"`
- **Wandb**: `usim / qwen3-4B-Instruct-2507-tau2-0206`

### Experiment 2: tau2-bench multi-model rotation
- **Status**: SCRIPT NOT YET CREATED
- **Script**: `cmd/slime/qwen3_4b_instruct/0206/train_tau2_multi.sh`
- **Agent (trainable)**: Qwen3-4B-Instruct-2507 (SGLang)
- **User sim (fixed)**: rotation of multiple models (e.g., `gpt-5-mini,gpt-4o-mini,deepseek-v3.2`)
- **Dataset**: tau2-bench retail
- **Key change**: `--usim-fixed-opponent-model "gpt-5-mini,gpt-4o-mini,deepseek-v3.2"`
- **Purpose**: Test whether diverse user simulators improve agent robustness

### Experiment 3: Persuasion for Good
- **Status**: SCRIPT READY, NOT YET RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0206/train_p4g.sh`
- **Agent (trainable)**: Qwen3-4B-Instruct-2507 (SGLang) — plays the persuader role
- **User sim (fixed)**: gpt-5-mini via OpenAI API — plays the persuadee role
- **Dataset**: Persuasion for Good corpus (739 train / 200 test dialogues)
- **Reward**: `donation_amount / 2.0` normalized to [0, 1]
- **Key args**: `--trainable-role agent --p4g-num-turns 10 --usim-fixed-opponent-model "gpt-5-mini" --p4g-word-limit 50`
- **Wandb**: `usim / qwen3-4B-Instruct-2507-p4g-0206`
- **Dependencies**: `convokit` (for persona loading)
- **Implementation**: `usim/p4g/` (prompts, persona, agent, user_simulator, reward, data_source, rollout)
- **Training entry**: `train_p4g_slime.py`

### Experiment 4: CooperBench
- **Status**: SCRIPT READY, NOT YET RUN
- **Script**: `cmd/slime/qwen3_4b_instruct/0206/train_cooperbench.sh`
- **Agent (trainable)**: Qwen3-4B-Instruct-2507 (SGLang) — implements feature 1
- **Partner (fixed)**: gpt-5-mini via CooperBench mini_swe_agent — implements feature 2
- **Dataset**: CooperBench lite subset (from `CodeConflict/cooperbench-dataset`)
- **Reward**: 1.0 if both features pass merged tests, 0.0 otherwise
- **Key args**: `--trainable-role agent --cooperbench-subset lite --cooperbench-partner-model "gpt-5-mini"`
- **Wandb**: `usim / qwen3-4B-Instruct-2507-cooperbench-0206`
- **Dependencies**: `cooperbench`, `mini-swe-agent[full]`, `redis`, `modal`
- **Pre-reqs**: Redis running, Modal configured, CooperBench dataset downloaded

## Pre-run Checklist (Remote Machine)

### 1. Model checkpoints on WORKSPACE_DIR (default `/mnt/workspace`)
```
Qwen3-4B-Instruct-2507/           # HF checkpoint
Qwen3-4B-Instruct-2507_torch_dist/ # Megatron torch_dist format
```
If only HF exists, convert via Slime's `scripts/convert_hf_to_torch_dist.sh`.

### 2. Packages
```bash
pip install -e ".[slime,tau2]"
pip install openai  # For fixed opponent API calls
```

### 3. Environment variables
```bash
export WANDB_API_KEY="..."
export OPENAI_API_KEY="..."          # For gpt-5-mini user sim
export WORKSPACE_DIR="/mnt/workspace" # If not default
```

### 4. Slime directory
Script expects slime at `${PROJECT_ROOT}/../slime` — verify `scripts/models/qwen3-4B-Instruct-2507.sh` exists.

### 5. Eval config & tau2 data
Both auto-generated at runtime. No manual prep needed.

## File Layout

```
usim/
├── cmd/slime/qwen3_4b_instruct/0206/   # Experiment scripts
│   ├── train_tau2.sh                     # Exp 1: tau2-bench
│   ├── train_cooperbench.sh              # Exp 4: CooperBench
│   └── train_p4g.sh                      # Exp 3: Persuasion for Good
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
