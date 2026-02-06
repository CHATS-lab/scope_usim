# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

USIM (User SIMulator) is a framework-agnostic Python package for training user simulators through two-agent RL rollouts. An Agent and a User Simulator exchange messages in turn-based conversations, with an optional Environment for tool execution. The system tracks tokens, loss masks, and logprobs for RL training. Supports both **Slime** (SGLang HTTP) and **Tinker** (SamplingClient) backends.

## Commands

### Installation
```bash
pip install -e .              # Core only
pip install -e ".[slime]"     # With Slime backend
pip install -e ".[tinker]"    # With Tinker backend (tinker + tinker_cookbook)
pip install -e ".[tau2]"      # With tau2-bench
pip install -e ".[full]"      # All backends + dev tools
```

### Testing
```bash
pytest tests/                              # All tests
pytest tests/test_orchestrator.py -v       # Single test file
pytest tests/test_types.py::TestMessage -v # Single test class
pytest tests/ --cov=usim --cov-report=html # With coverage
```
Async tests use `asyncio_mode = "auto"` (no manual `@pytest.mark.asyncio` needed).

### Code Quality
```bash
black --line-length 100 usim/ tests/   # Format code
isort --profile black usim/ tests/     # Sort imports
mypy usim/                             # Type check
pylint usim/                           # Lint
```
Line length is 100 characters. Target Python is 3.10+.

### Training
```bash
bash cmd/slime/qwen3_4b/train_usim_slime.sh   # Slime training with Qwen3-4B
```

## Architecture

### Core Design: Protocol-Based, Framework-Agnostic

The `usim/core/` package defines all interfaces as `@runtime_checkable` Protocol classes (not ABCs). This enables duck-typing and easy mocking in tests.

**Key protocols:**
- `ModelAdapter` (`core/model_adapter.py`) — Interface for LLM generation backends. `apply_template()` returns `List[int]` (token IDs). Methods: `tokenizer`, `apply_template()`, `generate()`, `generate_async()`.
- `BaseAgent` (`core/agent/base.py`) — Agent interface with `build_messages()`, `parse_response()`, and convenience `generate_next_message_async()`.
- `BaseUserSimulator` (`core/user_simulator/base.py`) — User simulator interface with `build_messages()`, `parse_response()`, and convenience `generate_next_message_async()`.
- `BaseEnvironment` (`core/environment/base.py`) — Environment for tool execution.

### Orchestration Flow (`core/orchestrator.py`)

`UserSimOrchestrator` runs the two-agent conversation loop. **The orchestrator calls `model.generate_async()` directly** — agent/user_sim only handle prompt building (`build_messages()`) and response parsing (`parse_response()`).

Flow:
1. Build agent messages via `agent.build_messages(state)` → call `agent_model.generate_async(messages, input_ids=all_tokens)` → parse via `agent.parse_response(text, state)`
2. If tool calls, Environment executes them → results added
3. Build user_sim messages via `user_simulator.build_messages(state)` → call `user_model.generate_async(messages)` → parse via `user_simulator.parse_response(text, state)`
4. Repeat until stop signal (`###STOP###`, `###TRANSFER###`, `###OUT_OF_SCOPE###`) or `max_turns`

**Token Tracking (spare convention):**
- `all_tokens`: ALL tokens (prompt + all responses)
- `all_masks`: loss masks for response tokens only (**excludes prompt tokens**)
- `all_logprobs`: logprobs for response tokens only (same length as `all_masks`)
- Invariant: `len(loss_mask) == len(rollout_log_probs)`
- Invariant: `len(tokens) >= len(loss_mask)`
- `base_offset = len(tokens) - len(loss_mask)` gives prompt length

Entry points: `run_session()` (async) and `run_session_sync()` (sync wrapper).

### State Management

`AgentState` and `UserState` use immutable updates — `add_message()` returns a new state instance rather than mutating in place.

### Role Flipping in User Simulator

`LLMUserSimulator` generates as the "assistant" role internally (since the LLM produces assistant tokens), then converts the output to "user" role for the conversation history. `build_messages()` returns role-flipped conversation history.

### Backend Integration Pattern

Each backend (e.g., `usim/slime/`, `usim/tinker/`) provides:
1. **ModelAdapter implementation** — wraps the backend's inference (e.g., `SlimeModelAdapter` wraps SGLang HTTP, `TinkerModelAdapter` wraps `SamplingClient`)
2. **Trajectory converter** — transforms `Trajectory` → backend-specific training format (Slime `Sample` or Tinker `Trajectory` with `Transition`s)
3. **Rollout function** — entry point called by the training framework
4. **Data source** — loads task data (e.g., `Tau2DataSource` for tau2-bench)

Backends are conditionally imported in `usim/__init__.py` — the core package works without any backend installed.

### Adding a New Backend

1. Create `usim/your_backend/model_adapter.py` implementing the `ModelAdapter` protocol
2. Create a trajectory converter (`Trajectory` → your format)
3. Create a rollout function as the training entry point
4. Add conditional import in `usim/__init__.py`
5. Add optional dependency group in `pyproject.toml`

## Key Data Types (`core/types.py`)

- `TrajectoryStatus` — Enum: `PENDING`, `RUNNING`, `COMPLETED`, `TRUNCATED`, `TIMEOUT`, `FAILED`, `ABORTED`
- `TrainableRole` — Enum: `AGENT`, `USER`, `BOTH` — controls which tokens get loss_mask=1
- `Message` — Text or tool calls, OpenAI-compatible format
- `ToolCall` — Tool invocation (id, name, arguments)
- `Trajectory` — Complete rollout output: `index`, `tokens`, `loss_mask`, `rollout_log_probs`, `response_length`, `reward`, `status`, `messages`, `turn_count`, `metadata`
- `UserSimConfig` — Config dataclass: `temperature`, `max_tokens`, `max_turns`, `max_context_length`, `trainable_role`, `stop_tokens`
- `compute_token_delta(tokenizer, messages)` — Standalone function for incremental token-in-token-out tracking

## Training Setup

### Models
- **Qwen3-4B-Base** — Primary usim training model (Slime backend)
- Qwen3 family (4B to 30B, Base and Instruct variants) via spare/

### Datasets
- **tau2-bench** — Customer service benchmark. Domains: `retail`, `airline`, `telecom`
- **AIME 2024/2025** — Math competition evaluation (via spare/)
- **Synthetic games** — Generated on-the-fly for cognitive skill training (via spare/)

### Evaluation/Opponent Models
- **gpt-5-mini** — Fixed evaluation model (assesses game difficulty)
- **gpt-5.1-mini** — Hint generation for regret-based rewards

## Environment Variables

```bash
SGLANG_ROUTER_IP=127.0.0.1    # SGLang router for Slime backend
SGLANG_ROUTER_PORT=8080
TAU2_DOMAIN=retail              # tau2-bench domain: retail, airline, telecom
WANDB_API_KEY=...               # Weights & Biases logging
```
