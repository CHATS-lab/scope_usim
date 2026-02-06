# USIM: User SIMulator

Two-agent RL training system for user simulation, following the architecture pattern from `spare/`.

## Overview

USIM provides framework-agnostic components for training user simulators through two-agent rollouts (Agent <-> User Simulator). The system supports configurable training targets - you can train the agent, the user simulator, or both.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                              │
│                                                              │
│   ┌──────────┐        ┌──────────────────┐                  │
│   │  Agent   │ ←────→ │  User Simulator  │                  │
│   │          │        │                  │                  │
│   │ (text or │        │ (text or tool    │                  │
│   │  tools)  │        │  calls)          │                  │
│   └────┬─────┘        └────────┬─────────┘                  │
│        │                       │                            │
│        └───────────┬───────────┘                            │
│                    ▼                                        │
│             ┌──────────────┐                                │
│             │ Environment  │                                │
│             │ (tools, state)│                                │
│             └──────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Framework-Agnostic Core**: Core logic works with any RL backend
- **Two-Agent Rollout**: Agent and User Simulator exchange messages
- **Configurable Training Target**: Train agent, user simulator, or both
- **Token Tracking**: Automatic loss mask generation for RL training
- **Slime Integration**: Ready-to-use with Slime backend
- **tau2-bench Compatible**: Works with tau2-bench tasks and environments

## Installation

```bash
# Core package only
pip install -e .

# With Slime backend
pip install -e ".[slime]"

# With tau2-bench integration
pip install -e ".[tau2]"

# Full development setup
pip install -e ".[full]"
```

## Quick Start

### Basic Usage

```python
from usim import UserSimOrchestrator, UserSimConfig, TrainableRole
from usim.core.agent import LLMAgent
from usim.core.user_simulator import LLMUserSimulator

# Configure training
config = UserSimConfig(
    trainable_role=TrainableRole.USER,  # Train user simulator
    max_turns=30,
    temperature=0.7,
)

# Create orchestrator
orchestrator = UserSimOrchestrator(
    agent_model=agent_adapter,
    user_model=user_adapter,
    config=config,
)

# Create agent and user simulator
agent = LLMAgent(model=agent_adapter, config=config)
user_sim = LLMUserSimulator(model=user_adapter, config=config, instructions="...")

# Run session
trajectory = await orchestrator.run_session(task, agent, user_sim)

# trajectory.tokens - all token IDs
# trajectory.loss_mask - 1 for trainable tokens, 0 for others
# trajectory.messages - full conversation history
```

### Training with Slime

```bash
# Run training
bash cmd/slime/qwen3_4b/train_usim_slime.sh
```

## Directory Structure

```
usim/
├── usim/
│   ├── __init__.py                    # Conditional imports
│   ├── __about__.py                   # Version info
│   │
│   ├── core/                          # Framework-agnostic core
│   │   ├── types.py                   # Message, Trajectory, UserSimConfig
│   │   ├── model_adapter.py           # ModelAdapter Protocol
│   │   ├── orchestrator.py            # UserSimOrchestrator
│   │   │
│   │   ├── user_simulator/            # User simulator implementations
│   │   │   ├── base.py                # BaseUserSimulator Protocol
│   │   │   └── llm_user.py            # LLM-based user simulator
│   │   │
│   │   ├── agent/                     # Agent implementations
│   │   │   ├── base.py                # BaseAgent Protocol
│   │   │   └── llm_agent.py           # LLM-based agent
│   │   │
│   │   ├── environment/               # Environment adapters
│   │   │   ├── base.py                # BaseEnvironment Protocol
│   │   │   └── tau_bench_env.py       # tau2-bench adapter
│   │   │
│   │   ├── prompts/                   # Prompt templates
│   │   │   ├── template.py            # System prompts
│   │   │   └── personas.py            # User personas
│   │   │
│   │   └── utils/                     # Utilities
│   │       ├── message_utils.py       # Message manipulation
│   │       └── trajectory_utils.py    # Trajectory helpers
│   │
│   ├── slime/                         # Slime backend
│   │   ├── model_adapter.py           # SlimeModelAdapter
│   │   ├── trajectory_converter.py    # Trajectory -> Sample
│   │   ├── rollout.py                 # Custom rollout function
│   │   └── data_source.py             # tau2-bench data loading
│   │
│   └── tinker/                        # Tinker backend (placeholder)
│       └── model_adapter.py
│
├── cmd/                               # Training commands
│   └── slime/
│       └── qwen3_4b/
│           └── train_usim_slime.sh
│
├── tests/                             # Test suite
│   ├── test_types.py
│   ├── test_orchestrator.py
│   └── test_slime_integration.py
│
├── pyproject.toml
└── README.md
```

## Core Components

### TrainableRole

Configures which role(s) produce training trajectories:

```python
from usim import TrainableRole

# Train only agent
config = UserSimConfig(trainable_role=TrainableRole.AGENT)

# Train only user simulator
config = UserSimConfig(trainable_role=TrainableRole.USER)

# Train both
config = UserSimConfig(trainable_role=TrainableRole.BOTH)
```

### Token Tracking

The orchestrator automatically tracks tokens and generates loss masks:

```python
trajectory = await orchestrator.run_session(task, agent, user_sim)

# Token tracking
print(f"Total tokens: {len(trajectory.tokens)}")
print(f"Trainable tokens: {sum(trajectory.loss_mask)}")
print(f"Response length: {trajectory.response_length}")
```

### Message Flow

1. **Agent -> User**: Agent sends text response
2. **User -> Agent**: User simulator responds
3. **Agent -> Env**: Agent makes tool calls
4. **Env -> Agent**: Environment returns results
5. Repeat until stop signal or max_turns

## Configuration

### UserSimConfig

```python
@dataclass
class UserSimConfig:
    temperature: float = 0.7           # Sampling temperature
    max_tokens: int = 2048             # Max tokens per generation
    max_turns: int = 30                # Max conversation turns
    max_context_length: int = 16384    # Max context in characters
    trainable_role: TrainableRole = TrainableRole.USER
    stop_tokens: List[str] = ["###STOP###", "###TRANSFER###"]
```

### Environment Variables

```bash
# Slime backend
export SGLANG_ROUTER_IP="127.0.0.1"
export SGLANG_ROUTER_PORT="8080"

# tau2-bench
export TAU2_DOMAIN="retail"  # retail, airline, telecom

# Weights & Biases
export WANDB_API_KEY="your-key"
```

## Training

### Slime Backend

```bash
# Train user simulator on tau2-bench retail
cd usim
bash cmd/slime/qwen3_4b/train_usim_slime.sh
```

Key training arguments:

```bash
--trainable-role user         # Train user simulator
--max-turns 30                # Max conversation turns
--usim-domain retail          # tau2-bench domain
--rollout-temperature 0.7     # Generation temperature
```

### Custom Rollout Function

For Slime integration, use the custom rollout:

```python
# In your Slime training script
--rollout-function-path usim.slime.rollout.usim_generate_rollout
--data-source-path usim.slime.data_source.get_tau2_samples
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_orchestrator.py -v

# Run with coverage
pytest tests/ --cov=usim --cov-report=html
```

## Development

### Adding a New Backend

1. Create `usim/your_backend/__init__.py`
2. Implement `ModelAdapter` protocol
3. Create trajectory converter
4. Add conditional import in `usim/__init__.py`

### Extending Prompts

```python
from usim.core.prompts.template import register_template

register_template("my_custom_user", """
Your custom user simulator prompt...
""")
```

## Related Projects

- **spare**: Self-Play with Adaptive Curriculum (same architecture pattern)
- **tau2-bench**: Customer service benchmark
- **slime**: Distributed RL training framework

## License

Apache 2.0
