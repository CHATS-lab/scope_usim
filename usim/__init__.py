"""USIM: User SIMulator - Two-agent RL training for user simulation.

This package provides framework-agnostic components for training user simulators
through two-agent rollouts using Gym-based environments.

Architecture:
    core/           - Framework-agnostic core logic (orchestrator, environment protocol)
    core/environment/ - Environment implementations (tau2, p4g, cooperbench)
    slime/          - Slime backend integration (optional)
    p4g/            - Persuasion for Good rollouts and data
    cooperbench/    - CooperBench rollouts and data

Example usage:
    from usim import UserSimOrchestrator, UserSimConfig, TrainableRole

    config = UserSimConfig(trainable_role=TrainableRole.AGENT, max_turns=30)
    orchestrator = UserSimOrchestrator(tokenizer=tokenizer, config=config)
    trajectory = await orchestrator.rollout(env, generate_fn, sampling_params)
"""

from usim.__about__ import __version__
from usim import core

# Convenience imports from core
from usim.core.types import (
    Message,
    ToolCall,
    Trajectory,
    TrajectoryStatus,
    UserSimConfig,
    TrainableRole,
    UserState,
    AgentState,
    compute_token_delta,
    get_token_delta,
    probe_inter_message_glue,
)
from usim.core.model_adapter import ModelAdapter
from usim.core.orchestrator import UserSimOrchestrator

__all__ = [
    "__version__",
    "core",
    "Message",
    "ToolCall",
    "Trajectory",
    "TrajectoryStatus",
    "UserSimConfig",
    "TrainableRole",
    "UserState",
    "AgentState",
    "ModelAdapter",
    "UserSimOrchestrator",
    "compute_token_delta",
    "get_token_delta",
    "probe_inter_message_glue",
]

# Conditional imports for optional backends (avoid requiring all dependencies)
# Slime backend - requires slime package
try:
    from usim import slime
    __all__.append("slime")
except ImportError:
    slime = None  # Slime backend not available (install with: pip install usim[slime])
