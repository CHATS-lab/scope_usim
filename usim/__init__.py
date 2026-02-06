"""USIM: User SIMulator - Two-agent RL training for user simulation.

This package provides framework-agnostic components for training user simulators
through two-agent rollouts (Agent <-> User Simulator).

Architecture:
    core/           - Framework-agnostic core logic
    slime/          - Slime backend integration (optional)
    tinker/         - Tinker backend integration (optional)

Example usage:
    from usim import UserSimOrchestrator, UserSimConfig, TrainableRole

    config = UserSimConfig(
        trainable_role=TrainableRole.USER,
        max_turns=30,
    )

    orchestrator = UserSimOrchestrator(
        agent_model=agent_adapter,
        user_model=user_adapter,
        config=config,
    )

    trajectory = await orchestrator.run_session(task, agent, user_simulator)
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
)
from usim.core.model_adapter import ModelAdapter
from usim.core.api_model_adapter import OpenAIModelAdapter, create_openai_model_adapter
from usim.core.orchestrator import UserSimOrchestrator, run_session_sync

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
    "OpenAIModelAdapter",
    "create_openai_model_adapter",
    "UserSimOrchestrator",
    "run_session_sync",
    "compute_token_delta",
]

# Conditional imports for optional backends (avoid requiring all dependencies)
# Slime backend - requires slime package
try:
    from usim import slime
    __all__.append("slime")
except ImportError:
    slime = None  # Slime backend not available (install with: pip install usim[slime])

# Tinker backend - requires tinker package
try:
    from usim import tinker
    __all__.append("tinker")
except ImportError:
    tinker = None  # Tinker backend not available (install with: pip install usim[tinker])
