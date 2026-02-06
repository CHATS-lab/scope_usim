"""Core components for user simulator training - framework-agnostic."""

from usim.core.types import (
    Message,
    ToolCall,
    Trajectory,
    UserSimConfig,
    TrainableRole,
    UserState,
    AgentState,
)
from usim.core.model_adapter import ModelAdapter
from usim.core.orchestrator import UserSimOrchestrator

__all__ = [
    "Message",
    "ToolCall",
    "Trajectory",
    "UserSimConfig",
    "TrainableRole",
    "UserState",
    "AgentState",
    "ModelAdapter",
    "UserSimOrchestrator",
]
