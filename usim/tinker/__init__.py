"""Tinker backend for user simulator training.

This module provides integration with the Tinker RL framework.

Requires: tinker and tinker_cookbook packages to be installed
"""

from usim.tinker.model_adapter import TinkerModelAdapter, create_tinker_model_adapter
from usim.tinker.trajectory_converter import usim_trajectory_to_tinker_trajectory
from usim.tinker.rollout import usim_generate_rollout

__all__ = [
    "TinkerModelAdapter",
    "create_tinker_model_adapter",
    "usim_trajectory_to_tinker_trajectory",
    "usim_generate_rollout",
]
