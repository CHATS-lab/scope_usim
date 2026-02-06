"""Slime backend for user simulator training.

This module provides integration with the Slime RL framework,
including model adapters, trajectory converters, and rollout functions.

Requires: slime package to be installed
"""

from usim.slime.model_adapter import SlimeModelAdapter, create_slime_model_adapter
from usim.slime.trajectory_converter import trajectory_to_slime_sample
from usim.slime.rollout import usim_generate_rollout

__all__ = [
    "SlimeModelAdapter",
    "create_slime_model_adapter",
    "trajectory_to_slime_sample",
    "usim_generate_rollout",
]
