"""Converter between usim's Trajectory and Tinker's Transition/Trajectory format.

This module converts trajectories using token IDs directly from the orchestrator,
avoiding redundant re-tokenization. The orchestrator tracks tokens continuously
via token-in-token-out, so we use those directly.
"""

import logging
from typing import Any, List

from usim.core.types import Trajectory as UsimTrajectory

logger = logging.getLogger(__name__)


def usim_trajectory_to_tinker_trajectory(
    trajectory: UsimTrajectory,
    role: str = "actor",
) -> Any:
    """Convert a usim Trajectory to a Tinker Trajectory.

    Uses token IDs directly from the trajectory (token-in-token-out tracking).
    No re-tokenization needed.

    Args:
        trajectory: Trajectory from orchestrator with tokens and loss_mask
        role: Role identifier ("actor" or "environment")

    Returns:
        Tinker Trajectory object
    """
    if role == "environment":
        return _env_trajectory_to_tinker(trajectory)
    else:
        return _actor_trajectory_to_tinker(trajectory)


def _env_trajectory_to_tinker(trajectory: UsimTrajectory) -> Any:
    """Convert an environment generation trajectory to Tinker format.

    Environment generation is single-turn:
    - obs: prompt tokens (loss_mask=0)
    - action: response tokens (loss_mask=1)

    Args:
        trajectory: Trajectory for environment generation

    Returns:
        Tinker Trajectory with single transition
    """
    from tinker import ModelInput
    from tinker_cookbook.rl.types import (
        Transition,
        Trajectory as TinkerTrajectory,
        TokensWithLogprobs,
    )

    tokens = list(trajectory.tokens)
    loss_mask = list(trajectory.loss_mask)

    if not tokens or not loss_mask:
        empty_ob = ModelInput.from_ints([])
        return TinkerTrajectory(transitions=[], final_ob=empty_ob)

    # Find where response starts (first loss_mask=1)
    response_start = len(loss_mask)

    # Split into observation and action
    obs_tokens = tokens[:-response_start]
    action_tokens = tokens[-response_start:]

    # Get logprobs for action tokens
    action_logprobs = list(trajectory.rollout_log_probs)
    assert len(action_logprobs) == len(action_tokens), (
        f"Logprobs length mismatch: {len(action_logprobs)} vs {len(action_tokens)}, "
        f"obs_tokens: {len(obs_tokens)}, action_tokens: {len(action_tokens)}"
    )

    # Create observation and action
    ob = ModelInput.from_ints(obs_tokens)
    ac_with_logprobs = TokensWithLogprobs(
        tokens=action_tokens,
        maybe_logprobs=action_logprobs,
    )

    # Create transition
    transition = Transition(
        ob=ob,
        ac=ac_with_logprobs,
        reward=trajectory.reward,
        episode_done=True,
        metrics={
            "turn_count": trajectory.turn_count,
        },
    )

    final_ob = ModelInput.from_ints([])
    return TinkerTrajectory(transitions=[transition], final_ob=final_ob)


def _actor_trajectory_to_tinker(trajectory: UsimTrajectory) -> Any:
    """Convert an actor gameplay trajectory to Tinker format.

    Multi-turn gameplay:
    - Turn N obs: all tokens from start up to where Nth response begins
    - Turn N action: the Nth response tokens

    Pattern in tokens/loss_mask:
    [obs1 tokens (0s)] [resp1 tokens (1s)] [obs2 tokens (0s)] [resp2 tokens (1s)] ...

    Args:
        trajectory: Trajectory for actor gameplay

    Returns:
        Tinker Trajectory with multiple transitions (one per turn)
    """
    from tinker import ModelInput
    from tinker_cookbook.rl.types import (
        Transition,
        Trajectory as TinkerTrajectory,
        TokensWithLogprobs,
    )

    tokens = list(trajectory.tokens)
    loss_mask = list(trajectory.loss_mask)
    rollout_log_probs = list(trajectory.rollout_log_probs)

    if not tokens or not loss_mask:
        empty_ob = ModelInput.from_ints([])
        return TinkerTrajectory(transitions=[], final_ob=empty_ob)

    # Find boundaries between 0-runs and 1-runs in the loss mask
    boundaries = (
        [0]
        + [i for i in range(1, len(loss_mask)) if loss_mask[i] != loss_mask[i - 1]]
        + [len(loss_mask)]
    )
    # Actions are the 1-runs (every other segment starting from the first 1-run)
    actions = [
        (boundaries[i], boundaries[i + 1])
        for i in range(0, len(boundaries) - 1, 2)
    ]

    if not actions:
        ob = ModelInput.from_ints(tokens)
        return TinkerTrajectory(transitions=[], final_ob=ob)

    base_offset = len(tokens) - len(loss_mask)
    transitions = []

    for turn_idx, (action_start, action_end) in enumerate(actions):
        obs = tokens[: base_offset + action_start]
        action_tokens_turn = tokens[base_offset + action_start : base_offset + action_end]
        action_len = len(action_tokens_turn)

        logprobs = rollout_log_probs[action_start:action_end]
        assert len(logprobs) == action_len, (
            f"Logprobs length mismatch: {len(logprobs)} vs {action_len}"
        )

        ob = ModelInput.from_ints(obs)
        ac_with_logprobs = TokensWithLogprobs(
            tokens=action_tokens_turn,
            maybe_logprobs=logprobs,
        )

        # Reward: only final turn gets the episode reward
        is_final_turn = turn_idx == len(actions) - 1
        reward = trajectory.reward if is_final_turn else 0.0

        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=reward,
            episode_done=is_final_turn,
            metrics={
                "turn": turn_idx + 1,
                "turn_count": trajectory.turn_count,
            },
        )
        transitions.append(transition)

    final_ob = ob
    return TinkerTrajectory(transitions=transitions, final_ob=final_ob)
