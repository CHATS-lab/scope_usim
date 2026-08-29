"""Trajectory manipulation utilities."""

from typing import Any, Dict, List

from usim.core.types import Trajectory, TrainableRole


def compute_loss_mask(
    role: str,
    num_tokens: int,
    trainable_role: TrainableRole,
) -> List[int]:
    """Compute loss mask for tokens based on role and training target.

    Args:
        role: Message role ("assistant", "user", "tool", "system")
        num_tokens: Number of tokens to generate mask for
        trainable_role: Which role(s) to train

    Returns:
        List of 0s and 1s indicating trainability
    """
    # System and tool messages are never trained
    if role in ("system", "tool"):
        return [0] * num_tokens

    if trainable_role == TrainableRole.BOTH:
        mask_value = 1 if role in ("assistant", "user") else 0
    elif trainable_role == TrainableRole.AGENT:
        mask_value = 1 if role == "assistant" else 0
    elif trainable_role == TrainableRole.USER:
        mask_value = 1 if role == "user" else 0
    else:
        mask_value = 0

    return [mask_value] * num_tokens


def trajectory_to_dict(trajectory: Trajectory) -> Dict[str, Any]:
    """Convert Trajectory to dictionary format.

    Args:
        trajectory: Trajectory to convert

    Returns:
        Dictionary representation
    """
    return {
        "index": trajectory.index,
        "prompt": trajectory.prompt,
        "tokens": trajectory.tokens,
        "loss_mask": trajectory.loss_mask,
        "rollout_log_probs": trajectory.rollout_log_probs,
        "response": trajectory.response,
        "response_length": trajectory.response_length,
        "reward": trajectory.reward,
        "status": trajectory.status.value,
        "messages": trajectory.messages,
        "turn_count": trajectory.turn_count,
        "metadata": trajectory.metadata,
    }


def merge_trajectories(trajectories: List[Trajectory]) -> Trajectory:
    """Merge multiple trajectories into one.

    Useful for combining agent and user trajectories.

    Args:
        trajectories: List of trajectories to merge

    Returns:
        Single merged trajectory
    """
    if not trajectories:
        return Trajectory()

    if len(trajectories) == 1:
        return trajectories[0]

    # Use first trajectory as base
    merged = Trajectory(
        prompt=trajectories[0].prompt,
        tokens=[],
        loss_mask=[],
        response="",
        reward=sum(t.reward for t in trajectories) / len(trajectories),
        messages=[],
        turn_count=max(t.turn_count for t in trajectories),
        metadata=trajectories[0].metadata.copy(),
    )

    # Merge tokens and masks
    for traj in trajectories:
        merged.tokens.extend(traj.tokens)
        merged.loss_mask.extend(traj.loss_mask)
        merged.messages.extend(traj.messages)

    # Concatenate responses
    merged.response = "\n---\n".join(t.response for t in trajectories if t.response)

    return merged


def split_trajectory_by_role(trajectory: Trajectory) -> Dict[str, Trajectory]:
    """Split a trajectory by role.

    Args:
        trajectory: Trajectory to split

    Returns:
        Dict mapping role to trajectory
    """
    from usim.core.types import Message

    agent_messages = []
    user_messages = []

    for msg_dict in trajectory.messages:
        msg = Message.from_dict(msg_dict)
        if msg.role == "assistant":
            agent_messages.append(msg_dict)
        elif msg.role == "user":
            user_messages.append(msg_dict)

    return {
        "agent": Trajectory(
            prompt=trajectory.prompt,
            messages=agent_messages,
            reward=trajectory.reward,
            metadata={**trajectory.metadata, "role": "agent"},
        ),
        "user": Trajectory(
            prompt=trajectory.prompt,
            messages=user_messages,
            reward=trajectory.reward,
            metadata={**trajectory.metadata, "role": "user"},
        ),
    }


def validate_trajectory(trajectory: Trajectory) -> List[str]:
    """Validate a trajectory for common issues.

    Args:
        trajectory: Trajectory to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check that tokens >= loss_mask (loss_mask excludes prompt tokens)
    if len(trajectory.tokens) < len(trajectory.loss_mask):
        errors.append(
            f"Token/mask length mismatch: {len(trajectory.tokens)} tokens, "
            f"{len(trajectory.loss_mask)} mask values (tokens must be >= mask)"
        )

    # Check for empty trajectory
    if not trajectory.tokens:
        errors.append("Trajectory has no tokens")

    # Check loss mask values
    invalid_mask = [m for m in trajectory.loss_mask if m not in (0, 1)]
    if invalid_mask:
        errors.append(f"Invalid loss mask values: {set(invalid_mask)}")

    # Check for trainable content
    if sum(trajectory.loss_mask) == 0:
        errors.append("Trajectory has no trainable tokens (all loss_mask=0)")

    return errors
