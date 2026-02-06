"""Convert usim Trajectory to Slime Sample format."""

import logging
from typing import Any, Dict, List

from usim.core.types import Trajectory, TrajectoryStatus


logger = logging.getLogger(__name__)


def trajectory_to_slime_sample(
    trajectory: Trajectory,
    index: int = 0,
    role: str = "actor",
) -> Any:
    """Convert a usim Trajectory to Slime's Sample format.

    This is the bridge between usim's framework-agnostic Trajectory
    and Slime's training data format.

    Args:
        trajectory: usim Trajectory to convert
        index: Sample index for batching
        role: Role identifier ("actor" or "environment")

    Returns:
        Slime Sample object
    """
    try:
        from slime.data.types import Sample
    except ImportError:
        raise ImportError("slime package required for trajectory conversion")

    # Map TrajectoryStatus to Slime Sample.Status
    status_map = {
        TrajectoryStatus.COMPLETED: Sample.Status.COMPLETED,
        TrajectoryStatus.TRUNCATED: Sample.Status.TRUNCATED,
        TrajectoryStatus.ABORTED: Sample.Status.ABORTED,
        TrajectoryStatus.FAILED: Sample.Status.FAILED,
        TrajectoryStatus.TIMEOUT: Sample.Status.TRUNCATED,
        TrajectoryStatus.PENDING: Sample.Status.PENDING,
        TrajectoryStatus.RUNNING: Sample.Status.PENDING,
    }
    slime_status = status_map.get(trajectory.status, Sample.Status.PENDING)

    # Build metadata, ensuring sample_weight is preserved for loss weighting
    metadata = {
        "role": role,
        "turn_count": trajectory.turn_count,
        "sample_weight": trajectory.metadata.get("sample_weight", 1.0),
        "messages": trajectory.messages,
        **trajectory.metadata,
    }

    # Determine response_length from loss_mask if not set
    response_length = trajectory.response_length
    if response_length == 0 and trajectory.loss_mask:
        response_length = sum(trajectory.loss_mask)

    return Sample(
        index=index,
        prompt=trajectory.prompt,
        tokens=trajectory.tokens,
        response=trajectory.response,
        response_length=response_length,
        reward=trajectory.reward,
        loss_mask=trajectory.loss_mask,
        rollout_log_probs=(
            trajectory.rollout_log_probs if trajectory.rollout_log_probs else None
        ),
        status=slime_status,
        metadata=metadata,
    )


def slime_sample_to_trajectory(sample: Any) -> Trajectory:
    """Convert a Slime Sample back to usim Trajectory.

    Args:
        sample: Slime Sample object

    Returns:
        usim Trajectory
    """
    metadata = getattr(sample, "metadata", {}) or {}

    return Trajectory(
        prompt=sample.prompt,
        tokens=sample.tokens,
        loss_mask=sample.loss_mask if hasattr(sample, "loss_mask") else [],
        response=sample.response,
        response_length=getattr(sample, "response_length", 0),
        reward=sample.reward,
        rollout_log_probs=list(getattr(sample, "rollout_log_probs", []) or []),
        messages=metadata.get("messages", []),
        turn_count=metadata.get("turn_count", 0),
        metadata={k: v for k, v in metadata.items() if k not in ("messages", "turn_count")},
    )


def batch_trajectories_to_samples(
    trajectories: list,
    start_index: int = 0,
    role: str = "actor",
) -> list:
    """Convert a batch of trajectories to Slime samples.

    Args:
        trajectories: List of usim Trajectories
        start_index: Starting index for samples
        role: Role identifier ("actor" or "environment")

    Returns:
        List of Slime Sample objects
    """
    samples = []
    for i, traj in enumerate(trajectories):
        samples.append(trajectory_to_slime_sample(traj, index=start_index + i, role=role))
    return samples


def trajectories_to_grouped_samples(
    env_trajectories: List[Trajectory],
    actor_trajectories: List[Trajectory],
    base_index: int = 0,
) -> List[List[Any]]:
    """Convert environment and actor trajectories to grouped Slime Samples.

    Slime expects list[list[Sample]] where inner lists are n_samples_per_prompt groups.
    For usim, we treat each sample as its own group.

    Args:
        env_trajectories: Environment generation trajectories
        actor_trajectories: Actor gameplay trajectories
        base_index: Starting index for samples

    Returns:
        List of Sample groups (each group is a single-element list)
    """
    grouped_samples = []
    current_index = base_index

    for traj in env_trajectories:
        sample = trajectory_to_slime_sample(
            trajectory=traj,
            index=current_index,
            role="environment",
        )
        grouped_samples.append([sample])
        current_index += 1

    for traj in actor_trajectories:
        sample = trajectory_to_slime_sample(
            trajectory=traj,
            index=current_index,
            role="actor",
        )
        grouped_samples.append([sample])
        current_index += 1

    return grouped_samples


def validate_sample_for_training(sample: Any) -> Dict[str, Any]:
    """Validate that a Sample is ready for training.

    Args:
        sample: Slime Sample to validate

    Returns:
        Dict with 'valid' bool and 'errors' list
    """
    errors = []

    # Check required fields
    if not hasattr(sample, "tokens") or not sample.tokens:
        errors.append("Sample has no tokens")

    if not hasattr(sample, "loss_mask") or not sample.loss_mask:
        errors.append("Sample has no loss_mask")

    # Check that tokens >= loss_mask (loss_mask excludes prompt tokens)
    if hasattr(sample, "tokens") and hasattr(sample, "loss_mask"):
        if len(sample.tokens) < len(sample.loss_mask):
            errors.append(
                f"Token/mask length mismatch: {len(sample.tokens)} tokens, "
                f"{len(sample.loss_mask)} mask values (tokens must be >= mask)"
            )

    # Check for trainable content
    if hasattr(sample, "loss_mask") and sum(sample.loss_mask) == 0:
        errors.append("Sample has no trainable tokens")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
