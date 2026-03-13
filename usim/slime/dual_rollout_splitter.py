"""Split combined rollout data into per-model training data.

After RolloutManager.generate() with dp_size=1 returns one combined chunk,
this module splits it into agent and opponent halves, then re-splits each
half by the respective model's DP size.

Data format (from RolloutManager._convert_samples_to_train_data):
  dict of lists, each list has one entry per sample. Samples are ordered:
  agent groups first (flattened), then opponent groups (flattened).
  Agent count = rollout_batch_size * n_samples_per_prompt.
"""

import logging

import ray
from slime.utils.misc import Box

logger = logging.getLogger(__name__)

# Keys that are per-sample (split along batch dimension)
_PER_SAMPLE_KEYS = [
    "tokens",
    "multimodal_train_inputs",
    "response_lengths",
    "rewards",
    "truncated",
    "loss_masks",
    "sample_indices",
    "rollout_log_probs",
    "rollout_routed_experts",
    "prompt",
    "teacher_log_probs",
    "metadata",
]

# Keys that are global (kept intact for each DP rank)
_GLOBAL_KEYS = ["raw_reward", "total_lengths"]

# Keys added by _split_train_data_by_dp that should not be copied
_SPLIT_INTERNAL_KEYS = {"partition", "dynamic_global_batch_size"}


def split_combined_rollout(
    combined_data_refs: list,
    n_agent_samples: int,
    actor_dp_size: int,
    opponent_dp_size: int,
    balance_data: bool = False,
) -> tuple:
    """Split combined rollout data into per-model DP-split training data.

    Args:
        combined_data_refs: Output from RolloutManager.generate() with dp_size=1.
            A list with one Box element.
        n_agent_samples: Number of agent samples (first N in the batch).
        actor_dp_size: DP size for actor training group.
        opponent_dp_size: DP size for opponent training group.
        balance_data: Whether to balance by sequence length across DP ranks.

    Returns:
        (actor_data_refs, opponent_data_refs): Each is a list of Box objects
        (one per DP rank), suitable for RayTrainGroup.async_train().
    """
    combined = ray.get(combined_data_refs[0].inner)

    n_total = len(combined["tokens"])
    n_opponent_samples = n_total - n_agent_samples

    if n_opponent_samples <= 0:
        raise ValueError(
            f"Expected agent+opponent samples, got {n_total} total with "
            f"{n_agent_samples} agent samples"
        )

    logger.info(
        f"Splitting rollout: {n_agent_samples} agent + "
        f"{n_opponent_samples} opponent samples"
    )

    agent_data = _slice_data(combined, 0, n_agent_samples)
    opponent_data = _slice_data(combined, n_agent_samples, n_total)

    actor_refs = _split_by_dp(agent_data, actor_dp_size, balance_data)
    opponent_refs = _split_by_dp(opponent_data, opponent_dp_size, balance_data)

    return actor_refs, opponent_refs


def _slice_data(data: dict, start: int, end: int) -> dict:
    """Slice a data dict along the batch dimension."""
    n_total = len(data["tokens"])
    result = {}

    for key in _PER_SAMPLE_KEYS:
        if key in data:
            result[key] = data[key][start:end]

    for key in _GLOBAL_KEYS:
        if key in data:
            result[key] = data[key]

    # Copy unknown keys (per-sample if length matches, else as-is)
    known = set(_PER_SAMPLE_KEYS) | set(_GLOBAL_KEYS) | _SPLIT_INTERNAL_KEYS
    for key in data:
        if key not in known and key not in result:
            val = data[key]
            if isinstance(val, list) and len(val) == n_total:
                result[key] = val[start:end]
            else:
                result[key] = val

    return result


def _split_by_dp(data: dict, dp_size: int, balance: bool = False) -> list:
    """Split data dict into dp_size chunks, returning list of Box(ObjectRef).

    Mirrors slime.ray.rollout.RolloutManager._split_train_data_by_dp().
    """
    n_samples = len(data["tokens"])
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if balance:
        try:
            from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions

            partitions = get_seqlen_balanced_partitions(
                total_lengths, dp_size, equal_size=True
            )
        except ImportError:
            partitions = [range(i, n_samples, dp_size) for i in range(dp_size)]
    else:
        partitions = [range(i, n_samples, dp_size) for i in range(dp_size)]

    refs = []
    for i in range(dp_size):
        partition = partitions[i]
        chunk = {"partition": partition}

        for key in _PER_SAMPLE_KEYS:
            if key not in data:
                continue
            chunk[key] = [data[key][j] for j in partition]

        for key in _GLOBAL_KEYS:
            if key in data:
                chunk[key] = data[key]

        refs.append(Box(ray.put(chunk)))

    return refs
