"""Shared rollout metrics computation for all training pipelines."""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def compute_rollout_metrics(
    grouped_results: List[List[Any]],
    rollout_id: int,
    prefix: str = "ROLLOUT",
) -> Dict[str, Any]:
    """Compute and log rollout metrics including per-sample debug info.

    Args:
        grouped_results: List of sample groups (each group is n_samples_per_prompt).
        rollout_id: Current rollout/step ID.
        prefix: Logging prefix (e.g. "USIM", "P4G", "CB").

    Returns:
        Dict of metric name -> value for wandb logging.
    """
    all_samples = [s for group in grouped_results for s in group]
    rewards = [s.reward for s in all_samples if s.reward is not None]
    turn_counts = [
        s.metadata.get("turn_count", 0)
        for s in all_samples
        if hasattr(s, "metadata") and s.metadata
    ]
    response_lengths = [s.response_length for s in all_samples]
    truncated = [s.status.value == "truncated" for s in all_samples]
    failed = [s.status.value == "failed" for s in all_samples]

    # Zero-std groups: groups where all samples have the same reward (no learning signal)
    zero_std_groups = 0
    for group in grouped_results:
        group_rewards = [s.reward for s in group if s.reward is not None]
        if len(group_rewards) >= 2:
            std = float(np.std(group_rewards))
            if std == 0.0:
                zero_std_groups += 1

    num_groups = len(grouped_results)
    zero_std_pct = zero_std_groups / max(num_groups, 1)

    metrics = {
        "rollout/num_samples": len(all_samples),
        "rollout/num_groups": num_groups,
        "rollout/raw_reward/mean": float(np.mean(rewards)) if rewards else 0.0,
        "rollout/raw_reward/std": float(np.std(rewards)) if rewards else 0.0,
        "rollout/raw_reward/min": float(np.min(rewards)) if rewards else 0.0,
        "rollout/raw_reward/max": float(np.max(rewards)) if rewards else 0.0,
        "rollout/turn_count/mean": float(np.mean(turn_counts)) if turn_counts else 0.0,
        "rollout/turn_count/max": int(np.max(turn_counts)) if turn_counts else 0,
        "rollout/response_len/mean": float(np.mean(response_lengths)) if response_lengths else 0.0,
        "rollout/truncated_ratio": float(np.mean(truncated)) if truncated else 0.0,
        "rollout/failed_ratio": float(np.mean(failed)) if failed else 0.0,
        "rollout/zero_std_group_pct": zero_std_pct,
    }

    # CooperBench-specific metrics from trajectory metadata
    has_patch_flags = [
        s.metadata.get("has_patch", False)
        for s in all_samples
        if hasattr(s, "metadata") and s.metadata
    ]
    patch_lines = [
        s.metadata.get("agent_patch_lines", 0)
        for s in all_samples
        if hasattr(s, "metadata") and s.metadata
    ]
    tool_ok = [
        s.metadata.get("tool_call_success", 0)
        for s in all_samples
        if hasattr(s, "metadata") and s.metadata
    ]
    tool_fail = [
        s.metadata.get("tool_call_fail", 0)
        for s in all_samples
        if hasattr(s, "metadata") and s.metadata
    ]
    wrote_code = [
        s.metadata.get("has_written_code", False)
        for s in all_samples
        if hasattr(s, "metadata") and s.metadata
    ]
    if has_patch_flags:
        metrics["rollout/has_patch_ratio"] = float(np.mean(has_patch_flags))
    if patch_lines:
        metrics["rollout/patch_lines/mean"] = float(np.mean(patch_lines))
    if tool_ok:
        metrics["rollout/tool_call_ok/mean"] = float(np.mean(tool_ok))
    if tool_fail:
        metrics["rollout/tool_call_fail/mean"] = float(np.mean(tool_fail))
    if wrote_code:
        metrics["rollout/wrote_code_ratio"] = float(np.mean(wrote_code))

    # Log first failed response for debugging tool call parsing
    for s in all_samples[:1]:
        meta = s.metadata or {}
        if meta.get("first_failed_response"):
            logger.info(
                f"[{prefix}] FIRST FAILED RESPONSE (sample {s.index}): "
                f"{meta['first_failed_response']!r}"
            )

    logger.info(
        f"[{prefix}] Rollout {rollout_id}: "
        f"{len(all_samples)} samples in {num_groups} groups | "
        f"reward={metrics['rollout/raw_reward/mean']:.3f}+-{metrics['rollout/raw_reward/std']:.3f} "
        f"[{metrics['rollout/raw_reward/min']:.3f}, {metrics['rollout/raw_reward/max']:.3f}] | "
        f"turns={metrics['rollout/turn_count/mean']:.1f} (max={metrics['rollout/turn_count/max']}) | "
        f"resp_len={metrics['rollout/response_len/mean']:.0f} | "
        f"truncated={metrics['rollout/truncated_ratio']:.1%} | "
        f"failed={metrics['rollout/failed_ratio']:.1%} | "
        f"zero_std_groups={zero_std_pct:.1%} ({zero_std_groups}/{num_groups})"
    )
    if has_patch_flags:
        logger.info(
            f"[{prefix}] Patches: {sum(has_patch_flags)}/{len(has_patch_flags)} samples, "
            f"tool_ok={sum(tool_ok):.0f} tool_fail={sum(tool_fail):.0f} "
            f"wrote_code={sum(wrote_code)}/{len(wrote_code)}"
        )

    # Debug: log first few samples
    for s in all_samples[:3]:
        meta = s.metadata or {}
        logger.info(
            f"[{prefix}] Sample {s.index}: "
            f"reward={s.reward:.3f}, turns={meta.get('turn_count', '?')}, "
            f"tokens={len(s.tokens)}, resp_len={s.response_length}, "
            f"status={s.status}, response={s.response[:100]}..."
        )

    return metrics
