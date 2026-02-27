"""Slime rollout entry point for CooperBench training.

Batch-level rollout function following the P4G pattern:
1. Gets samples from data_source
2. Dispatches per-sample async rollouts by setting (baseline/solo/coop)
3. Returns RolloutFnTrainOutput with grouped samples and metrics

Supports three settings:
- baseline: 1 agent, 1 feature, test that feature
- solo: 1 agent, 2 features (combined prompt), test both
- coop: 2 agents, 1 feature each, merge + test both
"""

import asyncio
import logging
import threading
import uuid
from argparse import Namespace
from typing import Any, Dict, List

import numpy as np
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.types import Sample

from usim.cooperbench.agent import CooperBenchAgent
from usim.cooperbench.partner import run_partner_agent
from usim.cooperbench.reward import (
    compute_baseline_reward_async,
    compute_coop_reward_async,
    compute_solo_reward_async,
)
from usim.core.coding_orchestrator import CodingAgentOrchestrator
from usim.core.environment.cooperbench.environment import CooperBenchEnvironment
from usim.core.environment.cooperbench.messaging import MessagingConnector
from usim.core.types import TrainableRole, UserSimConfig
from usim.slime.model_adapter import create_slime_model_adapter
from usim.slime.trajectory_converter import trajectory_to_slime_sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sample rollout functions
# ---------------------------------------------------------------------------


async def _baseline_single(
    args: Any,
    sample: Sample,
    sampling_params: Dict[str, Any],
) -> Sample:
    """Single baseline rollout: 1 agent, 1 feature."""
    environment = None
    try:
        metadata = getattr(sample, "metadata", {}) or {}
        repo = metadata["repo"]
        task_id = metadata["task_id"]
        image_name = metadata["image_name"]
        agent_feature_id = metadata["agent_feature_id"]
        descriptions = metadata.get("descriptions", {})
        max_steps = getattr(args, "cooperbench_max_steps", 50)
        dataset_dir = getattr(args, "cooperbench_dataset_dir", None)
        backend = getattr(args, "cooperbench_backend", "modal")

        # Create environment
        environment = CooperBenchEnvironment(image_name=image_name, timeout=3600)

        # Create agent (no collaboration)
        agent = CooperBenchAgent(agent_id="agent1", setting="baseline")

        # Task message from single feature description
        task_description = descriptions.get(
            str(agent_feature_id), descriptions.get(agent_feature_id, "")
        )
        task = {
            "id": f"{repo}/task{task_id}/f{agent_feature_id}",
            "repo": repo,
            "task_id": task_id,
            "feature_id": agent_feature_id,
            "instructions": agent.get_task_message(task_description),
        }

        # Use existing SlimeModelAdapter (handles SGLang HTTP, tokenizer, logprobs)
        model_adapter = create_slime_model_adapter(args)

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_steps,
            max_tokens=getattr(args, "rollout_max_response_len", 4096),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = CodingAgentOrchestrator(
            agent_model=model_adapter,
            config=config,
            environment=environment,
        )

        trajectory = await orchestrator.run_session(task, agent)

        # Get patch and compute reward (runs in thread pool, doesn't block event loop)
        agent_patch = await environment.get_patch()
        reward = await compute_baseline_reward_async(
            repo_name=repo,
            task_id=task_id,
            feature_id=agent_feature_id,
            patch=agent_patch,
            dataset_dir=dataset_dir,
            backend=backend,
        )

        trajectory.reward = reward
        trajectory.metadata["agent_patch_lines"] = len(agent_patch.splitlines())
        trajectory.metadata["setting"] = "baseline"

        logger.info(
            f"[CB] Baseline {sample.index}: {trajectory.turn_count} steps, "
            f"reward={reward:.1f}, status={trajectory.status}"
        )

        return trajectory_to_slime_sample(trajectory, sample.index)

    except Exception as e:
        logger.error(f"Baseline rollout failed (sample {sample.index}): {e}", exc_info=True)
        return _error_sample(sample, str(e))
    finally:
        if environment:
            try:
                await environment.cleanup()
            except Exception:
                pass


async def _solo_single(
    args: Any,
    sample: Sample,
    sampling_params: Dict[str, Any],
) -> Sample:
    """Single solo rollout: 1 agent, 2 features (combined prompt)."""
    environment = None
    try:
        metadata = getattr(sample, "metadata", {}) or {}
        repo = metadata["repo"]
        task_id = metadata["task_id"]
        image_name = metadata["image_name"]
        feature_ids = metadata["feature_ids"]
        descriptions = metadata.get("descriptions", {})
        max_steps = getattr(args, "cooperbench_max_steps", 50)
        dataset_dir = getattr(args, "cooperbench_dataset_dir", None)
        backend = getattr(args, "cooperbench_backend", "modal")
        f1_id, f2_id = feature_ids

        # Create environment
        environment = CooperBenchEnvironment(image_name=image_name, timeout=3600)

        # Create agent with combined prompt for both features
        agent = CooperBenchAgent(agent_id="agent1", setting="solo")

        task = {
            "id": f"{repo}/task{task_id}/f{f1_id}_f{f2_id}",
            "repo": repo,
            "task_id": task_id,
            "feature_id": f"{f1_id}_{f2_id}",
            "instructions": agent.get_solo_task_message(descriptions),
        }

        model_adapter = create_slime_model_adapter(args)

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_steps,
            max_tokens=getattr(args, "rollout_max_response_len", 4096),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = CodingAgentOrchestrator(
            agent_model=model_adapter,
            config=config,
            environment=environment,
        )

        trajectory = await orchestrator.run_session(task, agent)

        # Get patch and compute reward (both features must pass)
        agent_patch = await environment.get_patch()
        reward = await compute_solo_reward_async(
            repo_name=repo,
            task_id=task_id,
            f1_id=f1_id,
            f2_id=f2_id,
            patch=agent_patch,
            dataset_dir=dataset_dir,
            backend=backend,
        )

        trajectory.reward = reward
        trajectory.metadata["agent_patch_lines"] = len(agent_patch.splitlines())
        trajectory.metadata["setting"] = "solo"

        logger.info(
            f"[CB] Solo {sample.index}: {trajectory.turn_count} steps, "
            f"reward={reward:.1f}, status={trajectory.status}"
        )

        return trajectory_to_slime_sample(trajectory, sample.index)

    except Exception as e:
        logger.error(f"Solo rollout failed (sample {sample.index}): {e}", exc_info=True)
        return _error_sample(sample, str(e))
    finally:
        if environment:
            try:
                await environment.cleanup()
            except Exception:
                pass


async def _coop_single(
    args: Any,
    sample: Sample,
    sampling_params: Dict[str, Any],
) -> Sample:
    """Single coop rollout: trainable agent + fixed partner, merge + test."""
    environment = None
    partner_result = {"patch": ""}
    partner_error = {"error": None}

    try:
        metadata = getattr(sample, "metadata", {}) or {}
        repo = metadata["repo"]
        task_id = metadata["task_id"]
        image_name = metadata["image_name"]
        agent_feature_id = metadata["agent_feature_id"]
        partner_feature_id = metadata["partner_feature_id"]
        feature_ids = metadata["feature_ids"]
        descriptions = metadata.get("descriptions", {})
        partner_model = getattr(args, "cooperbench_partner_model", "gpt-5-mini")
        max_steps = getattr(args, "cooperbench_max_steps", 50)
        redis_url = getattr(args, "cooperbench_redis_url", "redis://localhost:6379")
        dataset_dir = getattr(args, "cooperbench_dataset_dir", None)
        backend = getattr(args, "cooperbench_backend", "modal")
        partial_reward = getattr(args, "cooperbench_partial_reward", False)
        f1_id, f2_id = feature_ids

        # Create unique run ID for Redis namespacing
        run_id = uuid.uuid4().hex[:8]
        namespaced_redis = f"{redis_url}#run:{run_id}"
        agents = ["agent1", "agent2"]

        # Messaging connector for trainable agent
        messaging = MessagingConnector(
            agent_id="agent1",
            agents=agents,
            url=namespaced_redis,
        )

        # Start partner agent in background thread
        def _run_partner():
            try:
                partner_result["patch"] = run_partner_agent(
                    repo_name=repo,
                    task_id=task_id,
                    feature_id=partner_feature_id,
                    model_name=partner_model,
                    agent_id="agent2",
                    agents=agents,
                    redis_url=namespaced_redis,
                    dataset_dir=dataset_dir,
                )
            except Exception as e:
                partner_error["error"] = str(e)
                logger.error(f"Partner agent failed: {e}")

        partner_thread = threading.Thread(target=_run_partner, daemon=True)
        partner_thread.start()

        # Create environment for trainable agent (with send_message tool enabled)
        environment = CooperBenchEnvironment(
            image_name=image_name,
            messaging=messaging,
            messaging_enabled=True,
            timeout=3600,
        )

        # Create agent with coop collaboration prompts
        agent = CooperBenchAgent(
            agent_id="agent1",
            agents=agents,
            messaging_enabled=True,
            setting="coop",
        )

        task_description = descriptions.get(
            str(agent_feature_id), descriptions.get(agent_feature_id, "")
        )
        task = {
            "id": f"{repo}/task{task_id}/f{agent_feature_id}",
            "repo": repo,
            "task_id": task_id,
            "feature_id": agent_feature_id,
            "instructions": agent.get_task_message(task_description),
        }

        model_adapter = create_slime_model_adapter(args)

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_steps,
            max_tokens=getattr(args, "rollout_max_response_len", 4096),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = CodingAgentOrchestrator(
            agent_model=model_adapter,
            config=config,
            environment=environment,
            messaging=messaging,
        )

        trajectory = await orchestrator.run_session(task, agent)

        # Get trainable agent's patch
        agent_patch = await environment.get_patch()

        # Wait for partner
        partner_thread.join(timeout=3600)
        partner_patch = partner_result["patch"]

        if partner_error["error"]:
            logger.warning(f"Partner had error: {partner_error['error']}")

        # Compute coop reward (merge + test, runs in thread pool)
        reward = await compute_coop_reward_async(
            repo_name=repo,
            task_id=task_id,
            f1_id=f1_id,
            f2_id=f2_id,
            agent_patch=agent_patch,
            partner_patch=partner_patch,
            dataset_dir=dataset_dir,
            backend=backend,
            partial_reward=partial_reward,
        )

        trajectory.reward = reward
        trajectory.metadata["agent_patch_lines"] = len(agent_patch.splitlines())
        trajectory.metadata["partner_patch_lines"] = len(partner_patch.splitlines())
        trajectory.metadata["partner_model"] = partner_model
        trajectory.metadata["setting"] = "coop"

        logger.info(
            f"[CB] Coop {sample.index}: {trajectory.turn_count} steps, "
            f"reward={reward:.1f}, status={trajectory.status}"
        )

        return trajectory_to_slime_sample(trajectory, sample.index)

    except Exception as e:
        logger.error(f"Coop rollout failed (sample {sample.index}): {e}", exc_info=True)
        return _error_sample(sample, str(e))
    finally:
        if environment:
            try:
                await environment.cleanup()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


async def _run_batch_async(
    args: Namespace,
    samples: List[List[Sample]],
) -> List[List[Sample]]:
    """Run a batch of samples concurrently, returning grouped results."""
    setting = getattr(args, "cooperbench_setting", "solo")
    sampling_params = dict(
        temperature=args.rollout_temperature,
        top_p=getattr(args, "rollout_top_p", 0.95),
        top_k=getattr(args, "rollout_top_k", -1),
        max_new_tokens=getattr(args, "rollout_max_response_len", 4096),
    )

    dispatch = {
        "baseline": _baseline_single,
        "solo": _solo_single,
        "coop": _coop_single,
    }
    fn = dispatch[setting]

    tasks = []
    for group in samples:
        for sample in group:
            tasks.append(fn(args, sample, sampling_params))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Reconstruct grouping
    grouped = []
    idx = 0
    for group in samples:
        group_results = []
        for _ in group:
            result = results[idx]
            if isinstance(result, Exception):
                logger.error(f"Sample failed: {result}")
                result = _error_sample(group[0], str(result))
            group_results.append(result)
            idx += 1
        grouped.append(group_results)

    return grouped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def cooperbench_generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Batch-level rollout function for CooperBench."""
    if evaluation:
        return _cooperbench_eval_rollout(args, rollout_id, data_source)

    samples = data_source.get_samples(args.rollout_batch_size)
    grouped_results = asyncio.run(_run_batch_async(args, samples))

    metrics = _compute_rollout_metrics(grouped_results, rollout_id, prefix="CB")

    return RolloutFnTrainOutput(samples=grouped_results, metrics=metrics)


def _cooperbench_eval_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
) -> RolloutFnEvalOutput:
    """Run CooperBench eval (placeholder — full eval TBD)."""
    logger.warning("[CB] Eval not yet implemented, returning empty")
    return RolloutFnEvalOutput(data={})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_rollout_metrics(
    grouped_results: List[List[Sample]],
    rollout_id: int,
    prefix: str = "CB",
) -> dict:
    """Compute and log rollout metrics."""
    all_samples = [s for group in grouped_results for s in group]
    rewards = [s.reward for s in all_samples if s.reward is not None]
    turn_counts = [
        s.metadata.get("turn_count", 0)
        for s in all_samples
        if hasattr(s, "metadata") and s.metadata
    ]
    response_lengths = [s.response_length for s in all_samples]
    truncated = [s.status == Sample.Status.TRUNCATED for s in all_samples]
    failed = [s.status == Sample.Status.FAILED for s in all_samples]

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

    for i, s in enumerate(all_samples[:3]):
        meta = s.metadata or {}
        logger.info(
            f"[{prefix}] Sample {s.index}: "
            f"reward={s.reward:.3f}, turns={meta.get('turn_count', '?')}, "
            f"tokens={len(s.tokens)}, resp_len={s.response_length}, "
            f"status={s.status}, response={s.response[:100]}..."
        )

    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_sample(sample: Sample, error: str) -> Sample:
    """Create a failed sample."""
    return Sample(
        index=sample.index,
        prompt=sample.prompt,
        tokens=[],
        response="",
        reward=0.0,
        loss_mask=[],
        response_length=0,
        metadata={"error": error},
    )
