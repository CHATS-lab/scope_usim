"""Slime rollout entry point for tau2-bench training.

Batch-level rollout function following spare's pattern:
1. Gets samples from data_source
2. Runs per-sample async rollouts via asyncio
3. Returns RolloutFnTrainOutput with grouped samples

All tau2-specific logic (observation conversion, tool parsing, prompt
postprocessing) lives in usim.core.environment.tau2.
"""

import asyncio
import logging
import os
from argparse import Namespace
from typing import Any, Callable, Dict, List

import weave
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.http_utils import post
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample
from tau2.run import get_tasks

from usim.core.environment.tau2 import Tau2Environment
from usim.core.orchestrator import UserSimOrchestrator
from usim.core.types import TrainableRole, UserSimConfig
from usim.slime.trajectory_converter import trajectory_to_slime_sample


logger = logging.getLogger(__name__)


def _get_user_model_config(
    args: Any, sample: Any
) -> tuple:
    """Get user model configuration from metadata overrides or CLI args."""
    metadata = getattr(sample, "metadata", {}) or {}

    fixed_model = metadata.get("usim_fixed_opponent_model")
    base_url = metadata.get(
        "usim_fixed_opponent_base_url",
        getattr(args, "usim_fixed_opponent_base_url", "https://api.openai.com/v1"),
    )
    api_key_var = metadata.get(
        "usim_fixed_opponent_api_key_var",
        getattr(args, "usim_fixed_opponent_api_key_var", "OPENAI_API_KEY"),
    )

    if not fixed_model:
        fixed_model = getattr(args, "usim_fixed_opponent_model", None)

    if not fixed_model:
        return None, base_url, None

    model_list = [m.strip() for m in fixed_model.split(",") if m.strip()]
    model_idx = sample.index % len(model_list)
    model_name = model_list[model_idx]

    base_url_list = [u.strip() for u in base_url.split(",") if u.strip()]
    api_key_var_list = [k.strip() for k in api_key_var.split(",") if k.strip()]
    model_base_url = base_url_list[model_idx % len(base_url_list)]
    model_api_key_var = api_key_var_list[model_idx % len(api_key_var_list)]
    api_key = os.environ.get(model_api_key_var)

    return model_name, model_base_url, api_key


async def _usim_generate_single(
    args: Any,
    sample: Any,
    sampling_params: Dict[str, Any],
    **kwargs: Any,
) -> List[Sample]:
    """Per-sample async generate function for tau2-bench."""
    try:
        metadata = getattr(sample, "metadata", {}) or {}
        domain = metadata.get("domain", getattr(args, "usim_domain", "retail"))
        task = metadata.get("tau2_task")
        if task is None:
            task_split = metadata.get("task_split", "train")
            tasks = get_tasks(task_set_name=domain, task_split_name=task_split)
            task_index = int(sample.prompt) if sample.prompt.isdigit() else sample.index
            task = tasks[task_index]

        user_model, user_base_url, user_api_key = _get_user_model_config(args, sample)
        max_turns = getattr(args, "max_turns", 30)

        env = Tau2Environment(
            domain=domain,
            task_id=task.id,
            user_model=user_model,
            user_base_url=user_base_url,
            user_api_key=user_api_key,
            max_turns=max_turns,
        )

        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )

        @weave.op
        async def generate_fn(
            input_ids: list, samp_params: Dict[str, Any]
        ) -> Dict[str, Any]:
            output = await post(sglang_url, {
                "input_ids": input_ids,
                "sampling_params": samp_params,
                "return_logprob": True,
            })
            meta = output.get("meta_info", {})
            token_logprobs = meta.get("output_token_logprobs", [])
            return {
                "text": output["text"],
                "token_ids": [item[1] for item in token_logprobs],
                "logprobs": [item[0] for item in token_logprobs],
                "meta_info": meta,
            }

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_turns,
            max_tokens=getattr(args, "max_tokens", 8192),
            max_context_length=getattr(args, "rollout_max_response_len", 32768),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = UserSimOrchestrator(tokenizer=tokenizer, config=config)
        trajectory = await orchestrator.rollout(env, generate_fn, sampling_params)

        logger.info(
            f"tau2 rollout {sample.index}: {trajectory.turn_count} turns, "
            f"reward={trajectory.reward:.3f}, status={trajectory.status}"
            + (f", usim={user_model}" if user_model else "")
        )

        output_sample = trajectory_to_slime_sample(trajectory, sample.index)
        return output_sample

    except Exception as e:
        logger.error(f"usim rollout failed (sample {sample.index}): {e}", exc_info=True)
        return Sample(
            index=sample.index,
            prompt=sample.prompt,
            tokens=[],
            response="",
            reward=0.0,
            loss_mask=[],
            response_length=0,
            metadata={"error": str(e)},
        )


async def _run_batch_async(
    args: Namespace,
    samples: List[List[Sample]],
) -> List[List[Sample]]:
    """Run a batch of samples concurrently, returning grouped results."""
    sampling_params = dict(
        temperature=args.rollout_temperature,
        top_p=getattr(args, "rollout_top_p", 0.95),
        top_k=getattr(args, "rollout_top_k", -1),
        max_new_tokens=8192,
    )

    tasks = []
    for group in samples:
        for sample in group:
            tasks.append(_usim_generate_single(args, sample, sampling_params))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Re-group by n_samples_per_prompt
    grouped = []
    idx = 0
    for group in samples:
        group_results = []
        for _ in group:
            result = results[idx]
            if isinstance(result, Exception):
                logger.error(f"Sample failed: {result}")
                result = Sample(
                    index=group[0].index,
                    tokens=[],
                    response="",
                    reward=0.0,
                    loss_mask=[],
                    response_length=0,
                    metadata={"error": str(result)},
                )
            group_results.append(result)
            idx += 1
        grouped.append(group_results)

    return grouped


async def _usim_eval_single_dataset(
    args: Namespace,
    dataset_cfg: Any,
) -> dict:
    """Run eval for a single dataset config."""
    metadata_overrides = getattr(dataset_cfg, "metadata_overrides", {}) or {}
    domain = metadata_overrides.get("domain", getattr(args, "usim_domain", "retail"))
    task_split = metadata_overrides.get("task_split", "test")
    max_tasks = metadata_overrides.get("max_tasks", 50)

    tasks_list = get_tasks(task_set_name=domain, task_split_name=task_split)
    if max_tasks:
        tasks_list = tasks_list[:max_tasks]

    n_samples = getattr(dataset_cfg, "n_samples_per_eval_prompt", 1)
    sampling_params = dict(
        temperature=getattr(dataset_cfg, "temperature", 0.7),
        top_p=getattr(dataset_cfg, "top_p", 0.95),
        top_k=getattr(dataset_cfg, "top_k", -1),
        max_new_tokens=getattr(dataset_cfg, "max_response_len", None)
            or args.rollout_max_response_len,
    )

    # Override user model config from metadata
    eval_args = Namespace(**vars(args))
    if "usim_fixed_opponent_model" in metadata_overrides:
        eval_args.usim_fixed_opponent_model = metadata_overrides["usim_fixed_opponent_model"]
    if "usim_fixed_opponent_base_url" in metadata_overrides:
        eval_args.usim_fixed_opponent_base_url = metadata_overrides["usim_fixed_opponent_base_url"]
    if "usim_fixed_opponent_api_key_var" in metadata_overrides:
        eval_args.usim_fixed_opponent_api_key_var = metadata_overrides["usim_fixed_opponent_api_key_var"]

    coros = []
    for i, task in enumerate(tasks_list):
        for j in range(n_samples):
            sample = Sample(
                index=i * n_samples + j,
                prompt=str(task.id),
                metadata={
                    "tau2_task": task,
                    "domain": domain,
                    "task_split": task_split,
                },
            )
            coros.append(_usim_generate_single(eval_args, sample, sampling_params))

    results = await asyncio.gather(*coros, return_exceptions=True)

    data = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Eval sample failed: {r}")
            data.append(Sample(index=0, reward=0.0))
        else:
            data.append(r)

    reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", None)
    return {
        dataset_cfg.name: {
            "rewards": [
                s.reward if not reward_key else s.reward[reward_key]
                for s in data
            ],
            "truncated": [s.status == Sample.Status.TRUNCATED for s in data],
            "samples": data,
        }
    }


def _usim_eval_rollout(
    args: Namespace,
    rollout_id: int,
) -> RolloutFnEvalOutput:
    """Run eval across all configured eval datasets."""
    eval_datasets = getattr(args, "eval_datasets", []) or []
    if not eval_datasets:
        logger.warning("[USIM] No eval datasets configured")
        return RolloutFnEvalOutput(data={})

    async def _run_all():
        coros = [_usim_eval_single_dataset(args, cfg) for cfg in eval_datasets]
        results_list = await asyncio.gather(*coros)
        results = {}
        for r in results_list:
            results.update(r)
        return results

    data = asyncio.run(_run_all())
    return RolloutFnEvalOutput(data=data)


def _compute_rollout_metrics(
    grouped_results: List[List[Sample]],
    rollout_id: int,
    prefix: str = "USIM",
) -> dict:
    """Compute and log rollout metrics including per-sample debug info."""
    import numpy as np

    all_samples = [s for group in grouped_results for s in group]
    rewards = [s.reward for s in all_samples if s.reward is not None]
    turn_counts = [
        s.metadata.get("turn_count", 0) for s in all_samples if hasattr(s, "metadata") and s.metadata
    ]
    response_lengths = [s.response_length for s in all_samples]
    truncated = [s.status == Sample.Status.TRUNCATED for s in all_samples]
    failed = [s.status == Sample.Status.FAILED for s in all_samples]

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

    # Debug: log first few samples
    for i, s in enumerate(all_samples[:3]):
        meta = s.metadata or {}
        logger.info(
            f"[{prefix}] Sample {s.index}: "
            f"reward={s.reward:.3f}, turns={meta.get('turn_count', '?')}, "
            f"tokens={len(s.tokens)}, resp_len={s.response_length}, "
            f"status={s.status}, response={s.response[:100]}..."
        )

    return metrics


def usim_generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Batch-level rollout function for tau2-bench.

    For eval: uses slime's eval_rollout with custom_generate_function_path.
    For training: gets samples from data_source, runs them concurrently,
    returns grouped samples for GRPO advantage calculation.
    """
    if evaluation:
        return _usim_eval_rollout(args, rollout_id)

    # Training: get samples from data_source, run rollouts, return grouped
    samples = data_source.get_samples(args.rollout_batch_size)
    grouped_results = asyncio.run(_run_batch_async(args, samples))

    # Compute rollout metrics
    metrics = _compute_rollout_metrics(grouped_results, rollout_id, prefix="USIM")

    return RolloutFnTrainOutput(samples=grouped_results, metrics=metrics)


def add_usim_arguments(parser: Any) -> None:
    """Add usim-specific arguments to Slime argument parser."""
    group = parser.add_argument_group("usim", "User Simulator arguments")

    group.add_argument(
        "--trainable-role",
        type=str,
        default="agent",
        choices=["agent", "user", "both"],
        help="Which role(s) to train: agent, user, or both",
    )
    group.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum conversation turns",
    )
    group.add_argument(
        "--usim-domain",
        type=str,
        default="retail",
        help="Domain for user simulation (retail, airline, telecom)",
    )
    group.add_argument(
        "--usim-fixed-opponent-model",
        type=str,
        default=None,
        help="Fixed opponent model(s), comma-separated for rotation",
    )
    group.add_argument(
        "--usim-fixed-opponent-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="API base URL(s), comma-separated to match model list",
    )
    group.add_argument(
        "--usim-fixed-opponent-api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        help="Env var(s) for API key, comma-separated to match model list",
    )
