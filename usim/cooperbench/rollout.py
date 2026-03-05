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
from usim.core.rollout_metrics import compute_rollout_metrics
from usim.core.environment.cooperbench.messaging import MessagingConnector
from usim.core.types import TrainableRole, UserSimConfig
from usim.slime.model_adapter import create_slime_model_adapter
from usim.slime.trajectory_converter import trajectory_to_slime_sample
from usim.utils.docent import get_docent_uploader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sample rollout functions
# ---------------------------------------------------------------------------


def _get_or_create_model_adapter(args: Any) -> Any:
    """Get or create a shared SlimeModelAdapter (cached on args)."""
    if not hasattr(args, "_cached_model_adapter"):
        args._cached_model_adapter = create_slime_model_adapter(args)
    return args._cached_model_adapter


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
        tool_call_parser = getattr(args, "cooperbench_tool_call_parser", "qwen")

        # Create environment and pre-start sandbox (pull image, start container)
        environment = CooperBenchEnvironment(image_name=image_name, timeout=3600)
        await environment.start()

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

        model_adapter = _get_or_create_model_adapter(args)

        per_turn_tokens = getattr(args, "cooperbench_max_tokens_per_turn", 4096)
        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_steps,
            max_tokens=per_turn_tokens,
            max_context_length=getattr(args, "cooperbench_max_context_length", 65536),
            temperature=sampling_params.get("temperature", 0.7),
        )

        chat_template_kwargs = {}
        if getattr(args, "cooperbench_enable_thinking", None) is not None:
            chat_template_kwargs["enable_thinking"] = args.cooperbench_enable_thinking
        else:
            chat_template_kwargs["enable_thinking"] = False

        max_tool_chars = getattr(args, "cooperbench_max_tool_output_chars", 4000)
        orchestrator = CodingAgentOrchestrator(
            agent_model=model_adapter,
            config=config,
            environment=environment,
            tool_call_parser=tool_call_parser,
            chat_template_kwargs=chat_template_kwargs,
            max_tool_output_chars=max_tool_chars,
        )

        trajectory = await orchestrator.run_session(task, agent)

        # Get patch and compute reward (runs in thread pool, doesn't block event loop)
        agent_patch = await environment.get_patch()
        patch_bonus = getattr(args, "cooperbench_patch_bonus", 0.1)

        if not agent_patch.strip():
            reward = 0.0
        else:
            reward = await compute_baseline_reward_async(
                repo_name=repo,
                task_id=task_id,
                feature_id=agent_feature_id,
                patch=agent_patch,
                dataset_dir=dataset_dir,
                backend=backend,
            )
            if reward == 0.0 and patch_bonus > 0:
                reward = patch_bonus  # Non-empty patch but tests fail

        trajectory.reward = reward
        trajectory.metadata["agent_patch_lines"] = len(agent_patch.splitlines())
        trajectory.metadata["has_patch"] = bool(agent_patch.strip())
        trajectory.metadata["setting"] = "baseline"

        logger.info(
            f"[CB] Baseline {sample.index}: {trajectory.turn_count} steps, "
            f"reward={reward:.2f}, patch_lines={len(agent_patch.splitlines())}, "
            f"status={trajectory.status}"
        )

        return trajectory_to_slime_sample(trajectory, sample.index)

    except Exception as e:
        import traceback
        err_msg = f"Baseline rollout failed (sample {sample.index}): {e}"
        logger.error(err_msg, exc_info=True)
        print(f"[CB ERROR] {err_msg}", flush=True)
        traceback.print_exc()
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
        tool_call_parser = getattr(args, "cooperbench_tool_call_parser", "qwen")
        f1_id, f2_id = feature_ids

        # Create environment and pre-start sandbox
        environment = CooperBenchEnvironment(image_name=image_name, timeout=3600)
        await environment.start()

        # Create agent with combined prompt for both features
        agent = CooperBenchAgent(agent_id="agent1", setting="solo")

        task = {
            "id": f"{repo}/task{task_id}/f{f1_id}_f{f2_id}",
            "repo": repo,
            "task_id": task_id,
            "feature_id": f"{f1_id}_{f2_id}",
            "instructions": agent.get_solo_task_message(descriptions),
        }

        model_adapter = _get_or_create_model_adapter(args)

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_steps,
            max_tokens=getattr(args, "rollout_max_response_len", 8192),
            max_context_length=getattr(args, "cooperbench_max_context_length", 65536),
            temperature=sampling_params.get("temperature", 0.7),
        )

        chat_template_kwargs = {}
        if getattr(args, "cooperbench_enable_thinking", None) is not None:
            chat_template_kwargs["enable_thinking"] = args.cooperbench_enable_thinking
        else:
            chat_template_kwargs["enable_thinking"] = False

        orchestrator = CodingAgentOrchestrator(
            agent_model=model_adapter,
            config=config,
            environment=environment,
            tool_call_parser=tool_call_parser,
            chat_template_kwargs=chat_template_kwargs,
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
        import traceback
        err_msg = f"Solo rollout failed (sample {sample.index}): {e}"
        logger.error(err_msg, exc_info=True)
        print(f"[CB ERROR] {err_msg}", flush=True)
        traceback.print_exc()
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
        tool_call_parser = getattr(args, "cooperbench_tool_call_parser", "qwen")
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

        # Create environment and pre-start sandbox
        environment = CooperBenchEnvironment(
            image_name=image_name,
            messaging=messaging,
            messaging_enabled=True,
            timeout=3600,
        )
        await environment.start()

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

        model_adapter = _get_or_create_model_adapter(args)

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_steps,
            max_tokens=getattr(args, "rollout_max_response_len", 8192),
            max_context_length=getattr(args, "cooperbench_max_context_length", 65536),
            temperature=sampling_params.get("temperature", 0.7),
        )

        chat_template_kwargs = {}
        if getattr(args, "cooperbench_enable_thinking", None) is not None:
            chat_template_kwargs["enable_thinking"] = args.cooperbench_enable_thinking
        else:
            chat_template_kwargs["enable_thinking"] = False

        orchestrator = CodingAgentOrchestrator(
            agent_model=model_adapter,
            config=config,
            environment=environment,
            messaging=messaging,
            tool_call_parser=tool_call_parser,
            chat_template_kwargs=chat_template_kwargs,
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
        import traceback
        err_msg = f"Coop rollout failed (sample {sample.index}): {e}"
        logger.error(err_msg, exc_info=True)
        print(f"[CB ERROR] {err_msg}", flush=True)
        traceback.print_exc()
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


async def _run_single_with_timeout(
    fn: Any,
    args: Any,
    sample: Sample,
    sampling_params: Dict[str, Any],
    timeout: int,
) -> Sample:
    """Run a single rollout with a timeout. Returns error sample on timeout."""
    try:
        return await asyncio.wait_for(
            fn(args, sample, sampling_params),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"Sample {sample.index} timed out after {timeout}s"
        )
        return _error_sample(sample, f"Timed out after {timeout}s")


def _preflight_check(args: Namespace) -> None:
    """Quick check that Modal auth + model adapter work before running full batch.

    Raises on failure so the error is visible before we launch 64 concurrent tasks.
    """
    import os

    print("[CB] === Pre-flight checks ===", flush=True)

    # 1. Check Modal auth
    print(f"[CB] MODAL_TOKEN_ID set: {bool(os.environ.get('MODAL_TOKEN_ID'))}", flush=True)
    print(f"[CB] MODAL_TOKEN_SECRET set: {bool(os.environ.get('MODAL_TOKEN_SECRET'))}", flush=True)
    try:
        import modal
        print(f"[CB] modal version: {modal.__version__}", flush=True)
        app = modal.App.lookup("cooperbench", create_if_missing=True)
        print(f"[CB] Modal auth OK, app={app}", flush=True)
    except Exception as e:
        print(f"[CB] Modal auth FAILED: {e}", flush=True)
        raise RuntimeError(f"Modal pre-flight failed: {e}") from e

    # 2. Check model adapter
    try:
        adapter = _get_or_create_model_adapter(args)
        print(
            f"[CB] Model adapter OK: {adapter.router_ip}:{adapter.router_port}",
            flush=True,
        )
    except Exception as e:
        print(f"[CB] Model adapter FAILED: {e}", flush=True)
        raise RuntimeError(f"Model adapter pre-flight failed: {e}") from e

    # 3. Check sglang function call parser
    try:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        print("[CB] FunctionCallParser import OK", flush=True)
    except Exception as e:
        print(f"[CB] FunctionCallParser import FAILED: {e}", flush=True)
        raise RuntimeError(f"FunctionCallParser pre-flight failed: {e}") from e

    print("[CB] === Pre-flight checks PASSED ===", flush=True)


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

    # Per-sample timeout: sandbox timeout (3600) + buffer for reward eval
    sample_timeout = getattr(args, "cooperbench_sample_timeout", 5400)

    # Limit concurrent sessions to avoid SGLang OOM from too many active KV caches.
    # Each multi-turn session accumulates 20-30k+ tokens; 64 concurrent is too much for 27B.
    max_concurrent = getattr(args, "cooperbench_max_concurrent", 16)
    semaphore = asyncio.Semaphore(max_concurrent)

    dispatch = {
        "baseline": _baseline_single,
        "solo": _solo_single,
        "coop": _coop_single,
    }
    fn = dispatch[setting]

    async def _run_with_semaphore(fn, args, sample, sampling_params, timeout):
        async with semaphore:
            return await _run_single_with_timeout(fn, args, sample, sampling_params, timeout)

    tasks = []
    for group in samples:
        for sample in group:
            tasks.append(
                _run_with_semaphore(fn, args, sample, sampling_params, sample_timeout)
            )

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

    # Run pre-flight checks on first rollout only
    if rollout_id == 0:
        _preflight_check(args)

    samples = data_source.get_samples(args.rollout_batch_size)
    grouped_results = asyncio.run(_run_batch_async(args, samples))

    metrics = compute_rollout_metrics(grouped_results, rollout_id, prefix="CB")

    # Upload to Docent (non-blocking)
    docent_uploader = get_docent_uploader(args)
    if docent_uploader:
        all_samples = [s for group in grouped_results for s in group]
        docent_uploader.upload_batch(all_samples, rollout_id)

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
