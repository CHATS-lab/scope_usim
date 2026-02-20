"""Slime rollout entry point for tau2-bench training.

Thin glue layer: creates the tau2 environment, sglang generate function,
and calls the generic orchestrator. Converts the resulting Trajectory
to Slime's Sample format.

All tau2-specific logic (observation conversion, tool parsing, prompt
postprocessing) lives in usim.core.environment.tau2.
"""

import logging
import os
from typing import Any, Dict, List

from slime.data.types import Sample
from slime.utils.http_utils import post
from slime.utils.processing_utils import load_tokenizer
from tau2.run import get_tasks

from usim.core.environment.tau2 import Tau2Environment
from usim.core.orchestrator import UserSimOrchestrator
from usim.core.types import TrainableRole, UserSimConfig
from usim.slime.trajectory_converter import trajectory_to_slime_sample


logger = logging.getLogger(__name__)


def _get_user_model_config(
    args: Any, sample: Any
) -> tuple:
    """Get user model configuration from metadata overrides or CLI args.

    Eval datasets specify per-sample overrides via metadata_overrides in YAML.
    Training uses CLI args (with model rotation support).

    Returns:
        (model_name, base_url, api_key) tuple
    """
    metadata = getattr(sample, "metadata", {}) or {}

    # Check metadata overrides first (for eval)
    fixed_model = metadata.get("usim_fixed_opponent_model")
    base_url = metadata.get(
        "usim_fixed_opponent_base_url",
        getattr(args, "usim_fixed_opponent_base_url", "https://api.openai.com/v1"),
    )
    api_key_var = metadata.get(
        "usim_fixed_opponent_api_key_var",
        getattr(args, "usim_fixed_opponent_api_key_var", "OPENAI_API_KEY"),
    )

    # Fall back to CLI args for training
    if not fixed_model:
        fixed_model = getattr(args, "usim_fixed_opponent_model", None)

    if not fixed_model:
        return None, base_url, None

    # Model rotation: comma-separated list
    model_list = [m.strip() for m in fixed_model.split(",") if m.strip()]
    model_idx = sample.index % len(model_list)
    model_name = model_list[model_idx]

    # Per-model base URL and API key (comma-separated, maps 1:1)
    base_url_list = [u.strip() for u in base_url.split(",") if u.strip()]
    api_key_var_list = [k.strip() for k in api_key_var.split(",") if k.strip()]
    model_base_url = base_url_list[model_idx % len(base_url_list)]
    model_api_key_var = api_key_var_list[model_idx % len(api_key_var_list)]
    api_key = os.environ.get(model_api_key_var)

    logger.info(
        f"User sim: {model_name} (sample {sample.index}, "
        f"{len(model_list)} model(s) in rotation)"
    )
    return model_name, model_base_url, api_key


async def usim_generate_rollout(
    args: Any,
    sample: Any,
    sampling_params: Dict[str, Any],
) -> List[Any]:
    """Generate rollout using Gym-based orchestration for Slime training.

    Args:
        args: Slime training arguments namespace
        sample: Slime Sample with task information
        sampling_params: Sampling parameters for generation

    Returns:
        List of Slime Sample objects with trajectories
    """
    try:
        # --- Task setup ---
        metadata = getattr(sample, "metadata", {}) or {}
        domain = metadata.get("domain", getattr(args, "usim_domain", "retail"))
        task = metadata.get("tau2_task")
        if task is None:
            task_split = metadata.get("task_split", "train")
            tasks = get_tasks(task_set_name=domain, task_split_name=task_split)
            task_index = int(sample.prompt) if sample.prompt.isdigit() else sample.index
            task = tasks[task_index]

        # --- User model config ---
        user_model, user_base_url, user_api_key = _get_user_model_config(args, sample)
        max_turns = getattr(args, "max_turns", 30)

        # --- Create environment ---
        env = Tau2Environment(
            domain=domain,
            task_id=task.id,
            user_model=user_model,
            user_base_url=user_base_url,
            user_api_key=user_api_key,
            max_turns=max_turns,
        )

        # --- Create generate function (TITO: input_ids → token_ids + logprobs) ---
        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )

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

        # --- Load tokenizer ---
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

        # --- Create orchestrator and run ---
        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=max_turns,
            max_tokens=getattr(args, "max_tokens", 2048),
            max_context_length=getattr(args, "rollout_max_response_len", 16384),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = UserSimOrchestrator(tokenizer=tokenizer, config=config)
        trajectory = await orchestrator.rollout(env, generate_fn, sampling_params)

        logger.info(
            f"tau2 rollout {sample.index}: {trajectory.turn_count} turns, "
            f"reward={trajectory.reward:.3f}, status={trajectory.status}"
            + (f", usim={user_model}" if user_model else "")
        )

        # Convert to Slime sample
        output_sample = trajectory_to_slime_sample(trajectory, sample.index)
        return [output_sample]

    except Exception as e:
        logger.error(f"usim rollout failed: {e}", exc_info=True)
        return [
            Sample(
                index=sample.index,
                prompt=sample.prompt,
                tokens=[],
                response="",
                reward=0.0,
                loss_mask=[],
                response_length=0,
                metadata={"error": str(e)},
            )
        ]


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
