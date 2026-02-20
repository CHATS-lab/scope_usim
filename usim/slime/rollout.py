"""Custom rollout function for Slime training.

This module provides the entry point for Slime's training loop,
integrating usim's orchestrator with Slime's data pipeline.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from usim.core.orchestrator import UserSimOrchestrator
from usim.core.types import TrainableRole, UserSimConfig
from usim.slime.model_adapter import SlimeModelAdapter, create_slime_model_adapter
from usim.slime.trajectory_converter import trajectory_to_slime_sample


logger = logging.getLogger(__name__)


async def usim_generate_rollout(
    args: Any,
    sample: Any,
    sampling_params: Dict[str, Any],
) -> List[Any]:
    """Generate rollout using usim orchestrator for Slime training.

    This is the custom rollout function that Slime calls during training.
    It creates the necessary adapters, orchestrator, and runs a session.

    Args:
        args: Slime training arguments namespace
        sample: Slime Sample with task information
        sampling_params: Sampling parameters for generation

    Returns:
        List of Slime Sample objects with trajectories
    """
    try:
        # Agent model = trainable, served by SGLang
        agent_model = create_slime_model_adapter(args)

        # User sim model = fixed opponent via API (or same SGLang model if not configured)
        fixed_usim_models = getattr(args, "usim_fixed_opponent_model", None)
        if fixed_usim_models:
            from usim.core.api_model_adapter import create_openai_model_adapter

            base_url = getattr(args, "usim_fixed_opponent_base_url", "https://api.openai.com/v1")
            api_key_var = getattr(args, "usim_fixed_opponent_api_key_var", "OPENAI_API_KEY")

            # Support model rotation: comma-separated list of models
            model_list = [m.strip() for m in fixed_usim_models.split(",") if m.strip()]
            model_idx = sample.index % len(model_list)
            model_name = model_list[model_idx]

            # Support per-model base URLs and API keys (comma-separated, maps 1:1 to models)
            base_url_list = [u.strip() for u in base_url.split(",") if u.strip()]
            api_key_var_list = [k.strip() for k in api_key_var.split(",") if k.strip()]
            model_base_url = base_url_list[model_idx % len(base_url_list)]
            model_api_key_var = api_key_var_list[model_idx % len(api_key_var_list)]
            api_key = os.environ.get(model_api_key_var)

            user_model = create_openai_model_adapter(
                model_name=model_name,
                tokenizer=agent_model.tokenizer,
                base_url=model_base_url,
                api_key=api_key,
            )
            logger.info(
                f"Using fixed user sim: {model_name} "
                f"(sample {sample.index}, {len(model_list)} model(s) in rotation)"
            )
        else:
            user_model = create_slime_model_adapter(args)

        # Get configuration from args
        trainable_role = TrainableRole(getattr(args, "trainable_role", "agent"))
        max_turns = getattr(args, "max_turns", 30)
        max_tokens = getattr(args, "max_tokens", 2048)

        config = UserSimConfig(
            trainable_role=trainable_role,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=sampling_params.get("temperature", 0.7),
        )

        # Create orchestrator
        orchestrator = UserSimOrchestrator(
            agent_model=agent_model,
            user_model=user_model,
            config=config,
        )

        # Convert sample to task format
        task = sample_to_task(sample, args)

        # Create agent and user simulator
        from usim.core.agent import LLMAgent
        from usim.core.user_simulator import LLMUserSimulator

        agent = LLMAgent(
            model=agent_model,
            config=config,
            domain_policy=task.get("domain_policy", ""),
        )

        user_simulator = LLMUserSimulator(
            model=user_model,
            config=config,
            instructions=task.get("instructions", ""),
        )

        # Run session
        trajectory = await orchestrator.run_session(task, agent, user_simulator)

        # Convert to Slime sample
        output_sample = trajectory_to_slime_sample(trajectory, sample.index)

        return [output_sample]

    except Exception as e:
        logger.error(f"usim rollout failed: {e}")
        # Return empty sample on error
        try:
            from slime.data.types import Sample
            return [Sample(
                index=sample.index,
                prompt=sample.prompt,
                tokens=[],
                response="",
                reward=0.0,
                loss_mask=[],
                response_length=0,
                metadata={"error": str(e)},
            )]
        except ImportError:
            raise


def sample_to_task(sample: Any, args: Any) -> Dict[str, Any]:
    """Convert Slime Sample to usim task format.

    Args:
        sample: Slime Sample object
        args: Training arguments

    Returns:
        Task dictionary for orchestrator
    """
    metadata = getattr(sample, "metadata", {}) or {}

    return {
        "id": metadata.get("task_id", str(sample.index)),
        "domain": metadata.get("domain", getattr(args, "domain", "retail")),
        "instructions": sample.prompt,
        "domain_policy": metadata.get("domain_policy", ""),
        "initial_state": metadata.get("initial_state"),
    }


def add_usim_arguments(parser: Any) -> None:
    """Add usim-specific arguments to Slime argument parser.

    Args:
        parser: argparse ArgumentParser
    """
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
        "--user-model",
        type=str,
        default=None,
        help="Optional separate model for user simulator",
    )


def create_rollout_function(args: Any) -> callable:
    """Create a configured rollout function for Slime.

    Args:
        args: Training arguments

    Returns:
        Async rollout function
    """
    async def rollout_fn(sample: Any, sampling_params: Dict[str, Any]) -> List[Any]:
        return await usim_generate_rollout(args, sample, sampling_params)

    return rollout_fn
