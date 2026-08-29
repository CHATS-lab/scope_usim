"""Async rollout entry point for Tinker training.

This module provides the interface to UserSimOrchestrator for Tinker training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from usim.core.orchestrator import UserSimOrchestrator
from usim.core.types import TrainableRole, UserSimConfig
from usim.tinker.model_adapter import TinkerModelAdapter, create_tinker_model_adapter
from usim.tinker.trajectory_converter import usim_trajectory_to_tinker_trajectory

logger = logging.getLogger(__name__)


# Global state (persists across rollouts within a training run)
_ORCHESTRATOR: Optional[UserSimOrchestrator] = None


def get_orchestrator(
    sampling_client: Any,
    tokenizer: Any,
    renderer: Any,
    config: UserSimConfig,
    use_weave: bool = False,
) -> UserSimOrchestrator:
    """Get or create the UserSimOrchestrator (reuses across rollouts).

    Args:
        sampling_client: Tinker SamplingClient for generation
        tokenizer: Tokenizer instance
        renderer: Renderer for formatting prompts
        config: User simulator configuration
        use_weave: Whether to enable Weave tracing

    Returns:
        UserSimOrchestrator instance
    """
    global _ORCHESTRATOR

    if _ORCHESTRATOR is None:
        model_adapter = create_tinker_model_adapter(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            renderer=renderer,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            use_weave=use_weave,
        )

        _ORCHESTRATOR = UserSimOrchestrator(
            agent_model=model_adapter,
            user_model=model_adapter,
            config=config,
        )

    return _ORCHESTRATOR


async def usim_generate_rollout(
    sampling_client: Any,
    current_step: int,
    config: UserSimConfig,
    renderer: Any,
    tokenizer: Any,
    task: Dict[str, Any],
    use_weave: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Perform usim rollout and return Tinker TrajectoryGroups with metrics.

    This is the main entry point for Tinker training. It:
    1. Gets/creates the orchestrator
    2. Runs a session
    3. Converts results to Tinker format
    4. Computes rollout metrics

    Args:
        sampling_client: Tinker SamplingClient for generation
        current_step: Current training step
        config: User simulator configuration
        renderer: Tinker renderer
        tokenizer: Tokenizer
        task: Task specification
        use_weave: Whether to enable Weave tracing

    Returns:
        Tuple of (List[TrajectoryGroup], metrics_dict)
    """
    orchestrator = get_orchestrator(
        sampling_client, tokenizer, renderer, config, use_weave
    )

    # Create agent and user simulator
    from usim.core.agent.llm_agent import LLMAgent
    from usim.core.user_simulator.llm_user import LLMUserSimulator

    agent = LLMAgent(
        model=orchestrator.agent_model,
        config=config,
        domain_policy=task.get("domain_policy", ""),
    )

    user_simulator = LLMUserSimulator(
        model=orchestrator.user_model,
        config=config,
        instructions=task.get("instructions", ""),
    )

    # Run session
    trajectory = await orchestrator.run_session(task, agent, user_simulator)

    # Convert to Tinker format
    try:
        from tinker_cookbook.rl.types import TrajectoryGroup

        role = task.get("role", "actor")
        tinker_traj = usim_trajectory_to_tinker_trajectory(trajectory, role=role)

        trajectory_group = TrajectoryGroup(
            trajectories_G=[tinker_traj],
            final_rewards_G=[trajectory.reward or 0.0],
            metrics_G=[{
                "role": role,
                "task_id": task.get("id", ""),
                "turn_count": trajectory.turn_count,
            }],
        )

        trajectory_groups = [trajectory_group]
    except ImportError:
        logger.warning("tinker_cookbook not available, returning raw trajectory")
        trajectory_groups = [trajectory]

    # Compute metrics
    metrics: Dict[str, Any] = {
        "rollout/num_trajectories": 1,
        "rollout/mean_return": trajectory.reward,
        "rollout/turn_count": trajectory.turn_count,
        "rollout/response_length": trajectory.response_length,
        "rollout/status": trajectory.status.value,
        "rollout/step": current_step,
    }

    logger.info(
        f"[ROLLOUT {current_step}] reward={trajectory.reward:.3f}, "
        f"turns={trajectory.turn_count}, status={trajectory.status.value}"
    )

    return trajectory_groups, metrics
