"""Tau2-bench reward computation for training rollouts.

Converts usim conversation messages to tau2 format, creates a SimulationRun,
and calls tau2's evaluate_simulation for reward. All evaluators (ENV, COMMUNICATE)
are local and fast — no LLM calls.
"""

import logging
from typing import Any, Dict, List

from usim.core.types import TrajectoryStatus

logger = logging.getLogger(__name__)


def compute_tau2_reward(
    tau2_task: Any,
    messages: List[Dict[str, Any]],
    domain: str,
    status: TrajectoryStatus,
) -> float:
    """Compute reward for a tau2-bench rollout.

    Args:
        tau2_task: The tau2 Task object (from tau2.data_model.tasks)
        messages: Conversation messages in usim dict format
        domain: tau2 domain (retail, airline, telecom)
        status: Trajectory status from orchestrator

    Returns:
        Reward float in [0.0, 1.0]
    """
    try:
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.utils.utils import get_now
    except ImportError:
        logger.warning("tau2-bench not available, returning reward=0.0")
        return 0.0

    # Map trajectory status to tau2 termination reason
    if status == TrajectoryStatus.COMPLETED:
        termination_reason = TerminationReason.AGENT_STOP
    elif status == TrajectoryStatus.TRUNCATED:
        termination_reason = TerminationReason.MAX_STEPS
    else:
        termination_reason = TerminationReason.AGENT_ERROR

    # Convert usim messages to tau2 message types
    tau2_messages = _convert_messages(messages)

    try:
        simulation = SimulationRun(
            id="rollout",
            task_id=tau2_task.id,
            start_time=get_now(),
            end_time=get_now(),
            duration=0.0,
            termination_reason=termination_reason,
            messages=tau2_messages,
        )

        reward_info = evaluate_simulation(
            simulation=simulation,
            task=tau2_task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=domain,
        )

        logger.info(
            f"tau2 reward for task {tau2_task.id}: {reward_info.reward:.3f} "
            f"(breakdown: {reward_info.reward_breakdown})"
        )
        return reward_info.reward

    except Exception as e:
        logger.error(f"tau2 evaluation failed for task {tau2_task.id}: {e}")
        return 0.0


def _convert_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """Convert usim message dicts to tau2 message types.

    Args:
        messages: usim conversation messages (OpenAI dict format)

    Returns:
        List of tau2 message objects
    """
    from tau2.data_model.message import (
        AssistantMessage,
        ToolCall as Tau2ToolCall,
        ToolMessage as Tau2ToolMessage,
        UserMessage,
    )

    tau2_msgs = []
    for msg in messages:
        role = msg.get("role")

        if role == "assistant":
            tool_calls = None
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = [
                    Tau2ToolCall(
                        id=tc.get("id", ""),
                        name=(
                            tc["function"]["name"]
                            if "function" in tc
                            else tc.get("name", "")
                        ),
                        arguments=(
                            tc["function"]["arguments"]
                            if "function" in tc
                            else tc.get("arguments", {})
                        ),
                        requestor="assistant",
                    )
                    for tc in msg["tool_calls"]
                ]
            tau2_msgs.append(
                AssistantMessage(
                    content=msg.get("content"),
                    tool_calls=tool_calls,
                )
            )

        elif role == "user":
            tau2_msgs.append(UserMessage(content=msg.get("content")))

        elif role == "tool":
            tau2_msgs.append(
                Tau2ToolMessage(
                    id=msg.get("tool_call_id", ""),
                    content=msg.get("content"),
                    requestor="assistant",
                )
            )

    return tau2_msgs
