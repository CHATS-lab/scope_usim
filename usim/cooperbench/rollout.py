"""Custom rollout function for CooperBench with Slime training.

Orchestrates a complete CooperBench rollout:
1. Trainable agent (SGLang) implements feature 1
2. Partner agent (API via CooperBench) implements feature 2
3. Merge + test determines reward
"""

import logging
import threading
import uuid
from typing import Any, Dict, List

from usim.cooperbench.agent import CooperBenchAgent
from usim.cooperbench.partner import run_partner_agent
from usim.cooperbench.reward import compute_reward
from usim.core.coding_orchestrator import CodingAgentOrchestrator
from usim.core.environment.cooperbench.environment import CooperBenchEnvironment
from usim.core.environment.cooperbench.messaging import MessagingConnector
from usim.core.types import TrainableRole, UserSimConfig
from usim.slime.model_adapter import create_slime_model_adapter
from usim.slime.trajectory_converter import trajectory_to_slime_sample

logger = logging.getLogger(__name__)


async def cooperbench_generate_rollout(
    args: Any,
    sample: Any,
    sampling_params: Dict[str, Any],
) -> List[Any]:
    """Generate rollout for CooperBench with Slime training.

    This is the custom rollout function that Slime calls during training.
    It creates the trainable agent + partner agent, runs both, evaluates
    the merge, and returns a training sample.

    Args:
        args: Slime training arguments namespace
        sample: Slime Sample with CooperBench task metadata
        sampling_params: Sampling parameters for generation

    Returns:
        List of Slime Sample objects with trajectory
    """
    environment = None
    partner_result = {"patch": ""}
    partner_error = {"error": None}

    try:
        # === Setup ===
        metadata = getattr(sample, "metadata", {}) or {}
        repo = metadata["repo"]
        task_id = metadata["task_id"]
        image_name = metadata["image_name"]
        agent_feature_id = metadata["agent_feature_id"]
        partner_feature_id = metadata["partner_feature_id"]
        feature_descriptions = metadata.get("feature_descriptions", {})

        partner_model = getattr(args, "cooperbench_partner_model", "gpt-5-mini")
        max_steps = getattr(args, "cooperbench_max_steps", 50)
        redis_url = getattr(args, "cooperbench_redis_url", "redis://localhost:6379")
        dataset_dir = getattr(args, "cooperbench_dataset_dir", None)

        # Create unique run ID for Redis namespacing
        run_id = uuid.uuid4().hex[:8]
        namespaced_redis = f"{redis_url}#run:{run_id}"

        agents = ["agent1", "agent2"]

        # === Agent model (trainable, SGLang) ===
        agent_model = create_slime_model_adapter(args)

        # === Config ===
        trainable_role = TrainableRole(getattr(args, "trainable_role", "agent"))
        config = UserSimConfig(
            trainable_role=trainable_role,
            max_turns=max_steps,
            max_tokens=getattr(args, "max_tokens", 4096),
            temperature=sampling_params.get("temperature", 0.7),
        )

        # === Messaging ===
        messaging = MessagingConnector(
            agent_id="agent1",
            agents=agents,
            url=namespaced_redis,
        )

        # === Start partner agent in background thread ===
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

        # === Create environment for trainable agent ===
        environment = CooperBenchEnvironment(
            image_name=image_name,
            messaging=messaging,
            timeout=3600,
        )

        # === Create agent ===
        agent = CooperBenchAgent(
            agent_id="agent1",
            agents=agents,
            messaging_enabled=True,
        )

        # === Task ===
        task_description = feature_descriptions.get(
            agent_feature_id,
            feature_descriptions.get(str(agent_feature_id), ""),
        )
        task = {
            "id": f"{repo}/task{task_id}/f{agent_feature_id}",
            "repo": repo,
            "task_id": task_id,
            "feature_id": agent_feature_id,
            "instructions": agent.get_task_message(task_description),
        }

        # === Create orchestrator and run ===
        orchestrator = CodingAgentOrchestrator(
            agent_model=agent_model,
            config=config,
            environment=environment,
            messaging=messaging,
        )

        trajectory = await orchestrator.run_session(task, agent)

        # === Extract trainable agent's patch ===
        agent_patch = await environment.get_patch()

        # === Wait for partner ===
        partner_thread.join(timeout=3600)
        partner_patch = partner_result["patch"]

        if partner_error["error"]:
            logger.warning(f"Partner had error: {partner_error['error']}")

        # === Compute reward ===
        partial_reward = getattr(args, "cooperbench_partial_reward", False)
        reward = compute_reward(
            repo_name=repo,
            task_id=task_id,
            f1_id=agent_feature_id,
            f2_id=partner_feature_id,
            agent_patch=agent_patch,
            partner_patch=partner_patch,
            dataset_dir=dataset_dir,
            partial_reward=partial_reward,
        )

        trajectory.reward = reward
        trajectory.metadata["agent_patch_lines"] = len(agent_patch.splitlines())
        trajectory.metadata["partner_patch_lines"] = len(partner_patch.splitlines())
        trajectory.metadata["partner_model"] = partner_model

        # === Convert to Slime sample ===
        output_sample = trajectory_to_slime_sample(trajectory, sample.index)
        return [output_sample]

    except Exception as e:
        logger.error(f"CooperBench rollout failed: {e}")
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
    finally:
        if environment:
            try:
                import asyncio

                asyncio.get_event_loop().run_until_complete(environment.cleanup())
            except Exception:
                pass


def add_cooperbench_arguments(parser: Any) -> None:
    """Add CooperBench-specific arguments to Slime argument parser.

    Args:
        parser: argparse ArgumentParser
    """
    group = parser.add_argument_group("cooperbench", "CooperBench arguments")

    group.add_argument(
        "--cooperbench-subset",
        type=str,
        default=None,
        help="CooperBench subset: lite, flash, or None for all",
    )

    group.add_argument(
        "--cooperbench-repo",
        type=str,
        default=None,
        help="Filter by repository name",
    )

    group.add_argument(
        "--cooperbench-partner-model",
        type=str,
        default="gpt-5-mini",
        help="LLM model for the partner agent (default: gpt-5-mini)",
    )

    group.add_argument(
        "--cooperbench-max-steps",
        type=int,
        default=50,
        help="Maximum steps per agent (default: 50)",
    )

    group.add_argument(
        "--cooperbench-redis-url",
        type=str,
        default="redis://localhost:6379",
        help="Redis URL for inter-agent messaging",
    )

    group.add_argument(
        "--cooperbench-dataset-dir",
        type=str,
        default=None,
        help="Path to CooperBench directory (with dataset/ subdirectory)",
    )

    group.add_argument(
        "--cooperbench-partial-reward",
        action="store_true",
        default=False,
        help="Give 0.5 reward when agent's feature passes but merge fails",
    )
