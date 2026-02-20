"""Partner agent runner for CooperBench.

Runs the partner (fixed, API-based) agent using CooperBench's
existing mini_swe_agent infrastructure. The partner runs in its own
thread with its own Modal sandbox.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_partner_agent(
    repo_name: str,
    task_id: int,
    feature_id: int,
    model_name: str = "gpt-5-mini",
    agent_id: str = "agent2",
    agents: Optional[list] = None,
    redis_url: Optional[str] = None,
    messaging_enabled: bool = True,
    dataset_dir: Optional[str] = None,
) -> str:
    """Run the partner agent on its feature and return the resulting patch.

    This function is designed to be called from a thread. It uses
    CooperBench's agent framework adapter to run a mini_swe_agent
    with an API model (e.g., gpt-5-mini via LiteLLM).

    Args:
        repo_name: Repository name
        task_id: Task ID
        feature_id: Feature ID for the partner to implement
        model_name: LLM model for the partner (e.g., "gpt-5-mini")
        agent_id: Partner's agent identifier
        agents: List of all agent IDs
        redis_url: Redis URL for messaging (with namespace)
        messaging_enabled: Whether messaging is enabled
        dataset_dir: Path to CooperBench dataset directory

    Returns:
        Patch string (git diff) from the partner agent
    """
    import os

    original_cwd = os.getcwd()
    if dataset_dir:
        os.chdir(dataset_dir)

    try:
        from cooperbench.agents import get_runner
        from cooperbench.utils import get_image_name

        # Load feature description
        task_dir = Path("dataset") / repo_name / f"task{task_id}"
        feature_file = task_dir / f"feature{feature_id}" / "feature.md"

        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        task = feature_file.read_text()
        image = get_image_name(repo_name, task_id)

        if agents is None:
            agents = ["agent1", "agent2"]

        # Run via CooperBench's agent adapter
        runner = get_runner("mini_swe_agent")
        result = runner.run(
            task=task,
            image=image,
            agent_id=agent_id,
            model_name=model_name,
            agents=agents,
            comm_url=redis_url,
            messaging_enabled=messaging_enabled,
            config={"backend": "modal"},
        )

        logger.info(
            f"Partner agent completed: status={result.status}, "
            f"cost={result.cost:.4f}, steps={result.steps}"
        )

        return result.patch

    except Exception as e:
        logger.error(f"Partner agent failed: {e}")
        return ""
    finally:
        if dataset_dir:
            os.chdir(original_cwd)
