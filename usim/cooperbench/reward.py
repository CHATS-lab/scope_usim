"""Reward computation for CooperBench via merge testing.

Uses CooperBench's evaluate_merge() to test whether both features pass
after merging the two agents' patches.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def compute_reward(
    repo_name: str,
    task_id: int,
    f1_id: int,
    f2_id: int,
    agent_patch: str,
    partner_patch: str,
    dataset_dir: Optional[str] = None,
    partial_reward: bool = False,
) -> float:
    """Compute reward by evaluating merged patches.

    Args:
        repo_name: Repository name
        task_id: Task ID
        f1_id: Feature ID implemented by the trainable agent
        f2_id: Feature ID implemented by the partner
        agent_patch: Trainable agent's patch (git diff)
        partner_patch: Partner agent's patch (git diff)
        dataset_dir: Path to CooperBench dataset directory
        partial_reward: If True, give 0.5 when agent's feature passes
                       but merge fails (default: binary 0/1)

    Returns:
        Reward value: 1.0 if both features pass, 0.0 otherwise
        (or 0.5 for partial credit if enabled)
    """
    original_cwd = os.getcwd()
    if dataset_dir:
        os.chdir(dataset_dir)

    try:
        from cooperbench.eval.sandbox import evaluate_merge

        result = evaluate_merge(
            repo_name=repo_name,
            task_id=task_id,
            feature1_id=f1_id,
            feature2_id=f2_id,
            patch1=agent_patch,
            patch2=partner_patch,
        )

        if result.get("error"):
            logger.warning(f"Evaluation error: {result['error']}")
            return 0.0

        f1_passed = result.get("feature1_tests_passed", 0) > 0
        f2_passed = result.get("feature2_tests_passed", 0) > 0

        if f1_passed and f2_passed:
            logger.info("Both features passed — reward=1.0")
            return 1.0
        elif partial_reward and f1_passed:
            logger.info("Agent's feature passed but merge failed — reward=0.5")
            return 0.5
        else:
            logger.info(
                f"Features: f1={'pass' if f1_passed else 'fail'}, "
                f"f2={'pass' if f2_passed else 'fail'} — reward=0.0"
            )
            return 0.0

    except Exception as e:
        logger.error(f"Reward computation failed: {e}")
        return 0.0
    finally:
        if dataset_dir:
            os.chdir(original_cwd)
