"""Data source for loading CooperBench tasks into Slime format."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CooperBenchDataSource:
    """Data source for CooperBench tasks.

    Loads tasks from CooperBench's dataset/ directory and converts them
    to Slime Sample format for training.
    """

    def __init__(
        self,
        subset: Optional[str] = None,
        repo_filter: Optional[str] = None,
        task_filter: Optional[int] = None,
        dataset_dir: Optional[str] = None,
    ):
        """Initialize CooperBench data source.

        Args:
            subset: Predefined subset (e.g., 'lite', 'flash')
            repo_filter: Filter by repository name
            task_filter: Filter by task ID
            dataset_dir: Path to CooperBench dataset directory
        """
        self.subset = subset
        self.repo_filter = repo_filter
        self.task_filter = task_filter
        self.dataset_dir = dataset_dir
        self._tasks: Optional[List[Dict[str, Any]]] = None

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from CooperBench."""
        if self._tasks is not None:
            return self._tasks

        # Change to dataset directory if specified
        original_cwd = os.getcwd()
        if self.dataset_dir:
            os.chdir(self.dataset_dir)

        try:
            from cooperbench.runner.tasks import discover_tasks
            from cooperbench.utils import get_image_name

            raw_tasks = discover_tasks(
                subset=self.subset,
                repo_filter=self.repo_filter,
                task_filter=self.task_filter,
            )

            tasks = []
            for t in raw_tasks:
                repo = t["repo"]
                task_id = t["task_id"]
                features = t["features"]
                image_name = get_image_name(repo, task_id)

                # Load feature descriptions
                feature_descriptions = {}
                task_dir = Path("dataset") / repo / f"task{task_id}"
                for fid in features:
                    feature_file = task_dir / f"feature{fid}" / "feature.md"
                    if feature_file.exists():
                        feature_descriptions[fid] = feature_file.read_text()

                tasks.append({
                    "repo": repo,
                    "task_id": task_id,
                    "features": features,
                    "feature_descriptions": feature_descriptions,
                    "image_name": image_name,
                })

            self._tasks = tasks
            logger.info(f"Loaded {len(tasks)} CooperBench tasks (subset={self.subset})")
            return tasks

        except ImportError:
            logger.warning("cooperbench not available, returning empty task list")
            self._tasks = []
            return []
        finally:
            if self.dataset_dir:
                os.chdir(original_cwd)

    def to_slime_samples(self) -> List[Any]:
        """Convert all tasks to Slime Sample format.

        Returns:
            List of Slime Sample objects
        """
        try:
            from slime.data.types import Sample
        except ImportError:
            raise ImportError("slime package required for to_slime_samples")

        tasks = self._load_tasks()
        samples = []

        for i, task in enumerate(tasks):
            f1, f2 = task["features"]
            # Use feature 1 description as the prompt (agent's task)
            prompt = task["feature_descriptions"].get(f1, "")

            sample = Sample(
                index=i,
                prompt=prompt,
                tokens=[],
                response="",
                reward=0.0,
                loss_mask=[],
                response_length=0,
                metadata={
                    "repo": task["repo"],
                    "task_id": task["task_id"],
                    "features": task["features"],
                    "feature_descriptions": task["feature_descriptions"],
                    "image_name": task["image_name"],
                    "agent_feature_id": f1,
                    "partner_feature_id": f2,
                },
            )
            samples.append(sample)

        return samples


def get_cooperbench_samples(
    subset: Optional[str] = None,
    repo_filter: Optional[str] = None,
    dataset_dir: Optional[str] = None,
) -> List[Any]:
    """Convenience function to get CooperBench tasks as Slime samples.

    Args:
        subset: Predefined subset (e.g., 'lite', 'flash')
        repo_filter: Filter by repository name
        dataset_dir: Path to CooperBench dataset directory

    Returns:
        List of Slime Sample objects
    """
    data_source = CooperBenchDataSource(
        subset=subset,
        repo_filter=repo_filter,
        dataset_dir=dataset_dir,
    )
    return data_source.to_slime_samples()
