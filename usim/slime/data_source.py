"""Data source for loading tau2-bench tasks into Slime format."""

import logging
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class Tau2DataSource:
    """Data source for tau2-bench tasks.

    Loads tasks from tau2-bench and converts them to Slime Sample format.
    """

    def __init__(
        self,
        domain: str = "retail",
        task_split: str = "test",
        max_tasks: Optional[int] = None,
    ):
        """Initialize the tau2-bench data source.

        Args:
            domain: tau2-bench domain (retail, airline, telecom)
            task_split: Task split to load (train, test)
            max_tasks: Optional limit on number of tasks
        """
        self.domain = domain
        self.task_split = task_split
        self.max_tasks = max_tasks
        self._tasks: Optional[List[Any]] = None

    def _load_tasks(self) -> List[Any]:
        """Load tasks from tau2-bench."""
        if self._tasks is not None:
            return self._tasks

        try:
            from tau2.data_model.tasks import get_tasks

            tasks = get_tasks(domain=self.domain, task_split=self.task_split)

            if self.max_tasks:
                tasks = tasks[: self.max_tasks]

            self._tasks = tasks
            logger.info(f"Loaded {len(tasks)} tasks from tau2-bench {self.domain}/{self.task_split}")

            return tasks

        except ImportError:
            logger.warning("tau2-bench not available, returning empty task list")
            self._tasks = []
            return []

    def __len__(self) -> int:
        """Get number of tasks."""
        return len(self._load_tasks())

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over tasks as dicts."""
        for task in self._load_tasks():
            yield self._task_to_dict(task)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get task by index."""
        tasks = self._load_tasks()
        return self._task_to_dict(tasks[index])

    def _task_to_dict(self, task: Any) -> Dict[str, Any]:
        """Convert tau2-bench Task to dict format.

        Args:
            task: tau2-bench Task object

        Returns:
            Task as dictionary
        """
        return {
            "id": task.id,
            "domain": self.domain,
            "instructions": task.user_instructions.instructions if task.user_instructions else "",
            "domain_policy": "",  # Loaded from environment
            "initial_state": task.initial_state,
            "evaluation": task.evaluation,
        }

    def to_slime_samples(self) -> List[Any]:
        """Convert all tasks to Slime Sample format.

        Returns:
            List of Slime Sample objects
        """
        try:
            from slime.data.types import Sample
        except ImportError:
            raise ImportError("slime package required for to_slime_samples")

        samples = []
        for i, task_dict in enumerate(self):
            sample = Sample(
                index=i,
                prompt=task_dict["instructions"],
                tokens=[],  # Will be filled during rollout
                response="",
                reward=0.0,
                loss_mask=[],
                response_length=0,
                metadata={
                    "task_id": task_dict["id"],
                    "domain": task_dict["domain"],
                    "domain_policy": task_dict["domain_policy"],
                    "initial_state": task_dict["initial_state"],
                },
            )
            samples.append(sample)

        return samples


def create_tau2_data_source(
    domain: str = "retail",
    task_split: str = "test",
    max_tasks: Optional[int] = None,
) -> Tau2DataSource:
    """Factory function to create a tau2-bench data source.

    Args:
        domain: tau2-bench domain
        task_split: Task split to load
        max_tasks: Optional task limit

    Returns:
        Configured Tau2DataSource
    """
    return Tau2DataSource(
        domain=domain,
        task_split=task_split,
        max_tasks=max_tasks,
    )


def get_tau2_samples(
    domain: str = "retail",
    task_split: str = "test",
    max_tasks: Optional[int] = None,
) -> List[Any]:
    """Convenience function to get tau2-bench tasks as Slime samples.

    Args:
        domain: tau2-bench domain
        task_split: Task split to load
        max_tasks: Optional task limit

    Returns:
        List of Slime Sample objects
    """
    data_source = create_tau2_data_source(domain, task_split, max_tasks)
    return data_source.to_slime_samples()
