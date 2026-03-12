"""Data source for loading tau2-bench tasks into Slime format."""

import logging
from typing import Any, Dict, Iterator, List, Optional

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class Tau2DataSource(DataSource):
    """Data source for tau2-bench tasks.

    Loads tasks from tau2-bench and provides them as Slime Samples.
    Like SPARE, USIM generates conversations on-the-fly during rollout,
    so this data source provides minimal samples with task metadata.
    """

    def __init__(self, args):
        """Initialize the tau2-bench data source.

        Args:
            args: Slime arguments (uses USIM-specific attributes)
        """
        self.args = args
        self.domain = getattr(args, "usim_domain", "retail")
        self.task_split = getattr(args, "task_split", "test")
        self.max_tasks = getattr(args, "max_tasks", None)

        # Initialize tracking attributes (required by DataSource)
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        self._tasks: Optional[List[Any]] = None

        logger.info(
            f"[USIM] Initialized Tau2DataSource: "
            f"domain={self.domain}, split={self.task_split}, "
            f"max_tasks={self.max_tasks}"
        )

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

    def _task_to_sample(self, task: Any, group_index: int, index: int) -> Sample:
        """Convert tau2-bench Task to Slime Sample."""
        return Sample(
            group_index=group_index,
            index=index,
            prompt=task.user_instructions.instructions if task.user_instructions else "",
            tokens=[],
            response="",
            reward=0.0,
            loss_mask=[],
            response_length=0,
            metadata={
                "task_id": task.id,
                "domain": self.domain,
                "initial_state": task.initial_state,
                "evaluation": task.evaluation,
                "tau2_task": task,
            },
        )

    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        """Return samples for rollout.

        Each sample contains task metadata that the rollout function uses
        to set up the conversation.
        """
        tasks = self._load_tasks()

        if not tasks:
            return [[Sample(group_index=i, index=i) for _ in range(self.args.n_samples_per_prompt)] for i in range(num_samples)]

        samples = []
        for _ in range(num_samples):
            task_idx = self.sample_offset % len(tasks)
            task = tasks[task_idx]
            self.sample_offset += 1

            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = self._task_to_sample(task, self.sample_group_index, self.sample_index)
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)

        return samples

    def __len__(self) -> int:
        """Return total number of tau2-bench tasks."""
        return len(self._load_tasks()) or 1

    def add_samples(self, samples: List[List[Sample]]):
        """No-op: USIM generates conversations on-the-fly."""
        pass

    def save(self, rollout_id):
        """Save data source state."""
        pass

    def load(self, rollout_id=None):
        """Load data source state."""
        pass


def get_tau2_data_source(args) -> Tau2DataSource:
    """Factory function called by Slime's RolloutManager.

    This is the entry point referenced by --data-source-path.

    Args:
        args: Slime arguments

    Returns:
        Configured Tau2DataSource instance
    """
    return Tau2DataSource(args)
