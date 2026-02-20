"""Data source for loading Persuasion for Good tasks into Slime format.

Implements the Slime DataSource protocol so the RolloutManager can
call get_samples(num_samples) and get back List[List[Sample]].
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class P4GDataSource(DataSource):
    """Data source for Persuasion for Good dialogues.

    Loads dialogue JSONL files and provides them as Slime Samples with
    speaker ID metadata for persona loading during rollout.
    """

    def __init__(self, args):
        """Initialize P4G data source from Slime args.

        Args:
            args: Slime arguments namespace. Uses:
                - p4g_dataset_dir: Path to directory with JSONL dialogue files
                - p4g_corpus_path: Path to convokit Corpus for persona loading
                - max_tasks: Optional limit on number of tasks
                - n_samples_per_prompt: Number of samples per task group
        """
        self.args = args
        self.dataset_dir = getattr(args, "p4g_dataset_dir", "")
        self.corpus_path = getattr(args, "p4g_corpus_path", "")
        self.max_tasks = getattr(args, "max_tasks", None)

        # Tracking attributes required by DataSource
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        self._tasks: Optional[List[Dict[str, Any]]] = None

        logger.info(
            f"[USIM] Initialized P4GDataSource: "
            f"dataset_dir={self.dataset_dir}, corpus_path={self.corpus_path}, "
            f"max_tasks={self.max_tasks}"
        )

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load P4G dialogue tasks from JSONL files."""
        if self._tasks is not None:
            return self._tasks

        dataset_path = Path(self.dataset_dir)
        if not dataset_path.exists():
            logger.warning(f"P4G dataset directory not found: {self.dataset_dir}")
            self._tasks = []
            return []

        tasks = []
        for jsonl_file in sorted(dataset_path.glob("*.jsonl")):
            lines = jsonl_file.read_text().strip().split("\n")
            if not lines:
                continue

            first_line = json.loads(lines[0])
            conversation_id = first_line.get("conversation_id", jsonl_file.stem)

            # Role 0 = persuader, Role 1 = persuadee
            persuader_speaker_id = None
            persuadee_speaker_id = None
            for line in lines:
                entry = json.loads(line)
                if entry.get("role") == 0 and persuader_speaker_id is None:
                    persuader_speaker_id = entry["speaker"]
                elif entry.get("role") == 1 and persuadee_speaker_id is None:
                    persuadee_speaker_id = entry["speaker"]
                if persuader_speaker_id and persuadee_speaker_id:
                    break

            if not persuader_speaker_id or not persuadee_speaker_id:
                logger.warning(f"Skipping {jsonl_file}: missing speaker IDs")
                continue

            tasks.append({
                "conversation_id": conversation_id,
                "persuader_speaker_id": persuader_speaker_id,
                "persuadee_speaker_id": persuadee_speaker_id,
                "num_interactions": len(lines),
            })

        if self.max_tasks:
            tasks = tasks[: self.max_tasks]

        self._tasks = tasks
        logger.info(f"Loaded {len(tasks)} P4G tasks from {self.dataset_dir}")
        return tasks

    def _task_to_sample(self, task: Dict[str, Any], group_index: int, index: int) -> Sample:
        """Convert a P4G task dict to a Slime Sample."""
        return Sample(
            group_index=group_index,
            index=index,
            prompt=f"P4G conversation {task['conversation_id']}",
            tokens=[],
            response="",
            reward=0.0,
            loss_mask=[],
            response_length=0,
            metadata={
                "conversation_id": task["conversation_id"],
                "persuader_speaker_id": task["persuader_speaker_id"],
                "persuadee_speaker_id": task["persuadee_speaker_id"],
                "num_interactions": task["num_interactions"],
                "corpus_path": self.corpus_path,
            },
        )

    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        """Return sample groups for rollout.

        Each group contains n_samples_per_prompt samples for the same task,
        cycling through tasks as needed.
        """
        tasks = self._load_tasks()

        if not tasks:
            return [
                [
                    Sample(group_index=i, index=i)
                    for _ in range(self.args.n_samples_per_prompt)
                ]
                for i in range(num_samples)
            ]

        samples = []
        for _ in range(num_samples):
            task_idx = self.sample_offset % len(tasks)
            task = tasks[task_idx]
            self.sample_offset += 1

            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = self._task_to_sample(
                    task, self.sample_group_index, self.sample_index
                )
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)

        return samples

    def add_samples(self, samples: List[List[Sample]]):
        """No-op: P4G generates conversations on-the-fly."""
        pass

    def save(self, rollout_id):
        """Save data source state."""
        pass

    def load(self, rollout_id=None):
        """Load data source state."""
        pass


def get_p4g_data_source(args) -> P4GDataSource:
    """Factory function called by Slime's RolloutManager.

    This is the entry point referenced by --data-source-path.

    Args:
        args: Slime arguments

    Returns:
        Configured P4GDataSource instance
    """
    return P4GDataSource(args)
