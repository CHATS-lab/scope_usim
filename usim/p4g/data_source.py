"""Data source for loading Persuasion for Good tasks into Slime format."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class P4GDataSource:
    """Data source for Persuasion for Good dialogues.

    Loads dialogue JSONL files and extracts speaker IDs for persona loading.
    """

    def __init__(
        self,
        dataset_dir: str,
        corpus_path: str,
        max_tasks: Optional[int] = None,
    ):
        self.dataset_dir = dataset_dir
        self.corpus_path = corpus_path
        self.max_tasks = max_tasks
        self._tasks: Optional[List[Dict[str, Any]]] = None

    def _load_tasks(self) -> List[Dict[str, Any]]:
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

    def to_slime_samples(self) -> List[Any]:
        """Convert all tasks to Slime Sample format."""
        try:
            from slime.data.types import Sample
        except ImportError:
            raise ImportError("slime package required for to_slime_samples")

        tasks = self._load_tasks()
        samples = []
        for i, task in enumerate(tasks):
            sample = Sample(
                index=i,
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
            samples.append(sample)

        return samples


def get_p4g_samples(
    dataset_dir: str,
    corpus_path: str,
    max_tasks: Optional[int] = None,
) -> List[Any]:
    """Factory function for Slime data source.

    Args:
        dataset_dir: Path to directory with JSONL dialogue files
        corpus_path: Path to convokit Corpus for persona loading
        max_tasks: Optional limit on number of tasks

    Returns:
        List of Slime Sample objects
    """
    data_source = P4GDataSource(dataset_dir, corpus_path, max_tasks)
    return data_source.to_slime_samples()
