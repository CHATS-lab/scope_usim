"""Data source for loading CooperBench tasks into Slime format.

Loads pre-split JSON files from usim/data/cooperbench/ and implements
the Slime DataSource protocol for training.

Supports three settings:
- baseline: 1 agent, 1 feature (train_baseline.json)
- solo: 1 agent, 2 features (train_pairs.json)
- coop: 2 agents, 1 feature each — each pair expanded into 2 directed entries
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

from cooperbench.utils import get_image_name

logger = logging.getLogger(__name__)


class CooperBenchDataSource(DataSource):
    """Data source for CooperBench tasks following Slime's DataSource protocol."""

    def __init__(self, args):
        self.args = args
        self.setting = getattr(args, "cooperbench_setting", "solo")
        self.data_dir = Path(
            getattr(args, "cooperbench_data_dir", "")
            or str(Path(__file__).parent.parent.parent / "data" / "cooperbench")
        )

        # Tracking attributes required by DataSource
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        self._entries: Optional[List[Dict[str, Any]]] = None

        logger.info(
            f"[CB] Initialized CooperBenchDataSource: "
            f"setting={self.setting}, data_dir={self.data_dir}"
        )

    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load and expand task entries from JSON files."""
        if self._entries is not None:
            return self._entries

        if self.setting == "baseline":
            json_path = self.data_dir / "train_baseline.json"
            raw = self._read_json(json_path)
            # baseline entries have: id, repo, task_id, feature_id, description
            entries = []
            for item in raw:
                entries.append({
                    "repo": item["repo"],
                    "task_id": item["task_id"],
                    "feature_ids": [item["feature_id"]],
                    "descriptions": {str(item["feature_id"]): item["description"]},
                    "setting": "baseline",
                    "agent_feature_id": item["feature_id"],
                    "partner_feature_id": None,
                    "image_name": get_image_name(item["repo"], item["task_id"]),
                })
        elif self.setting == "solo":
            json_path = self.data_dir / "train_pairs.json"
            raw = self._read_json(json_path)
            # pairs entries have: id, repo, task_id, feature_ids, descriptions
            entries = []
            for item in raw:
                f1, f2 = item["feature_ids"]
                entries.append({
                    "repo": item["repo"],
                    "task_id": item["task_id"],
                    "feature_ids": item["feature_ids"],
                    "descriptions": item["descriptions"],
                    "setting": "solo",
                    "agent_feature_id": f1,
                    "partner_feature_id": f2,
                    "image_name": get_image_name(item["repo"], item["task_id"]),
                })
        elif self.setting == "coop":
            json_path = self.data_dir / "train_pairs.json"
            raw = self._read_json(json_path)
            # Expand each pair into 2 directed entries
            entries = []
            for item in raw:
                f1, f2 = item["feature_ids"]
                base = {
                    "repo": item["repo"],
                    "task_id": item["task_id"],
                    "feature_ids": item["feature_ids"],
                    "descriptions": item["descriptions"],
                    "setting": "coop",
                    "image_name": get_image_name(item["repo"], item["task_id"]),
                }
                # Direction A: agent=f1, partner=f2
                entries.append({
                    **base,
                    "agent_feature_id": f1,
                    "partner_feature_id": f2,
                })
                # Direction B: agent=f2, partner=f1
                entries.append({
                    **base,
                    "agent_feature_id": f2,
                    "partner_feature_id": f1,
                })
        else:
            raise ValueError(f"Unknown setting: {self.setting}")

        self._entries = entries
        logger.info(f"[CB] Loaded {len(entries)} entries (setting={self.setting})")
        return entries

    def _read_json(self, path: Path) -> list:
        if not path.exists():
            logger.warning(f"JSON file not found: {path}")
            return []
        with open(path) as f:
            return json.load(f)

    def _entry_to_sample(self, entry: Dict[str, Any], group_index: int, index: int) -> Sample:
        """Convert an entry dict to a Slime Sample."""
        agent_fid = entry["agent_feature_id"]
        prompt = entry["descriptions"].get(
            str(agent_fid), entry["descriptions"].get(agent_fid, "")
        )
        return Sample(
            group_index=group_index,
            index=index,
            prompt=prompt,
            tokens=[],
            response="",
            reward=0.0,
            loss_mask=[],
            response_length=0,
            metadata={
                "repo": entry["repo"],
                "task_id": entry["task_id"],
                "feature_ids": entry["feature_ids"],
                "descriptions": entry["descriptions"],
                "setting": entry["setting"],
                "agent_feature_id": entry["agent_feature_id"],
                "partner_feature_id": entry["partner_feature_id"],
                "image_name": entry["image_name"],
            },
        )

    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        """Return sample groups for rollout.

        Each group contains n_samples_per_prompt samples for the same task,
        cycling through entries as needed.
        """
        entries = self._load_entries()

        if not entries:
            return [
                [
                    Sample(group_index=i, index=i)
                    for _ in range(self.args.n_samples_per_prompt)
                ]
                for i in range(num_samples)
            ]

        samples = []
        for _ in range(num_samples):
            entry_idx = self.sample_offset % len(entries)
            entry = entries[entry_idx]
            self.sample_offset += 1

            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = self._entry_to_sample(
                    entry, self.sample_group_index, self.sample_index
                )
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)

        return samples

    def add_samples(self, samples: List[List[Sample]]):
        """No-op: CooperBench generates trajectories on-the-fly."""
        pass

    def save(self, rollout_id):
        pass

    def load(self, rollout_id=None):
        pass

    def __len__(self) -> int:
        return len(self._load_entries())


def get_cooperbench_data_source(args) -> CooperBenchDataSource:
    """Factory function called by Slime's RolloutManager.

    This is the entry point referenced by --data-source-path.
    """
    return CooperBenchDataSource(args)
