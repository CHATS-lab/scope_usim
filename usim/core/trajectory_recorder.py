"""Record episode trajectories to JSONL for downstream model training."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """Records trajectory messages + metadata to JSONL files.

    Each rollout batch produces one file: {output_dir}/rollout_{rollout_id:06d}.jsonl
    Disabled (no-op) when output_dir is None.
    """

    def __init__(self, output_dir: Optional[str]):
        self.output_dir = output_dir

    def record_batch(
        self,
        samples: List[Any],
        rollout_id: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write one JSONL line per sample with messages + metadata.

        Skips samples with no messages (failed rollouts with no conversation).

        Args:
            samples: Slime Sample objects (flat list, not grouped).
            rollout_id: Training step / rollout batch ID.
            extra_metadata: Extra fields merged into each JSONL row.
        """
        if self.output_dir is None:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"rollout_{rollout_id:06d}.jsonl")
        timestamp = datetime.now(timezone.utc).isoformat()

        with open(path, "w") as f:
            for sample in samples:
                meta = getattr(sample, "metadata", {}) or {}
                messages = meta.get("messages", [])
                if not messages:
                    continue

                row = {
                    "messages": messages,
                    "reward": getattr(sample, "reward", 0.0),
                    "status": str(getattr(sample, "status", "unknown")),
                    "turn_count": meta.get("turn_count", 0),
                    "metadata": {
                        k: v for k, v in meta.items() if k not in ("messages", "turn_count")
                    },
                    "rollout_id": rollout_id,
                    "timestamp": timestamp,
                }
                if extra_metadata:
                    row.update(extra_metadata)

                f.write(json.dumps(row) + "\n")

        logger.info(f"Recorded {len(samples)} trajectories to {path}")
