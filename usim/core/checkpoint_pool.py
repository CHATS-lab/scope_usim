"""Checkpoint pool for self-play training (Mode 3).

Manages a collection of model checkpoint paths with FIFO eviction.
Each entry records: path, training step, timestamp.

The pool persists state to a JSON manifest file so it survives restarts.

Usage:
    pool = CheckpointPool("/path/to/pool", max_size=10)
    pool.add("/checkpoints/step_0", step=0)
    pool.add("/checkpoints/step_16", step=16)

    ckpt = pool.sample()           # random selection
    ckpt = pool.sample("latest")   # most recent
    ckpt = pool.latest()           # always most recent
"""

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CheckpointEntry:
    """A single checkpoint in the pool."""
    path: str
    step: int
    timestamp: float = 0.0


@dataclass
class PoolManifest:
    """Persistent state of the checkpoint pool."""
    entries: List[CheckpointEntry] = field(default_factory=list)
    max_size: int = 10


class CheckpointPool:
    """File-based checkpoint pool with FIFO eviction.

    Stores checkpoint paths (not the checkpoints themselves) in a JSON
    manifest. The pool grows up to max_size, then evicts the oldest entry.
    """

    def __init__(self, pool_dir: str, max_size: int = 10):
        self._pool_dir = Path(pool_dir)
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._pool_dir / "pool_manifest.json"
        self._max_size = max_size
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> PoolManifest:
        """Load manifest from disk, or create empty one."""
        if self._manifest_path.exists():
            data = json.loads(self._manifest_path.read_text())
            entries = [CheckpointEntry(**e) for e in data.get("entries", [])]
            return PoolManifest(
                entries=entries,
                max_size=data.get("max_size", self._max_size),
            )
        return PoolManifest(max_size=self._max_size)

    def _save_manifest(self):
        """Write manifest to disk."""
        data = {
            "entries": [asdict(e) for e in self._manifest.entries],
            "max_size": self._manifest.max_size,
        }
        self._manifest_path.write_text(json.dumps(data, indent=2))

    def add(self, checkpoint_path: str, step: int):
        """Add a checkpoint to the pool, evicting oldest if at capacity."""
        import time

        entry = CheckpointEntry(
            path=checkpoint_path,
            step=step,
            timestamp=time.time(),
        )

        # Check for duplicate step
        for existing in self._manifest.entries:
            if existing.step == step:
                logger.info(f"Checkpoint pool: updating step {step} -> {checkpoint_path}")
                existing.path = checkpoint_path
                existing.timestamp = entry.timestamp
                self._save_manifest()
                return

        # FIFO eviction
        if len(self._manifest.entries) >= self._max_size:
            evicted = self._manifest.entries.pop(0)
            logger.info(
                f"Checkpoint pool: evicting step {evicted.step} ({evicted.path})"
            )

        self._manifest.entries.append(entry)
        self._save_manifest()
        logger.info(
            f"Checkpoint pool: added step {step} ({checkpoint_path}), "
            f"pool size={len(self._manifest.entries)}/{self._max_size}"
        )

    def sample(self, strategy: str = "random") -> str:
        """Sample a checkpoint path from the pool.

        Args:
            strategy: "random" for uniform random, "latest" for most recent

        Returns:
            Checkpoint path string

        Raises:
            ValueError: If pool is empty
        """
        if not self._manifest.entries:
            raise ValueError("Checkpoint pool is empty")

        if strategy == "latest":
            return self._manifest.entries[-1].path

        return random.choice(self._manifest.entries).path

    def latest(self) -> str:
        """Return the most recently added checkpoint path."""
        return self.sample("latest")

    def __len__(self) -> int:
        return len(self._manifest.entries)

    def __repr__(self) -> str:
        steps = [e.step for e in self._manifest.entries]
        return f"CheckpointPool(size={len(self)}/{self._max_size}, steps={steps})"
