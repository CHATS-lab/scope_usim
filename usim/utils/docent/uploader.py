"""Upload USIM trajectories to Docent for analysis.

Provides:
- DocentUploader: Background-thread uploader for use during training
- get_docent_uploader: Cached factory for use in rollout functions
- upload_from_directory: Standalone function for uploading JSONL files after the fact
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

try:
    from docent import Docent
    from docent.data_models import AgentRun

    from usim.utils.docent.converter import load_jsonl_directory, sample_to_agent_run

    _HAS_DOCENT = True
except Exception:
    _HAS_DOCENT = False

logger = logging.getLogger(__name__)

# Shared cache for get_docent_uploader(). Maps collection name -> DocentUploader | None.
# None means init was attempted and failed (don't retry every rollout step).
_uploader_cache: dict = {}

_SENTINEL_FAILED = object()


def get_docent_uploader(args: Any) -> Optional["DocentUploader"]:
    """Lazily create (and cache) a DocentUploader from CLI args.

    Returns None if --docent-collection is not set, or if initialization fails.
    Failures are cached so we don't retry on every rollout step.

    NOTE: Must be called from a single thread (the training loop).
    """
    collection = getattr(args, "docent_collection", None)
    if not collection:
        return None

    if not _HAS_DOCENT:
        logger.warning("docent-python not installed. Docent uploads are disabled.")
        return None

    cached = _uploader_cache.get(collection)
    if cached is _SENTINEL_FAILED:
        return None
    if cached is not None:
        return cached

    try:
        uploader = DocentUploader(
            collection_name=collection,
            upload_interval=getattr(args, "docent_upload_interval", 1),
        )
        _uploader_cache[collection] = uploader
        return uploader
    except Exception as e:
        logger.warning(
            f"Failed to initialize DocentUploader: {e}. "
            "Docent uploads are disabled for this run."
        )
        _uploader_cache[collection] = _SENTINEL_FAILED
        return None


class DocentUploader:
    """Non-blocking Docent uploader for use during training.

    Uploads batches of Slime Samples to a Docent collection in a background
    thread so training is never blocked by upload latency or failures.
    """

    def __init__(
        self,
        collection_name: str,
        description: str = "",
        api_key: Optional[str] = None,
        upload_interval: int = 1,
    ):
        api_key = api_key or os.environ.get("DOCENT_API_KEY")
        if not api_key:
            raise ValueError(
                "Docent API key required. Set DOCENT_API_KEY env var "
                "or pass api_key to DocentUploader."
            )

        self._client = Docent(api_key=api_key)
        self._collection_id = self._client.create_collection(
            name=collection_name,
            description=description,
        )
        self._upload_interval = upload_interval
        self._call_count = 0
        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(
            f"Docent collection created: {self.collection_url} "
            f"(upload every {upload_interval} rollout(s))"
        )

    @property
    def collection_url(self) -> str:
        return f"https://docent.transluce.org/collection/{self._collection_id}"

    def upload_batch(self, samples: List[Any], rollout_id: int) -> None:
        """Convert samples to AgentRuns and upload in a background thread.

        Args:
            samples: Flat list of Slime Sample objects.
            rollout_id: Current rollout/step ID (used for interval gating).
        """
        self._call_count += 1
        if self._upload_interval > 1 and (self._call_count % self._upload_interval != 0):
            return

        # Convert synchronously (cheap) before submitting to thread
        agent_runs: List[AgentRun] = []
        for s in samples:
            meta = getattr(s, "metadata", {}) or {}
            if not meta.get("messages"):
                continue
            try:
                agent_runs.append(sample_to_agent_run(s))
            except Exception as e:
                logger.warning(f"Failed to convert sample {getattr(s, 'index', '?')}: {e}")

        if not agent_runs:
            return

        # Fire-and-forget upload in background thread
        self._executor.submit(self._upload, agent_runs, rollout_id)

    def _upload(self, agent_runs: List[AgentRun], rollout_id: int) -> None:
        """Actual upload (runs in background thread). Errors are logged, never raised."""
        try:
            self._client.add_agent_runs(self._collection_id, agent_runs)
            logger.info(
                f"Docent: uploaded {len(agent_runs)} runs (rollout {rollout_id}) "
                f"→ {self.collection_url}"
            )
        except Exception as e:
            logger.error(f"Docent upload failed (rollout {rollout_id}): {e}")


def upload_from_directory(
    data_dir: str,
    collection_name: str,
    description: str = "",
    api_key: Optional[str] = None,
) -> str:
    """Load JSONL files from a directory and upload all trajectories to Docent.

    Args:
        data_dir: Directory containing rollout_*.jsonl files.
        collection_name: Name for the Docent collection.
        description: Optional collection description.
        api_key: Docent API key (falls back to DOCENT_API_KEY env var).

    Returns:
        URL of the created Docent collection.
    """
    api_key = api_key or os.environ.get("DOCENT_API_KEY")
    if not api_key:
        raise ValueError(
            "Docent API key required. Set DOCENT_API_KEY env var or pass api_key."
        )

    agent_runs = load_jsonl_directory(data_dir)
    if not agent_runs:
        raise ValueError(f"No trajectory data found in {data_dir}")

    client = Docent(api_key=api_key)
    collection_id = client.create_collection(
        name=collection_name,
        description=description,
    )
    client.add_agent_runs(collection_id, agent_runs)

    url = f"https://docent.transluce.org/collection/{collection_id}"
    logger.info(f"Uploaded {len(agent_runs)} runs to {url}")
    return url
