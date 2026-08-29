"""CooperBench sandbox using mini-swe-agent-v2's ModalEnvironment.

Uses cooperbench.agents.mini_swe_agent_v2.environments.modal.ModalEnvironment
which provides retry logic, sandbox resurrection, and the v2 action dict interface.

All ModalEnvironment operations are pinned to a single dedicated thread because
Modal's gRPC state is not safe to use across different threads.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CooperBenchSandbox:
    """Async wrapper around mini-swe-agent-v2's ModalEnvironment.

    All sync ModalEnvironment calls (init, execute, cleanup) run on a single
    dedicated thread to avoid cross-thread gRPC issues with Modal's SDK.
    """

    def __init__(
        self,
        image_name: str,
        cwd: str = "/workspace/repo",
        timeout: int = 3600,
    ):
        self.image_name = image_name
        self.cwd = cwd
        self.timeout = timeout
        self._env: Optional[Any] = None
        self._started = False
        self._pool = ThreadPoolExecutor(max_workers=1)

    def _run_sync(self, fn, *args, **kwargs):
        """Run a sync function on the dedicated thread."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self._pool, lambda: fn(*args, **kwargs))

    async def start(self) -> None:
        """Create the Modal sandbox."""
        if self._started:
            return

        from cooperbench.agents.mini_swe_agent_v2.environments.modal import ModalEnvironment

        self._env = await self._run_sync(
            ModalEnvironment,
            image=self.image_name,
            cwd=self.cwd,
            timeout=self.timeout,
        )
        self._started = True
        logger.info(f"CooperBench sandbox started: image={self.image_name}")

    async def execute(self, command: str) -> Dict[str, Any]:
        """Execute a bash command. Returns {'output', 'returncode'}."""
        if not self._started:
            await self.start()

        result = await self._run_sync(self._env.execute, {"command": command})
        return {
            "output": result.get("output", ""),
            "returncode": result.get("returncode", 0),
        }

    async def get_patch(self) -> str:
        """Get git diff from the sandbox."""
        result = await self.execute("git diff HEAD")
        if result["returncode"] == 0:
            return result["output"]
        result = await self.execute("git diff")
        return result.get("output", "")

    async def cleanup(self) -> None:
        """Terminate the Modal sandbox."""
        if self._env and self._started:
            try:
                await self._run_sync(self._env.cleanup)
            except Exception as e:
                logger.warning(f"Error terminating sandbox: {e}")
            self._started = False
            logger.info("CooperBench sandbox stopped")
        self._pool.shutdown(wait=False)
