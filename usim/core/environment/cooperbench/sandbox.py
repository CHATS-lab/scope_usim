"""CooperBench sandbox using mini-swe-agent-v2's ModalEnvironment.

Uses cooperbench.agents.mini_swe_agent_v2.environments.modal.ModalEnvironment
which provides retry logic, sandbox resurrection, and the v2 action dict interface.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CooperBenchSandbox:
    """Async wrapper around mini-swe-agent-v2's ModalEnvironment.

    ModalEnvironment creates a persistent Modal sandbox with retry logic
    (up to 5 attempts, exponential backoff). We wrap its synchronous
    execute() / cleanup() with asyncio.to_thread to avoid blocking the loop.
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
        self._env: Optional[Any] = None  # ModalEnvironment instance
        self._started = False

    async def start(self) -> None:
        """Create the Modal sandbox (blocking constructor, run in thread)."""
        if self._started:
            return

        from cooperbench.agents.mini_swe_agent_v2.environments.modal import ModalEnvironment

        self._env = await asyncio.to_thread(
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

        result = await asyncio.to_thread(self._env.execute, {"command": command})
        return {
            "output": result.get("output", ""),
            "returncode": result.get("returncode", -1),
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
                await asyncio.to_thread(self._env.cleanup)
            except Exception as e:
                logger.warning(f"Error terminating sandbox: {e}")
            self._started = False
            logger.info("CooperBench sandbox stopped")
