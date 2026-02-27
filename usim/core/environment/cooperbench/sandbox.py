"""CooperBench sandbox using CooperBench's Modal backend.

Provides async command execution in a persistent Modal sandbox,
wrapping CooperBench's synchronous ModalBackend with asyncio.to_thread.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CooperBenchSandbox:
    """Async wrapper around CooperBench's ModalBackend for agent interaction.

    Uses cooperbench.eval.backends.modal.ModalBackend to create a long-lived
    Modal sandbox, then wraps the synchronous exec() calls with asyncio.to_thread
    so they don't block the event loop.
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
        self._sandbox: Optional[Any] = None  # cooperbench ModalSandbox
        self._started = False

    async def start(self) -> None:
        """Create the Modal sandbox via CooperBench's backend."""
        if self._started:
            return

        from cooperbench.eval.backends.modal import ModalBackend

        backend = ModalBackend(app_name="cooperbench-agent")
        self._sandbox = await asyncio.to_thread(
            backend.create_sandbox,
            image=self.image_name,
            timeout=self.timeout,
            workdir=self.cwd,
        )
        self._started = True
        logger.info(f"CooperBench sandbox started: image={self.image_name}")

    async def execute(self, command: str) -> Dict[str, Any]:
        """Execute a bash command in the sandbox.

        Returns:
            Dict with 'output' (stdout+stderr) and 'returncode'
        """
        if not self._started:
            await self.start()

        result = await asyncio.to_thread(
            self._sandbox.exec, "bash", "-c", command,
        )
        stdout = result.stdout_read()
        stderr = result.stderr_read()
        output = stdout + stderr if stderr else stdout

        return {
            "output": output,
            "returncode": result.returncode,
        }

    async def get_patch(self) -> str:
        """Get git diff from the sandbox."""
        result = await self.execute("git diff HEAD")
        if result["returncode"] == 0:
            return result["output"]
        result = await self.execute("git diff")
        return result.get("output", "")

    async def cleanup(self) -> None:
        """Terminate the sandbox."""
        if self._sandbox and self._started:
            try:
                await asyncio.to_thread(self._sandbox.terminate)
            except Exception as e:
                logger.warning(f"Error terminating sandbox: {e}")
            self._started = False
            logger.info("CooperBench sandbox stopped")
