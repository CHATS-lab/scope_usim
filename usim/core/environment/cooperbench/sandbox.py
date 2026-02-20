"""CooperBench sandbox wrapping mini-swe-agent's SwerexModalEnvironment.

Provides command execution in a Modal sandbox with persistent state,
using SWE-ReX's ModalDeployment under the hood.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CooperBenchSandbox:
    """Wraps mini-swe-agent's SwerexModalEnvironment for command execution.

    The SwerexModalEnvironment uses SWE-ReX's ModalDeployment which creates
    a Modal sandbox with a swerex-remote server for session-based execution.
    """

    def __init__(
        self,
        image_name: str,
        cwd: str = "/workspace/repo",
        timeout: int = 3600,
        per_command_timeout: int = 30,
    ):
        """Initialize sandbox.

        Args:
            image_name: Docker image for the sandbox (e.g., from get_image_name())
            cwd: Working directory inside the sandbox
            timeout: How long the sandbox stays alive (seconds)
            per_command_timeout: Per-command timeout (seconds)
        """
        self.image_name = image_name
        self.cwd = cwd
        self.timeout = timeout
        self.per_command_timeout = per_command_timeout
        self._env: Optional[Any] = None
        self._started = False

    async def start(self) -> None:
        """Start the sandbox environment."""
        if self._started:
            return

        from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment

        self._env = SwerexModalEnvironment(
            image=self.image_name,
            cwd=self.cwd,
            timeout=self.per_command_timeout,
            runtime_timeout=self.timeout,
        )
        await self._env.start()
        self._started = True
        logger.info(f"CooperBench sandbox started: image={self.image_name}")

    async def execute(self, command: str) -> Dict[str, Any]:
        """Execute a command in the sandbox.

        Args:
            command: Bash command to execute

        Returns:
            Dict with 'output' (stdout+stderr) and 'returncode'
        """
        if not self._started:
            await self.start()

        result = await self._env.execute(command, cwd=self.cwd)
        return {
            "output": result.get("output", ""),
            "returncode": result.get("returncode", -1),
        }

    async def get_patch(self) -> str:
        """Get git diff from the sandbox.

        Returns:
            Patch content as string
        """
        result = await self.execute("git diff HEAD")
        if result["returncode"] == 0:
            return result["output"]
        # Also capture committed but not-yet-diffed changes
        result = await self.execute("git diff")
        return result.get("output", "")

    async def cleanup(self) -> None:
        """Stop and clean up the sandbox."""
        if self._env and self._started:
            try:
                await self._env.stop()
            except Exception as e:
                logger.warning(f"Error stopping sandbox: {e}")
            self._started = False
            logger.info("CooperBench sandbox stopped")
