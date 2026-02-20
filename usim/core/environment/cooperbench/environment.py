"""CooperBench environment implementing USIM's BaseEnvironment protocol.

Wraps a CooperBenchSandbox for bash command execution and optionally
a MessagingConnector for inter-agent communication.
"""

import logging
from typing import Any, Dict, List, Optional

from usim.core.environment.cooperbench.messaging import MessagingConnector
from usim.core.environment.cooperbench.sandbox import CooperBenchSandbox
from usim.core.types import Message, ToolCall

logger = logging.getLogger(__name__)

# Truncation limits for command output (matches CooperBench's mini.yaml)
MAX_OUTPUT_LEN = 10000
TRUNCATED_HEAD = 5000
TRUNCATED_TAIL = 5000


class CooperBenchEnvironment:
    """USIM environment for CooperBench coding tasks.

    Manages a Modal sandbox for bash execution and optional messaging
    for inter-agent coordination.
    """

    def __init__(
        self,
        image_name: str,
        messaging: Optional[MessagingConnector] = None,
        timeout: int = 3600,
    ):
        """Initialize the environment.

        Args:
            image_name: Docker image for the sandbox
            messaging: Optional messaging connector for inter-agent communication
            timeout: Sandbox lifetime in seconds
        """
        self.image_name = image_name
        self.messaging = messaging
        self.sandbox = CooperBenchSandbox(
            image_name=image_name,
            timeout=timeout,
        )
        self._step_count = 0

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tool schemas (bash_execute)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash_execute",
                    "description": "Execute a bash command in the sandbox",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute",
                            }
                        },
                        "required": ["command"],
                    },
                },
            }
        ]

    def execute_tool(self, tool_call: ToolCall) -> Message:
        """Synchronous tool execution (not used in coding orchestrator)."""
        raise NotImplementedError("Use execute_tool_async for CooperBench")

    async def execute_tool_async(self, tool_call: ToolCall) -> Message:
        """Execute a bash command in the sandbox.

        Args:
            tool_call: ToolCall with command in arguments

        Returns:
            Message with execution result as content
        """
        command = tool_call.arguments.get("command", "")
        result = await self.sandbox.execute(command)
        self._step_count += 1

        output = result["output"]
        returncode = result["returncode"]

        content = _format_observation(output, returncode)

        return Message(
            role="user",
            content=content,
            tool_call_id=tool_call.id,
        )

    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a bash command directly (used by coding orchestrator).

        Args:
            command: Bash command to execute

        Returns:
            Dict with 'output' and 'returncode'
        """
        self._step_count += 1
        return await self.sandbox.execute(command)

    async def get_patch(self) -> str:
        """Get git diff from the sandbox.

        Returns:
            Patch content as string
        """
        return await self.sandbox.get_patch()

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """Reset environment state."""
        self._step_count = 0
        return {"image_name": self.image_name}

    def get_info(self) -> Dict[str, Any]:
        """Get current environment info."""
        return {
            "step_count": self._step_count,
            "image_name": self.image_name,
        }

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        await self.sandbox.cleanup()


def _format_observation(output: str, returncode: int) -> str:
    """Format command execution result as observation text.

    Follows CooperBench's action_observation_template format.
    """
    parts = [f"<returncode>{returncode}</returncode>"]

    if len(output) < MAX_OUTPUT_LEN:
        parts.append(f"<output>\n{output}\n</output>")
    else:
        elided = len(output) - MAX_OUTPUT_LEN
        parts.append(
            "<warning>\n"
            "The output of your last command was too long.\n"
            "Please try a different command that produces less output.\n"
            "</warning>"
        )
        parts.append(f"<output_head>\n{output[:TRUNCATED_HEAD]}\n</output_head>")
        parts.append(f"<elided_chars>\n{elided} characters elided\n</elided_chars>")
        parts.append(f"<output_tail>\n{output[-TRUNCATED_TAIL:]}\n</output_tail>")

    return "\n".join(parts)
