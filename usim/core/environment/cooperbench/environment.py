"""CooperBench environment implementing USIM's BaseEnvironment protocol.

Wraps a CooperBenchSandbox for bash command execution and optionally
a MessagingConnector for inter-agent communication.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from usim.core.environment.cooperbench.messaging import MessagingConnector
from usim.core.environment.cooperbench.sandbox import CooperBenchSandbox
from usim.core.types import Message, ToolCall

logger = logging.getLogger(__name__)

# Truncation limits for command output (matches CooperBench mini_swe_agent_v2/config/mini.yaml)
MAX_OUTPUT_LEN = 10000
TRUNCATED_HEAD = 5000
TRUNCATED_TAIL = 5000


class CooperBenchEnvironment:
    """USIM environment for CooperBench coding tasks.

    Manages a Modal sandbox for bash execution and optional messaging
    for inter-agent coordination. Tool calling follows mini-swe-agent-v2
    protocol: bash tool + optional send_message tool.
    """

    def __init__(
        self,
        image_name: str,
        messaging: Optional[MessagingConnector] = None,
        messaging_enabled: bool = False,
        timeout: int = 3600,
    ):
        """Initialize the environment.

        Args:
            image_name: Docker image for the sandbox
            messaging: Optional messaging connector for inter-agent communication
            messaging_enabled: Whether to expose the send_message tool
            timeout: Sandbox lifetime in seconds
        """
        self.image_name = image_name
        self.messaging = messaging
        self.messaging_enabled = messaging_enabled
        self.sandbox = CooperBenchSandbox(
            image_name=image_name,
            timeout=timeout,
        )
        self._step_count = 0

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tool schemas: bash + optional send_message."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
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
        if self.messaging_enabled:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "send_message",
                        "description": "Send a message to a teammate agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "recipient": {
                                    "type": "string",
                                    "description": "The agent ID to send the message to",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The message content",
                                },
                            },
                            "required": ["recipient", "content"],
                        },
                    },
                }
            )
        return tools

    def execute_tool(self, tool_call: ToolCall) -> Message:
        """Synchronous tool execution (not used in coding orchestrator)."""
        raise NotImplementedError("Use execute_tool_async for CooperBench")

    async def execute_tool_async(self, tool_call: ToolCall) -> Message:
        """Execute a bash command in the sandbox.

        Args:
            tool_call: ToolCall with command in arguments

        Returns:
            Message with execution result as JSON content (v2 format)
        """
        command = tool_call.arguments.get("command", "") if isinstance(tool_call.arguments, dict) else ""
        result = await self.sandbox.execute(command)
        self._step_count += 1

        content = _format_observation(result["output"], result["returncode"])

        return Message(
            role="tool",
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
    """Format command execution result as JSON observation.

    Follows CooperBench mini_swe_agent_v2 observation_template format.
    """
    if len(output) < MAX_OUTPUT_LEN:
        return json.dumps({"returncode": returncode, "output": output})
    return json.dumps({
        "returncode": returncode,
        "output_head": output[:TRUNCATED_HEAD],
        "output_tail": output[-TRUNCATED_TAIL:],
        "elided_chars": len(output) - MAX_OUTPUT_LEN,
        "warning": "Output too long.",
    })
