"""Base protocol for environment implementations."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from usim.core.types import Message, ToolCall


@runtime_checkable
class BaseEnvironment(Protocol):
    """Protocol for environment implementations.

    An environment handles tool execution and maintains state for the
    conversation. It processes tool calls from agents and users,
    returning the results.
    """

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools as OpenAI-format schemas.

        Returns:
            List of tool schemas in OpenAI function format
        """
        ...

    def execute_tool(self, tool_call: ToolCall) -> Message:
        """Execute a tool call and return the result.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolMessage with the execution result
        """
        ...

    async def execute_tool_async(self, tool_call: ToolCall) -> Message:
        """Execute a tool call asynchronously.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolMessage with the execution result
        """
        ...

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """Reset the environment to initial state.

        Args:
            **kwargs: Environment-specific reset parameters

        Returns:
            Info dict with environment state
        """
        ...

    def get_info(self) -> Dict[str, Any]:
        """Get current environment information.

        Returns:
            Dict with environment state and metadata
        """
        ...
