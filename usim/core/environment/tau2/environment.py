"""Tau2-bench environment adapter.

This module provides an adapter for tau2-bench's environment, allowing
it to be used with the usim orchestrator.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from usim.core.types import Message, ToolCall


logger = logging.getLogger(__name__)


class Tau2BenchEnvironment:
    """Adapter for tau2-bench environment.

    Wraps tau2-bench's Environment class to implement the BaseEnvironment
    protocol for use with UserSimOrchestrator.
    """

    def __init__(
        self,
        domain: str,
        task_id: Optional[str] = None,
        tau2_env: Optional[Any] = None,
    ):
        """Initialize the tau2-bench environment adapter.

        Args:
            domain: Domain name (retail, airline, telecom)
            task_id: Optional task ID for initialization
            tau2_env: Optional pre-created tau2 Environment instance
        """
        self.domain = domain
        self.task_id = task_id
        self._tau2_env = tau2_env
        self._tools: List[Dict[str, Any]] = []
        self._info: Dict[str, Any] = {}

    def _ensure_env(self) -> None:
        """Ensure tau2 environment is initialized."""
        if self._tau2_env is not None:
            return

        try:
            from tau2.environment.environment import Environment
            from tau2.data_model.tasks import get_tasks

            tasks = get_tasks(domain=self.domain)
            if self.task_id:
                task = next((t for t in tasks if t.id == self.task_id), None)
                if task is None:
                    raise ValueError(f"Task {self.task_id} not found in domain {self.domain}")
            else:
                task = tasks[0] if tasks else None

            self._tau2_env = Environment(domain=self.domain)

            # Initialize with task's initial state
            if task and task.initial_state:
                self._tau2_env.set_state(
                    initialization_data=task.initial_state.initialization_data,
                    initialization_actions=task.initial_state.initialization_actions,
                )

            # Cache tools
            self._tools = [tool.openai_schema for tool in self._tau2_env.tools]
            self._info = {
                "domain": self.domain,
                "task_id": self.task_id,
                "policy": getattr(self._tau2_env, "policy", ""),
            }

        except ImportError as e:
            logger.warning(f"tau2-bench not available: {e}")
            raise

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools as OpenAI-format schemas.

        Returns:
            List of tool schemas
        """
        self._ensure_env()
        return self._tools

    def execute_tool(self, tool_call: ToolCall) -> Message:
        """Execute a tool call synchronously.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolMessage with the result
        """
        self._ensure_env()

        try:
            # Convert to tau2 format
            from tau2.data_model.message import ToolCall as Tau2ToolCall

            tau2_tool_call = Tau2ToolCall(
                id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                requestor=tool_call.requestor,
            )

            # Execute
            tool_msg = self._tau2_env.get_response(tau2_tool_call)

            return Message(
                role="tool",
                content=tool_msg.content,
                tool_call_id=tool_msg.id,
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return Message(
                role="tool",
                content=f"Error: {str(e)}",
                tool_call_id=tool_call.id,
            )

    async def execute_tool_async(self, tool_call: ToolCall) -> Message:
        """Execute a tool call asynchronously.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolMessage with the result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_tool, tool_call)

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """Reset the environment.

        Args:
            **kwargs: Reset parameters

        Returns:
            Info dict
        """
        self._tau2_env = None  # Force re-initialization
        self.task_id = kwargs.get("task_id", self.task_id)
        self._ensure_env()
        return self.get_info()

    def get_info(self) -> Dict[str, Any]:
        """Get environment information.

        Returns:
            Dict with tools and policy
        """
        self._ensure_env()
        return {
            **self._info,
            "tools": self._tools,
        }
