"""LLM-based agent implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from usim.core.model_adapter import ModelAdapter
from usim.core.types import AgentState, Message, ToolCall, UserSimConfig
from usim.core.agent.base import AGENT_STOP


logger = logging.getLogger(__name__)


# Default system prompt for agent
DEFAULT_AGENT_SYSTEM_PROMPT = """You are a helpful customer service agent.

<agent_guidelines>
- Help the user with their request professionally and efficiently
- Use available tools to look up information or perform actions
- If you cannot help or the conversation is complete, respond with ###STOP###
- Be concise but thorough in your responses
</agent_guidelines>

<domain_policy>
{domain_policy}
</domain_policy>
"""


class LLMAgent:
    """LLM-based agent for customer service.

    Uses an LLM to generate agent responses, including tool calls
    when appropriate.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: UserSimConfig,
        domain_policy: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize the LLM agent.

        Args:
            model: ModelAdapter for generation
            config: Configuration settings
            domain_policy: Domain-specific policy/instructions
            system_prompt_template: Custom system prompt template
            tools: Available tools for the agent
        """
        self.model = model
        self.config = config
        self.domain_policy = domain_policy or ""
        self._system_prompt_template = system_prompt_template or DEFAULT_AGENT_SYSTEM_PROMPT
        self.tools = tools
        self._seed: Optional[int] = None

    @property
    def system_prompt(self) -> str:
        """Get the formatted system prompt."""
        return self._system_prompt_template.format(domain_policy=self.domain_policy)

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> AgentState:
        """Initialize agent state with system prompt.

        Args:
            message_history: Optional prior message history

        Returns:
            Initialized AgentState
        """
        system_message = Message(role="system", content=self.system_prompt)

        if message_history is None:
            message_history = []

        return AgentState(
            system_messages=[system_message],
            messages=message_history,
            metadata={"domain_policy": self.domain_policy},
        )

    def build_messages(
        self,
        state: AgentState,
    ) -> List[Dict[str, Any]]:
        """Build messages for LLM generation.

        Returns system prompt + conversation history in OpenAI message format.

        Args:
            state: Current agent state

        Returns:
            List of message dicts for the LLM
        """
        llm_messages = []
        for sys_msg in state.system_messages:
            llm_messages.append(sys_msg.to_dict())
        for msg in state.messages:
            llm_messages.append(msg.to_dict())
        return llm_messages

    def parse_response(
        self,
        text: str,
        state: AgentState,
    ) -> Tuple[Message, AgentState]:
        """Parse generation output into a Message and update state.

        Args:
            text: Generated text from the model
            state: Current agent state

        Returns:
            Tuple of (agent_message, updated_state)
        """
        # Parse tool calls from text if present
        tool_calls = self._parse_tool_calls({"text": text})

        agent_message = Message(
            role="assistant",
            content=text if not tool_calls else None,
            tool_calls=tool_calls,
        )

        state = state.add_message(agent_message)
        return agent_message, state

    def generate_next_message(
        self,
        user_message: Optional[Message],
        state: AgentState,
    ) -> Tuple[Message, AgentState]:
        """Generate agent response synchronously.

        Convenience method that combines build_messages + generate + parse_response.

        Args:
            user_message: User's message to respond to
            state: Current agent state

        Returns:
            Tuple of (agent_message, updated_state)
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_next_message_async(user_message, state)
        )

    async def generate_next_message_async(
        self,
        user_message: Optional[Message],
        state: AgentState,
    ) -> Tuple[Message, AgentState]:
        """Generate agent response asynchronously.

        Convenience method that combines build_messages + generate + parse_response.

        Args:
            user_message: User's message to respond to (None for first turn)
            state: Current agent state

        Returns:
            Tuple of (agent_message, updated_state)
        """
        # Add user message to state if provided
        if user_message is not None:
            state = state.add_message(user_message)

        # Build messages for LLM
        llm_messages = self.build_messages(state)

        # Generate response
        results = await self.model.generate_async(
            messages=llm_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=self.tools,
        )

        if not results or "error" in results[0]:
            error_msg = results[0].get("error", "Unknown error") if results else "No response"
            logger.error(f"Agent generation failed: {error_msg}")
            # Return stop message on error
            agent_message = Message(role="assistant", content=AGENT_STOP)
            state = state.add_message(agent_message)
        else:
            response_text = results[0]["text"]
            logger.debug(f"Agent response: {response_text[:200]}...")

            # Parse tool calls if present
            tool_calls = self._parse_tool_calls(results[0])

            agent_message = Message(
                role="assistant",
                content=response_text if not tool_calls else None,
                tool_calls=tool_calls,
                cost=results[0].get("cost"),
                usage=results[0].get("usage"),
            )
            state = state.add_message(agent_message)

        return agent_message, state

    def _parse_tool_calls(self, result: Dict[str, Any]) -> Optional[List[ToolCall]]:
        """Parse tool calls from generation result.

        Args:
            result: Generation result dict

        Returns:
            List of ToolCall objects or None
        """
        if "tool_calls" not in result:
            return None

        tool_calls = []
        for tc in result["tool_calls"]:
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc["function"]["name"] if "function" in tc else tc["name"],
                arguments=tc["function"]["arguments"] if "function" in tc else tc["arguments"],
                requestor="assistant",
            ))

        return tool_calls if tool_calls else None

    @classmethod
    def is_stop(cls, message: Message) -> bool:
        """Check if message signals conversation end.

        Args:
            message: Message to check

        Returns:
            True if stop signal detected
        """
        if message.content is None:
            return False

        return AGENT_STOP in message.content

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed
        """
        self._seed = seed

    def stop(
        self,
        last_message: Optional[Message],
        state: AgentState,
    ) -> None:
        """Called when session ends. No-op for LLM agent.

        Args:
            last_message: Last message sent
            state: Final state
        """
        pass
