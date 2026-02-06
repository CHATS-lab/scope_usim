"""LLM-based user simulator implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from usim.core.model_adapter import ModelAdapter
from usim.core.types import Message, ToolCall, UserSimConfig, UserState
from usim.core.user_simulator.base import STOP_SIGNALS, is_stop_signal


logger = logging.getLogger(__name__)


# Default system prompt template for user simulation
DEFAULT_USER_SIM_SYSTEM_PROMPT = """You are simulating a user interacting with a customer service agent.

<simulation_guidelines>
- Stay in character as the user throughout the conversation
- Respond naturally based on the scenario provided
- If your issue is resolved satisfactorily, respond with ###STOP###
- If you need to transfer to another department, respond with ###TRANSFER###
- If the agent asks about something outside your scenario, respond with ###OUT_OF_SCOPE###
</simulation_guidelines>

<scenario>
{instructions}
</scenario>
"""


class LLMUserSimulator:
    """LLM-based user simulator.

    Uses an LLM to generate user responses based on a scenario/instructions.
    The simulator flips message roles when generating responses since it
    acts as the "user" but the LLM generates "assistant" responses.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: UserSimConfig,
        instructions: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize the LLM user simulator.

        Args:
            model: ModelAdapter for generation
            config: User simulator configuration
            instructions: User scenario/instructions
            system_prompt_template: Custom system prompt template
            tools: Optional tools the user can invoke
        """
        self.model = model
        self.config = config
        self.instructions = instructions or ""
        self._system_prompt_template = system_prompt_template or DEFAULT_USER_SIM_SYSTEM_PROMPT
        self.tools = tools
        self._seed: Optional[int] = None

    @property
    def system_prompt(self) -> str:
        """Get the formatted system prompt."""
        return self._system_prompt_template.format(instructions=self.instructions)

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> UserState:
        """Initialize user state with system prompt.

        Args:
            message_history: Optional prior message history

        Returns:
            Initialized UserState
        """
        system_message = Message(role="system", content=self.system_prompt)

        if message_history is None:
            message_history = []

        return UserState(
            system_messages=[system_message],
            messages=message_history,
            metadata={"instructions": self.instructions},
        )

    def build_messages(
        self,
        state: UserState,
    ) -> List[Dict[str, Any]]:
        """Build messages for LLM generation with flipped roles.

        Returns system prompt + role-flipped conversation history.
        The user simulator sees agent messages as "user" and its own
        messages as "assistant".

        Args:
            state: Current user state

        Returns:
            List of message dicts for the LLM
        """
        llm_messages = []
        for sys_msg in state.system_messages:
            llm_messages.append(sys_msg.to_dict())
        for msg in state.flip_roles():
            llm_messages.append(msg.to_dict())
        return llm_messages

    def parse_response(
        self,
        text: str,
        state: UserState,
    ) -> Tuple[Message, UserState]:
        """Parse generation output into a user Message and update state.

        Converts the LLM's "assistant" output into a "user" role message.

        Args:
            text: Generated text from the model
            state: Current user state

        Returns:
            Tuple of (user_message, updated_state)
        """
        # Parse tool calls from text if present
        tool_calls = self._parse_tool_calls({"text": text})

        user_message = Message(
            role="user",
            content=text if not tool_calls else None,
            tool_calls=tool_calls,
        )

        # Flip tool call requestor to "user"
        if user_message.tool_calls:
            for tc in user_message.tool_calls:
                tc.requestor = "user"

        state = state.add_message(user_message)
        return user_message, state

    def generate_next_message(
        self,
        agent_message: Message,
        state: UserState,
    ) -> Tuple[Message, UserState]:
        """Generate user response synchronously.

        Convenience method that combines build_messages + generate + parse_response.

        Args:
            agent_message: Agent's message to respond to
            state: Current user state

        Returns:
            Tuple of (user_message, updated_state)
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_next_message_async(agent_message, state)
        )

    async def generate_next_message_async(
        self,
        agent_message: Message,
        state: UserState,
    ) -> Tuple[Message, UserState]:
        """Generate user response asynchronously.

        Convenience method that combines build_messages + generate + parse_response.

        From user simulator's perspective:
        - Agent messages become "user" messages (the simulator sees agent as user)
        - User messages become "assistant" messages (the simulator generates these)

        Args:
            agent_message: Agent's message to respond to
            state: Current user state

        Returns:
            Tuple of (user_message, updated_state)
        """
        # Add the agent message to state
        state = state.add_message(agent_message)

        # Build messages for LLM with flipped roles
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
            logger.error(f"User simulator generation failed: {error_msg}")
            # Return stop message on error
            user_message = Message(role="user", content=STOP_SIGNALS[0])
            state = state.add_message(user_message)
        else:
            response_text = results[0]["text"]
            logger.debug(f"User simulator response: {response_text[:200]}...")

            # Parse tool calls if present (from response parsing)
            tool_calls = self._parse_tool_calls(results[0])

            user_message = Message(
                role="user",
                content=response_text if not tool_calls else None,
                tool_calls=tool_calls,
                cost=results[0].get("cost"),
                usage=results[0].get("usage"),
            )

            # Flip tool call requestor to "user"
            if user_message.tool_calls:
                for tc in user_message.tool_calls:
                    tc.requestor = "user"

            state = state.add_message(user_message)

        return user_message, state

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
                requestor="user",
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
        if message.is_tool_call():
            return False

        if message.content is None:
            return False

        return is_stop_signal(message.content)

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed
        """
        self._seed = seed

    def stop(
        self,
        last_message: Optional[Message],
        state: UserState,
    ) -> None:
        """Called when session ends. No-op for LLM simulator.

        Args:
            last_message: Last message sent
            state: Final state
        """
        pass
