"""Base protocol for agent implementations."""

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from usim.core.types import AgentState, Message


# Stop signal used by agents
AGENT_STOP = "###STOP###"


@runtime_checkable
class BaseAgent(Protocol):
    """Protocol for agent implementations.

    An agent interacts with users and environments, generating responses
    and tool calls to accomplish tasks.

    The orchestrator uses build_messages() to get prompt messages, calls
    model.generate_async() directly, then uses parse_response() to
    interpret the result.
    """

    @property
    def system_prompt(self) -> str:
        """Get the system prompt defining agent behavior."""
        ...

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> AgentState:
        """Initialize agent state, optionally with existing history.

        Args:
            message_history: Optional list of prior messages

        Returns:
            Initial AgentState for the session
        """
        ...

    def build_messages(
        self,
        state: AgentState,
    ) -> List[Dict[str, Any]]:
        """Build messages for LLM generation.

        Returns the system prompt + conversation history in OpenAI message format,
        ready to pass to model.apply_template() or model.generate_async().

        Args:
            state: Current agent state

        Returns:
            List of message dicts for the LLM
        """
        ...

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
        ...

    def generate_next_message(
        self,
        user_message: Optional[Message],
        state: AgentState,
    ) -> Tuple[Message, AgentState]:
        """Generate the agent's response to a user message.

        Convenience method that combines build_messages + generate + parse_response.

        Args:
            user_message: The user's message to respond to (None for first turn)
            state: Current agent state

        Returns:
            Tuple of (agent_response, updated_state)
        """
        ...

    async def generate_next_message_async(
        self,
        user_message: Optional[Message],
        state: AgentState,
    ) -> Tuple[Message, AgentState]:
        """Async version of generate_next_message.

        Args:
            user_message: The user's message to respond to (None for first turn)
            state: Current agent state

        Returns:
            Tuple of (agent_response, updated_state)
        """
        ...

    @classmethod
    def is_stop(cls, message: Message) -> bool:
        """Check if a message signals conversation end.

        Args:
            message: Message to check

        Returns:
            True if message contains a stop signal
        """
        ...

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        ...

    def stop(
        self,
        last_message: Optional[Message],
        state: AgentState,
    ) -> None:
        """Called when the session ends.

        Args:
            last_message: The last message sent (if any)
            state: Final agent state
        """
        ...
