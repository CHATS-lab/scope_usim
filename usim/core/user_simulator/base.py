"""Base protocol for user simulator implementations."""

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from usim.core.types import Message, UserState


# Stop signals used by the user simulator
STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT_OF_SCOPE###"

STOP_SIGNALS = [STOP, TRANSFER, OUT_OF_SCOPE]


@runtime_checkable
class BaseUserSimulator(Protocol):
    """Protocol for user simulator implementations.

    A user simulator generates responses from the user's perspective in a
    simulated conversation with an agent. The simulator maintains state
    and can detect when the conversation should end.

    The orchestrator uses build_messages() to get prompt messages, calls
    model.generate_async() directly, then uses parse_response() to
    interpret the result.
    """

    @property
    def system_prompt(self) -> str:
        """Get the system prompt defining user behavior."""
        ...

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> UserState:
        """Initialize user state, optionally with existing history.

        Args:
            message_history: Optional list of prior messages

        Returns:
            Initial UserState for the session
        """
        ...

    def build_messages(
        self,
        state: UserState,
    ) -> List[Dict[str, Any]]:
        """Build messages for LLM generation.

        Returns the system prompt + role-flipped conversation history in OpenAI
        message format. The user simulator generates as "assistant" internally,
        so conversation roles are flipped.

        Args:
            state: Current user state

        Returns:
            List of message dicts for the LLM
        """
        ...

    def parse_response(
        self,
        text: str,
        state: UserState,
    ) -> Tuple[Message, UserState]:
        """Parse generation output into a Message and update state.

        Converts the LLM's "assistant" output into a "user" role message.

        Args:
            text: Generated text from the model
            state: Current user state

        Returns:
            Tuple of (user_message, updated_state)
        """
        ...

    def generate_next_message(
        self,
        agent_message: Message,
        state: UserState,
    ) -> Tuple[Message, UserState]:
        """Generate the user's response to an agent message.

        Convenience method that combines build_messages + generate + parse_response.

        Args:
            agent_message: The agent's message to respond to
            state: Current user state

        Returns:
            Tuple of (user_response, updated_state)
        """
        ...

    async def generate_next_message_async(
        self,
        agent_message: Message,
        state: UserState,
    ) -> Tuple[Message, UserState]:
        """Async version of generate_next_message.

        Args:
            agent_message: The agent's message to respond to
            state: Current user state

        Returns:
            Tuple of (user_response, updated_state)
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
        state: UserState,
    ) -> None:
        """Called when the session ends.

        Args:
            last_message: The last message sent (if any)
            state: Final user state
        """
        ...


def is_stop_signal(content: str) -> bool:
    """Check if content contains any stop signal.

    Args:
        content: Text content to check

    Returns:
        True if any stop signal is found
    """
    return any(signal in content for signal in STOP_SIGNALS)
