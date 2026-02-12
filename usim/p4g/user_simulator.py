"""Persuadee user simulator for Persuasion for Good."""

from typing import Any, Dict, List, Optional, Tuple

from usim.core.types import Message, UserState
from usim.p4g.prompts import build_persuadee_system_prompt


class PersuadeeUserSimulator:
    """User simulator that plays the persuadee role in P4G conversations.

    Simplified version of LLMUserSimulator — no tool calls, no stop signal.
    Uses role-flipping so the LLM generates as "assistant" internally,
    then output is converted to "user" role.
    """

    def __init__(
        self,
        persona_text: str,
        word_limit: int = 50,
    ):
        self._system_prompt = build_persuadee_system_prompt(persona_text, word_limit)
        self._seed: Optional[int] = None

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> UserState:
        system_message = Message(role="system", content=self._system_prompt)
        return UserState(
            system_messages=[system_message],
            messages=message_history or [],
            metadata={},
        )

    def build_messages(self, state: UserState) -> List[Dict[str, Any]]:
        llm_messages = []
        for sys_msg in state.system_messages:
            llm_messages.append(sys_msg.to_dict())
        # Flip roles: agent messages become "user", persuadee messages become "assistant"
        for msg in state.flip_roles():
            llm_messages.append(msg.to_dict())
        return llm_messages

    def parse_response(
        self,
        text: str,
        state: UserState,
    ) -> Tuple[Message, UserState]:
        user_message = Message(role="user", content=text)
        state = state.add_message(user_message)
        return user_message, state

    @classmethod
    def is_stop(cls, message: Message) -> bool:
        return False

    def set_seed(self, seed: int) -> None:
        self._seed = seed

    def stop(self, last_message: Optional[Message], state: UserState) -> None:
        pass
