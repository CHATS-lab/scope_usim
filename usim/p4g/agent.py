"""Persuader agent for Persuasion for Good."""

from typing import Any, Dict, List, Optional, Tuple

from usim.core.types import AgentState, Message
from usim.p4g.prompts import build_persuader_system_prompt


class PersuaderAgent:
    """Agent that plays the persuader role in P4G conversations.

    Simplified version of LLMAgent — no tool calls, no stop signal.
    Conversation ends via max_turns in the orchestrator.
    """

    def __init__(
        self,
        persona_text: str,
        word_limit: int = 50,
        num_turns: int = 10,
    ):
        self._system_prompt = build_persuader_system_prompt(persona_text, word_limit, num_turns)
        self._seed: Optional[int] = None

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> AgentState:
        system_message = Message(role="system", content=self._system_prompt)
        return AgentState(
            system_messages=[system_message],
            messages=message_history or [],
            metadata={},
        )

    def build_messages(self, state: AgentState) -> List[Dict[str, Any]]:
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
        agent_message = Message(role="assistant", content=text)
        state = state.add_message(agent_message)
        return agent_message, state

    @classmethod
    def is_stop(cls, message: Message) -> bool:
        return False

    def set_seed(self, seed: int) -> None:
        self._seed = seed

    def stop(self, last_message: Optional[Message], state: AgentState) -> None:
        pass
