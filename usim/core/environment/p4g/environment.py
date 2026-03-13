"""Persuasion for Good environment implementing BaseEnvironment protocol.

Pure text conversation (no tools). The environment manages the persuadee
(fixed opponent via OpenAI API) and computes donation-based reward.

Flow per step:
1. Agent (persuader) sends a message
2. Environment calls persuadee LLM (role-flipped conversation)
3. Extracts donation signals from persuadee response
4. Returns (persuadee_response, reward, terminated, truncated, info)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm

from usim.core.types import Message
from usim.p4g.prompts import build_persuadee_system_prompt, build_persuader_system_prompt
from usim.p4g.reward import compute_p4g_reward


logger = logging.getLogger(__name__)


class P4gEnvironment:
    """Persuasion for Good environment.

    Implements BaseEnvironment protocol for the orchestrator.
    Manages the persuadee (fixed opponent) via OpenAI-compatible API.
    No tools — pure text conversation.
    """

    def __init__(
        self,
        persuader_persona: str,
        persuadee_persona: str,
        word_limit: int = 50,
        num_turns: int = 10,
        persuadee_model: str = "gpt-5-mini",
        persuadee_base_url: str = "https://api.openai.com/v1",
        persuadee_api_key: Optional[str] = None,
        persuadee_max_tokens: int = 8192,
        conversation_id: str = "",
        persuadee_prompt_prefix: str = "",
    ):
        """Initialize P4G environment.

        Args:
            persuader_persona: Persuader persona text (with <persona> tags)
            persuadee_persona: Persuadee persona text (with <persona> tags)
            word_limit: Max words per response in prompts
            num_turns: Total conversation turns (each side gets num_turns // 2)
            persuadee_model: Model name for persuadee API calls
            persuadee_base_url: API base URL for persuadee
            persuadee_api_key: API key for persuadee
            persuadee_max_tokens: Max tokens for persuadee generation
            conversation_id: Conversation ID for tracking
            persuadee_prompt_prefix: Optional prefix prepended to persuadee system prompt
        """
        self._num_exchanges = num_turns // 2
        self._word_limit = word_limit
        self._num_turns = num_turns
        self._conversation_id = conversation_id

        # Build system prompts
        self._persuader_system = build_persuader_system_prompt(
            persuader_persona, word_limit, num_turns
        )
        self._persuadee_system = build_persuadee_system_prompt(
            persuadee_persona, word_limit, prompt_prefix=persuadee_prompt_prefix
        )

        # Persuadee model config for litellm
        # OpenRouter models need openrouter/ prefix; OpenAI models work as-is
        self._persuadee_model = persuadee_model
        if persuadee_base_url and "openrouter" in persuadee_base_url:
            if not persuadee_model.startswith("openrouter/"):
                self._persuadee_model = f"openrouter/{persuadee_model}"
        self._persuadee_api_key = persuadee_api_key
        self._persuadee_base_url = persuadee_base_url
        self._persuadee_max_tokens = persuadee_max_tokens

        # Internal state (reset in reset())
        self._persuadee_messages: List[Dict[str, Any]] = []
        self._all_messages: List[Dict[str, Any]] = []
        self._exchange_count = 0

    async def reset(
        self,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Reset environment and return initial state.

        Returns persuader's system prompt only (agent speaks first).
        No tools for P4G.
        """
        self._exchange_count = 0
        self._persuadee_messages = [
            {"role": "system", "content": self._persuadee_system}
        ]
        self._all_messages = []

        initial_messages = [{"role": "system", "content": self._persuader_system}]
        task_info = {
            "id": self._conversation_id,
            "domain": "p4g",
        }

        return initial_messages, None, task_info

    async def step(
        self, action: str
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Step environment with persuader's message.

        Calls persuadee API and returns response with donation reward.

        Args:
            action: Persuader's text message

        Returns:
            (persuadee_response, reward, terminated, truncated, info)
        """
        # Track persuader's message
        self._all_messages.append({"role": "assistant", "content": action})

        # Add to persuadee's conversation (role-flipped: persuader = "user")
        self._persuadee_messages.append({"role": "user", "content": action})

        # Call persuadee API
        persuadee_text = await self._call_persuadee()

        # Track persuadee's response
        self._persuadee_messages.append({"role": "assistant", "content": persuadee_text})
        self._all_messages.append({"role": "user", "content": persuadee_text})

        self._exchange_count += 1

        # Compute reward from donation signals in persuadee messages
        reward = self._compute_reward()

        # Terminate after num_exchanges OR when donation is made
        donated = reward > 0.0
        terminated = self._exchange_count >= self._num_exchanges or donated

        info = {
            "exchange": self._exchange_count,
            "num_exchanges": self._num_exchanges,
            "donated": donated,
        }

        return persuadee_text, reward, terminated, False, info

    def parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse agent response — no tools in P4G, just plain text."""
        return {"normal_text": response_text, "calls": []}

    @property
    def prompt_postprocess_fn(self) -> Optional[Callable[[str], str]]:
        """No prompt postprocessing needed for P4G."""
        return None

    async def _call_persuadee(self) -> str:
        """Call persuadee LLM via litellm."""
        try:
            kwargs = {
                "model": self._persuadee_model,
                "messages": self._persuadee_messages,
                "max_tokens": self._persuadee_max_tokens,
                "temperature": 0.7,
            }
            if self._persuadee_api_key:
                kwargs["api_key"] = self._persuadee_api_key

            logger.debug(
                f"Calling persuadee: model={self._persuadee_model}, "
                f"messages={len(self._persuadee_messages)}, max_tokens={self._persuadee_max_tokens}"
            )

            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content or ""

            if not content.strip():
                logger.warning(
                    f"Persuadee returned empty response: model={self._persuadee_model}, "
                    f"finish_reason={response.choices[0].finish_reason}, "
                    f"usage={response.usage}"
                )

            return content
        except Exception as e:
            logger.error(f"Persuadee API call failed ({self._persuadee_model}): {e}")
            raise

    def _compute_reward(self) -> float:
        """Compute reward from donation signals in persuadee messages."""
        msg_objects = [
            Message(role=m["role"], content=m.get("content", ""))
            for m in self._all_messages
        ]
        return compute_p4g_reward(msg_objects)
