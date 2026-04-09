"""Persuasion for Good environment implementing BaseEnvironment protocol.

Pure text conversation (no tools). The environment manages the persuadee
(fixed opponent via OpenAI API or pluggable generate_fn) and computes
donation-based reward.

Flow per step:
1. Agent (persuader) sends a message
2. Environment calls persuadee LLM (role-flipped conversation)
3. Extracts donation signals from persuadee response
4. Returns (persuadee_response, reward, terminated, truncated, info)

For co-training / self-play modes, an `opponent_generate_fn` replaces the
litellm API call. The orchestrator reads `last_opponent_output` to get
the opponent's token_ids and logprobs for dual-trajectory tracking.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm

from usim.core.types import Message
from usim.core.vs_parsing import (
    extract_json_object,
    sample_candidate,
    validate_candidates,
)
from usim.p4g.prompts import build_persuadee_system_prompt, build_persuader_system_prompt
from usim.p4g.reward import compute_p4g_reward


logger = logging.getLogger(__name__)


class P4gEnvironment:
    """Persuasion for Good environment.

    Implements BaseEnvironment protocol for the orchestrator.
    Manages the persuadee (fixed opponent) via OpenAI-compatible API
    or a pluggable opponent_generate_fn for co-training/self-play.
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
        opponent_generate_fn: Optional[Callable] = None,
        verbalized_sampling: bool = False,
        vs_num_samples: int = 5,
        vs_method: str = "prob",
        vs_max_retries: int = 2,
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
            opponent_generate_fn: Optional callable for co-training/self-play.
                Signature: async (input_ids: List[int], sampling_params: Dict) -> Dict
                Returns {"text", "token_ids", "logprobs", "meta_info"}.
                When set, replaces litellm API calls for the persuadee.
            verbalized_sampling: Enable Verbalized Sampling (arxiv:2510.01171)
                for the persuadee. The persuadee system prompt is augmented
                with an instruction to emit a JSON object of N candidate
                replies with verbalized probabilities; we then parse the
                JSON and sample one candidate per turn. Only applies to the
                litellm API path (``_call_persuadee_via_api``). Ignored when
                ``opponent_generate_fn`` is set (co-training path).
            vs_num_samples: Number of candidate replies the persuadee should
                generate per turn when ``verbalized_sampling`` is enabled.
            vs_method: ``"prob"`` (weight by verbalized probability, default)
                or ``"random"`` (uniform over candidates).
            vs_max_retries: How many times to retry if the persuadee's
                response can't be parsed as a VS JSON object. After all
                retries are exhausted we fall back to using the raw response
                verbatim.
        """
        self._num_exchanges = num_turns // 2
        self._word_limit = word_limit
        self._num_turns = num_turns
        self._conversation_id = conversation_id

        # Verbalized Sampling config
        self._verbalized_sampling = bool(verbalized_sampling)
        self._vs_num_samples = int(vs_num_samples)
        self._vs_method = str(vs_method)
        self._vs_max_retries = int(vs_max_retries)

        # Build system prompts
        self._persuader_system = build_persuader_system_prompt(
            persuader_persona, word_limit, num_turns
        )
        self._persuadee_system = build_persuadee_system_prompt(
            persuadee_persona,
            word_limit,
            prompt_prefix=persuadee_prompt_prefix,
            verbalized_sampling=self._verbalized_sampling,
            vs_num_samples=self._vs_num_samples,
            vs_method=self._vs_method,
        )

        # Pluggable opponent generation (for co-training/self-play)
        self._opponent_generate_fn = opponent_generate_fn

        # Persuadee model config for litellm (used when opponent_generate_fn is None)
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

        # Last opponent output for co-training trajectory tracking.
        # Set after each _call_persuadee() when using opponent_generate_fn.
        self.last_opponent_output: Optional[Dict[str, Any]] = None

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
        self.last_opponent_output = None

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

        Calls persuadee (API or opponent_generate_fn) and returns response
        with donation reward.

        Args:
            action: Persuader's text message

        Returns:
            (persuadee_response, reward, terminated, truncated, info)
        """
        # Track persuader's message
        self._all_messages.append({"role": "assistant", "content": action})

        # Add to persuadee's conversation (role-flipped: persuader = "user")
        self._persuadee_messages.append({"role": "user", "content": action})

        # Call persuadee (API or pluggable generate_fn)
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

    @property
    def persuadee_system_prompt(self) -> str:
        """Return the persuadee system prompt (for opponent tokenization)."""
        return self._persuadee_system

    @property
    def persuadee_messages(self) -> List[Dict[str, Any]]:
        """Return the persuadee's conversation history (for opponent tokenization)."""
        return self._persuadee_messages

    async def _call_persuadee(self) -> str:
        """Call persuadee LLM via litellm or opponent_generate_fn."""
        if self._opponent_generate_fn is not None:
            return await self._call_persuadee_via_generate_fn()
        return await self._call_persuadee_via_api()

    async def _call_persuadee_via_generate_fn(self) -> str:
        """Call persuadee via pluggable generate function (co-training/self-play).

        The opponent_generate_fn receives input_ids and sampling_params,
        same interface as the agent's generate_fn. The caller (co-training
        orchestrator) is responsible for tokenizing the persuadee's messages
        and passing them as input_ids.

        We store the full output in self.last_opponent_output so the
        co-training orchestrator can extract token_ids and logprobs.
        """
        # The co-training orchestrator handles tokenization externally
        # and wraps it into the generate_fn. We just need to call it
        # with the persuadee messages and get back text + tokens.
        assert self._opponent_generate_fn is not None
        output = await self._opponent_generate_fn(
            self._persuadee_messages,
            {"temperature": 0.7, "max_new_tokens": self._persuadee_max_tokens},
        )
        self.last_opponent_output = output
        return output["text"]

    async def _call_persuadee_via_api(self) -> str:
        """Call persuadee LLM via litellm (original API path).

        When ``verbalized_sampling`` is enabled, delegates to
        ``_call_persuadee_via_api_vs`` which parses the JSON response and
        samples one candidate. Otherwise returns the raw litellm response.
        """
        if self._verbalized_sampling:
            return await self._call_persuadee_via_api_vs()

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

            self.last_opponent_output = None
            return content
        except Exception as e:
            logger.error(f"Persuadee API call failed ({self._persuadee_model}): {e}")
            raise

    async def _call_persuadee_via_api_vs(self) -> str:
        """Call persuadee LLM and sample one response via Verbalized Sampling.

        The persuadee system prompt (built at init time) instructs the LLM
        to emit a JSON object with ``{responses: [{text, probability}, ...]}``.
        We parse the JSON, sample one candidate (weighted by probability
        for ``method="prob"``, uniform for ``method="random"``), and return
        the sampled text as if it were the original response.

        On parse failure we retry up to ``vs_max_retries`` times, then fall
        back to returning the raw response verbatim so a transient parse
        error does not abort the whole episode.
        """
        kwargs = {
            "model": self._persuadee_model,
            "messages": self._persuadee_messages,
            "max_tokens": self._persuadee_max_tokens,
            "temperature": 0.7,
        }
        if self._persuadee_api_key:
            kwargs["api_key"] = self._persuadee_api_key

        last_content = ""
        for attempt in range(self._vs_max_retries + 1):
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as e:
                logger.error(
                    f"Persuadee VS API call failed ({self._persuadee_model}): {e}"
                )
                raise

            content = response.choices[0].message.content or ""
            last_content = content

            if not content.strip():
                logger.warning(
                    f"Persuadee VS returned empty response: "
                    f"model={self._persuadee_model}, "
                    f"finish_reason={response.choices[0].finish_reason}"
                )
                continue

            parsed = extract_json_object(content)
            if parsed is None:
                logger.warning(
                    f"VS parse failed (attempt {attempt + 1}/"
                    f"{self._vs_max_retries + 1}): {content[:200]!r}"
                )
                continue

            candidates = validate_candidates(parsed)
            if not candidates:
                logger.warning(
                    f"VS JSON had no valid candidates (attempt {attempt + 1}/"
                    f"{self._vs_max_retries + 1}): {content[:200]!r}"
                )
                continue

            sampled_text = sample_candidate(candidates, method=self._vs_method)
            logger.debug(
                f"P4G VS sampled 1/{len(candidates)} candidates "
                f"(method={self._vs_method}): {sampled_text[:80]!r}"
            )
            self.last_opponent_output = None
            return sampled_text

        # All retries exhausted. Fall back to the raw response verbatim so
        # we don't abort the episode on a transient parse failure. The
        # donation-marker regex will still fire on well-formed raw text.
        logger.warning(
            "P4G VS parsing failed after %d attempts; falling back to raw response",
            self._vs_max_retries + 1,
        )
        self.last_opponent_output = None
        return last_content

    def _compute_reward(self) -> float:
        """Compute reward from donation signals in persuadee messages."""
        msg_objects = [
            Message(role=m["role"], content=m.get("content", ""))
            for m in self._all_messages
        ]
        return compute_p4g_reward(msg_objects)
