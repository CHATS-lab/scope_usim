"""Verbalized Sampling user simulator for tau2-bench.

Implements the Verbalized Sampling (VS) prompting strategy from
https://arxiv.org/abs/2510.01171. Instead of asking the LLM for a single
response, VS asks for N plausible responses along with their verbalized
probabilities, then samples one from the resulting distribution. This
mitigates mode collapse in instruction-tuned user simulators.

The prompt template mirrors the one used in the public P4G verbalized-sampling
reference implementation (external/persuasion_simulation) for consistency with
our own 0313 P4G baselines, adapted for the tau2-bench setting.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from tau2.data_model.message import MultiToolMessage, ToolCall, UserMessage
from tau2.user.base import UserState, ValidUserInputMessage
from tau2.user.user_simulator import UserSimulator
from tau2.utils.llm_utils import generate

from usim.core.environment.tau2.vs_parsing import (
    extract_json_object,
    sample_candidate,
    validate_candidates,
)


# Prompt appended to the base system prompt when VS is enabled. We keep this
# close to the persuasion_simulation reference but swap the P4G-specific
# framing ("chat partner") for tau2's "customer service agent" context.
_VS_PROB_INSTRUCTION = """
## Response Diversity Instructions

Instead of committing to a single response, generate {n} plausible responses
you might naturally give as the user in your current situation, along with an
estimated probability for each. Responses should be diverse — cover different
phrasings, tones, and information you might reveal.

Return ONLY a JSON object with the key "responses" (a list of dictionaries,
each with 'text' and 'probability'):
- 'text': the response string only, no explanation
- 'probability': a number in [0.0, 1.0] representing how likely you are to
  give this particular response (the probabilities should roughly sum to 1.0)

Example format:
{{"responses": [{{"text": "Sure, I'll wait.", "probability": 0.5}}, {{"text": "How much longer will this take?", "probability": 0.3}}, {{"text": "OK, thanks.", "probability": 0.2}}]}}

Output ONLY the JSON object, no markdown fences, no explanations, no extra
text before or after.
""".strip()


_VS_RANDOM_INSTRUCTION = """
## Response Diversity Instructions

Instead of committing to a single response, generate {n} plausible responses
you might naturally give as the user in your current situation. Responses
should be diverse — cover different phrasings, tones, and information you
might reveal.

Return ONLY a JSON object with the key "responses" (a list of dictionaries,
each with a 'text' field):
- 'text': the response string only, no explanation

Example format:
{{"responses": [{{"text": "Sure, I'll wait."}}, {{"text": "How much longer will this take?"}}, {{"text": "OK, thanks."}}]}}

Output ONLY the JSON object, no markdown fences, no explanations.
""".strip()


class VerbalizedSamplingUserSimulator(UserSimulator):
    """UserSimulator that uses Verbalized Sampling (arxiv:2510.01171).

    The user simulator's system prompt is augmented with an instruction to
    generate N candidate responses and their verbalized probabilities as a
    JSON object. We then parse the JSON, sample one candidate (weighted by
    probability for ``method="prob"`` or uniformly for ``method="random"``),
    and return it as a normal ``UserMessage``.

    On parse failure we retry up to ``max_retries`` times, then fall back to
    treating the raw model output as the user response (best-effort).
    """

    def __init__(
        self,
        *args,
        vs_num_samples: int = 5,
        vs_method: str = "prob",
        vs_max_retries: int = 2,
        **kwargs,
    ):
        if vs_method not in ("prob", "random"):
            raise ValueError(
                f"vs_method must be 'prob' or 'random', got {vs_method!r}"
            )
        super().__init__(*args, **kwargs)
        self.vs_num_samples = vs_num_samples
        self.vs_method = vs_method
        self.vs_max_retries = vs_max_retries

    @property
    def system_prompt(self) -> str:
        """Append the VS instruction to the base system prompt.

        Returning a computed property (rather than storing on __init__) keeps
        us compatible with the monkey-patching trick used by
        ``_ConfiguredAgentGymEnv`` and lets us re-read ``vs_num_samples`` if
        it's ever tuned at runtime.
        """
        base = super().system_prompt
        if self.vs_method == "prob":
            instruction = _VS_PROB_INSTRUCTION.format(n=self.vs_num_samples)
        else:
            instruction = _VS_RANDOM_INSTRUCTION.format(n=self.vs_num_samples)
        return base + "\n\n" + instruction

    def _sample_candidate(self, candidates: List[dict]) -> str:
        """Thin wrapper around the pure ``sample_candidate`` helper."""
        return sample_candidate(candidates, method=self.vs_method)

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """VS generation: ask for N candidates, sample one, return as UserMessage.

        Falls back to the raw response on parse failure so a transient JSON
        error doesn't abort the episode.
        """
        # Update state with incoming agent/tool message (mirrors base impl)
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.flip_roles()

        assistant_message = None
        candidates: Optional[List[dict]] = None

        for attempt in range(self.vs_max_retries + 1):
            assistant_message = generate(
                model=self.llm,
                messages=messages,
                tools=self.tools,
                **self.llm_args,
            )

            # If the model routed to a tool call instead of producing a JSON
            # response, fall through to the non-VS path: respect the tool
            # call and skip VS sampling for this turn. Rare in tau2.
            if assistant_message.tool_calls:
                break

            parsed = extract_json_object(assistant_message.content or "")
            if parsed is not None:
                candidates = validate_candidates(parsed)
                if candidates:
                    break

            logger.warning(
                f"VS parse failed (attempt {attempt + 1}/{self.vs_max_retries + 1}): "
                f"{(assistant_message.content or '')[:200]!r}"
            )

        assert assistant_message is not None, "generate() loop did not assign a message"

        # Branch 1: tool call → fall back to base-simulator behaviour
        if assistant_message.tool_calls:
            user_message = UserMessage(
                role="user",
                content=assistant_message.content or "",
                cost=assistant_message.cost,
                usage=assistant_message.usage,
                raw_data=assistant_message.raw_data,
            )
            user_message.tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    requestor="user",
                )
                for tc in assistant_message.tool_calls
            ]
            state.messages.append(user_message)
            return user_message, state

        # Branch 2: successful VS parse → sample one candidate
        if candidates:
            sampled_text = self._sample_candidate(candidates)
            logger.debug(
                f"VS sampled 1/{len(candidates)} candidates "
                f"(method={self.vs_method}): {sampled_text[:80]!r}"
            )
            user_message = UserMessage(
                role="user",
                content=sampled_text,
                cost=assistant_message.cost,
                usage=assistant_message.usage,
                raw_data=assistant_message.raw_data,
            )
            state.messages.append(user_message)
            return user_message, state

        # Branch 3: parse failed after all retries → use raw content verbatim.
        # Better to keep the episode going with a possibly-malformed response
        # than to crash.
        logger.warning(
            "VS parsing failed after %d retries; falling back to raw response",
            self.vs_max_retries + 1,
        )
        user_message = UserMessage(
            role="user",
            content=(assistant_message.content or "").strip(),
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )
        state.messages.append(user_message)
        return user_message, state
