"""SGLang-backed user simulator for tau2-bench co-training.

Replaces the litellm-based UserSimulator inside AgentGymEnv with one
that routes generation to an SGLang endpoint, capturing token_ids and
logprobs for opponent trajectory construction.

This lets us co-train the user simulator alongside the agent: the agent
uses the standard rollout path (UserSimOrchestrator), while the opponent
(user simulator) is served by a separate SGLang engine whose weights
are updated by a separate training group.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.user.base import UserState, ValidUserInputMessage
from tau2.user.user_simulator import UserSimulator

logger = logging.getLogger(__name__)


class SGLangUserSimulator(UserSimulator):
    """UserSimulator that calls an SGLang endpoint instead of litellm.

    Captures token_ids and logprobs from each generation for opponent
    trajectory construction in co-training.
    """

    def __init__(
        self,
        sglang_url: str,
        tokenizer: Any,
        sampling_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sglang_url = sglang_url
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params or {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 1024,
        }
        # Captured outputs from each generation call
        self.generation_outputs: List[Dict[str, Any]] = []

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Generate response via SGLang instead of litellm."""
        # Update state with new message (same as parent)
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Build messages in OpenAI format (role-flipped for user sim perspective)
        messages = []
        for sys_msg in state.system_messages:
            messages.append({"role": "system", "content": sys_msg.content})
        for msg in state.flip_roles():
            if isinstance(msg, AssistantMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, UserMessage):
                messages.append({"role": "user", "content": msg.content})
            else:
                messages.append({"role": msg.role, "content": getattr(msg, "content", "")})

        # Tokenize with chat template (TITO)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        # Call SGLang endpoint
        response = requests.post(
            self.sglang_url,
            json={
                "input_ids": input_ids,
                "sampling_params": self.sampling_params,
                "return_logprob": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        output = response.json()

        meta = output.get("meta_info", {})
        token_logprobs = meta.get("output_token_logprobs", [])
        response_text = output["text"]

        # Strip EOS tokens
        if response_text.endswith("<|im_end|>"):
            response_text = response_text[:-10].strip()

        # Capture output for opponent trajectory construction
        self.generation_outputs.append({
            "text": response_text,
            "token_ids": [item[1] for item in token_logprobs],
            "logprobs": [item[0] for item in token_logprobs],
            "meta_info": meta,
            "input_ids": input_ids,
        })

        logger.debug(f"SGLang user sim response: {response_text[:100]}...")

        user_message = UserMessage(
            role="user",
            content=response_text,
        )
        return user_message, state
