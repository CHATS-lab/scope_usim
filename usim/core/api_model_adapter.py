"""LiteLLM-based model adapter for fixed opponent models.

This adapter wraps litellm for generation while using a local tokenizer for
token tracking. Used when the user simulator (opponent) is a fixed API model
like gpt-5-mini while the agent is the trainable SGLang model.

Since the opponent's tokens are never trained (loss_mask=0), logprobs are
set to 0.0 — only the local tokenizer is needed for token-in-token-out tracking.

litellm handles provider-specific quirks (max_tokens vs max_completion_tokens,
OpenRouter routing, etc.) automatically.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import litellm

logger = logging.getLogger(__name__)


class OpenAIModelAdapter:
    """ModelAdapter using litellm for generation.

    Uses a shared local tokenizer for template application / token tracking,
    and litellm for actual text generation across providers.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer: Any,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        default_sampling_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the API model adapter.

        Args:
            model_name: Model name for API calls (e.g., "gpt-5-mini")
            tokenizer: Local tokenizer for token tracking (shared with trainable model)
            base_url: API base URL
            api_key: API key
            default_sampling_params: Default sampling parameters
        """
        # litellm needs provider prefix for non-OpenAI models
        # OpenRouter: openrouter/google/gemini-3-flash-preview
        # OpenAI: gpt-5-mini (litellm recognizes natively)
        self._model_name = model_name
        if base_url and "openrouter" in base_url:
            if not model_name.startswith("openrouter/"):
                self._model_name = f"openrouter/{model_name}"

        self._tokenizer = tokenizer
        self._api_key = api_key
        self._base_url = base_url
        self._default_params = default_sampling_params or {}

    @property
    def tokenizer(self) -> Any:
        """Get the shared local tokenizer."""
        return self._tokenizer

    def apply_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Apply chat template using local tokenizer."""
        kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            kwargs["tools"] = tools

        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Synchronously generate responses."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_async(messages, input_ids, temperature, top_p, max_tokens, **kwargs)
        )

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        input_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate responses using litellm.

        Logprobs are set to 0.0 since this is a fixed opponent model
        (loss_mask=0, not trained).
        """
        kwargs.pop("tools", None)

        try:
            call_kwargs = {
                "model": self._model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if self._api_key:
                call_kwargs["api_key"] = self._api_key

            response = await litellm.acompletion(**call_kwargs)

            text = response.choices[0].message.content or ""
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)

            return [{
                "text": text,
                "token_ids": token_ids,
                "logprobs": [0.0] * len(token_ids),
                "prompt_token_ids": input_ids or [],
            }]

        except Exception as e:
            logger.error(f"API generation failed ({self._model_name}): {e}")
            return [{
                "text": "",
                "token_ids": [],
                "logprobs": [],
                "prompt_token_ids": input_ids or [],
                "error": str(e),
            }]


def create_openai_model_adapter(
    model_name: str,
    tokenizer: Any,
    base_url: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
) -> OpenAIModelAdapter:
    """Factory function to create an OpenAI-compatible model adapter."""
    return OpenAIModelAdapter(
        model_name=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
        api_key=api_key,
    )
